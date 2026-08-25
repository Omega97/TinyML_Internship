#!/usr/bin/env python3
"""Stream a Lichess monthly ``.pgn.zst`` dump → unique {fen, wdl, value, visits} JSON.

Takes 1-based game numbers ``n`` (inclusive) and ``m`` (exclusive): games
``[n, m)``. Skip is ``[Event `` header count (no chess parse). Unique EPDs are
kept unless ``--max-unique`` is set; later games in the range only increment
visits for positions already in that set.

Teacher: Lc0 STM WDL probabilities ``[W, D, L]`` plus White-POV ``value = ±(W-L)``.
Then encodes the slice to ``features.npz`` (same as ``encode_slice_features.py``).
Output under ``data/processed/board_eval/fen_value_visits/<stem>/``:

    fen_value_visits_lichess_db_standard_rated_2026-07_<n>-<m>.json
    features.npz

Default ``--dropout 0.90`` keeps each ply independently with probability 0.10
(decorrelates consecutive positions from the same game). Slice stem gets
``_d90``. ``--dropout 0`` keeps every ply and omits the suffix.

``--max-draw P`` keeps only positions with STM draw probability ``D < P``
(decisive slice). Stem gets ``_draw30`` when ``P=0.30``. Applied after Lc0
labels; extract/labeled parquet still hold every unique EPD.

``--max-moves N`` keeps startpos plus the first N full moves (2N half-moves).
Stem gets ``_m10`` when N=10. ``N=0`` keeps the whole game.

Example (first 10 games, stem ``…_0-10_d90``)::

    py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10
    py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10 --max-draw 0.30
    py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 100000 105000 --dropout 0 --max-moves 10
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

import chess
import chess.pgn
import pandas as pd
import zstandard
from tqdm import tqdm

from tinymlinternship.config.settings import (
    LICHESS_DUMPS_DIR,
    LC0_NETWORK_DEFAULT,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)
from tinymlinternship.data.board_store import (
    BOARD_EVAL_DIR_NAME,
    FEN_VALUE_VISITS_DIR_NAME,
    fen_value_visits_slice_path,
)
from tinymlinternship.data.wdl import dumps_labeled_json, labeled_payload, stm_wdl_from_white_value
from tinymlinternship.engine.eval_lc0 import Lc0Teacher


DEFAULT_DUMP = LICHESS_DUMPS_DIR / "lichess_db_standard_rated_2026-07.pgn.zst"
PROGRESS_INTERVAL_S = 1.0


def _progress_bar(
    *,
    total: int | None,
    desc: str,
    unit: str,
    disable: bool,
    initial: int = 0,
) -> tqdm:
    """Bar that redraws about once a second (tqdm ETA from rate)."""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        unit_scale=True,
        unit_divisor=1000,
        mininterval=PROGRESS_INTERVAL_S,
        maxinterval=PROGRESS_INTERVAL_S,
        miniters=1,
        initial=initial,
        disable=disable,
        dynamic_ncols=True,
        file=sys.stderr,
        smoothing=0.1,
        leave=True,
    )


def dump_month_id(path: Path) -> str:
    name = path.name
    for suffix in (".pgn.zst", ".pgn", ".zst"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def game_range_to_skip_max(n: int, m: int) -> tuple[int, int]:
    """Half-open ``[n, m)`` → (skip_games, max_games).

    ``n >= 1`` is 1-based (``1 11`` = first 10 games). ``n == 0`` is the start of
    the dump so the slice stem can be ``0-m`` (``0 10`` = first 10 games, same
    as ``1 11``).
    """
    if n < 0:
        raise ValueError("n must be >= 0 (0 = start of dump; 1-based otherwise)")
    if m <= n:
        raise ValueError("m must be > n (range is [n, m), m exclusive)")
    if n == 0:
        return 0, m
    return n - 1, m - n


def dropout_suffix(dropout: float) -> str:
    """``0.90`` → ``_d90``; ``0`` → empty (old slice names)."""
    if dropout <= 0.0:
        return ""
    return f"_d{int(round(float(dropout) * 100))}"


def max_draw_suffix(max_draw: float | None) -> str:
    """``0.30`` → ``_draw30``; ``None`` / ``>=1`` → empty."""
    if max_draw is None or float(max_draw) >= 1.0:
        return ""
    if float(max_draw) < 0.0:
        raise ValueError("max_draw must be >= 0")
    return f"_draw{int(round(float(max_draw) * 100))}"


def max_moves_suffix(max_moves: int) -> str:
    """``10`` → ``_m10``; ``0`` → empty."""
    if max_moves <= 0:
        return ""
    return f"_m{int(max_moves)}"


def slice_stem_extra(
    *,
    dropout: float = 0.0,
    max_moves: int = 0,
    max_draw: float | None = None,
) -> str:
    return f"{dropout_suffix(dropout)}{max_moves_suffix(max_moves)}{max_draw_suffix(max_draw)}"


def slice_json_name(
    dump: Path,
    n: int,
    m: int,
    *,
    dropout: float = 0.0,
    max_moves: int = 0,
    max_draw: float | None = None,
) -> str:
    return (
        f"fen_value_visits_{dump_month_id(dump)}_{n}-{m}"
        f"{slice_stem_extra(dropout=dropout, max_moves=max_moves, max_draw=max_draw)}.json"
    )


def slice_extract_name(
    dump: Path,
    n: int,
    m: int,
    *,
    dropout: float = 0.0,
    max_moves: int = 0,
) -> str:
    extra = f"{dropout_suffix(dropout)}{max_moves_suffix(max_moves)}"
    return f"{dump_month_id(dump)}_{n}-{m}{extra}_extract.parquet"


def slice_labeled_name(
    dump: Path,
    n: int,
    m: int,
    *,
    dropout: float = 0.0,
    max_moves: int = 0,
) -> str:
    extra = f"{dropout_suffix(dropout)}{max_moves_suffix(max_moves)}"
    return f"{dump_month_id(dump)}_{n}-{m}{extra}.parquet"


def _epd_key(board: chess.Board) -> str:
    return hashlib.sha256(board.epd().encode("utf-8")).hexdigest()


class _ChainText(io.TextIOBase):
    """Unread prefix + remaining PGN stream (for skip-games)."""

    def __init__(self, prefix: str, rest: io.TextIOBase) -> None:
        self._buf = prefix
        self._rest = rest

    def readline(self, size: int = -1) -> str:  # noqa: ARG002
        if self._buf:
            nl = self._buf.find("\n")
            if nl >= 0:
                line, self._buf = self._buf[: nl + 1], self._buf[nl + 1 :]
                return line
            more = self._rest.readline()
            line = self._buf + more
            self._buf = ""
            return line
        return self._rest.readline()

    def read(self, n: int = -1) -> str:
        if n == 0:
            return ""
        if n is None or n < 0:
            out, self._buf = self._buf, ""
            return out + self._rest.read()
        if self._buf:
            take, self._buf = self._buf[:n], self._buf[n:]
            if len(take) == n:
                return take
            return take + self._rest.read(n - len(take))
        return self._rest.read(n)

    def close(self) -> None:
        self._rest.close()


def _open_pgn(path: Path) -> io.TextIOBase:
    if path.suffix == ".zst" or path.name.endswith(".pgn.zst"):
        raw = path.open("rb")
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        reader = dctx.stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        text._zst_raw = raw  # noqa: SLF001 — keep file handle alive
        text._zst_reader = reader  # noqa: SLF001
        return text
    return path.open(encoding="utf-8", errors="replace")


def skip_pgn_games(
    handle: io.TextIOBase,
    skip: int,
    *,
    progress_every: int = 100_000,
) -> io.TextIOBase:
    """Advance ``handle`` so the next parse starts at 0-based game ``skip``."""
    if skip <= 0:
        return handle
    needle = "[Event "
    seen = 0
    carry = ""
    t0 = time.perf_counter()
    rest: io.TextIOBase | None = None
    disable = not progress_every
    with _progress_bar(total=skip, desc="skip games", unit="game", disable=disable) as bar:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                raise EOFError(f"EOF after {seen:,} [Event headers; wanted skip={skip:,}")
            data = carry + chunk
            start = 0
            while True:
                idx = data.find(needle, start)
                if idx < 0:
                    break
                seen += 1
                if seen == skip + 1:
                    if bar.n < skip:
                        bar.update(skip - bar.n)
                    rest = _ChainText(data[idx:], handle)
                    break
                start = idx + 1
            if rest is not None:
                break
            skipped = min(seen, skip)
            if skipped > bar.n:
                bar.update(skipped - bar.n)
            carry = data[-(len(needle) - 1) :]
    if progress_every:
        print(f"skipped {skip:,} games in {time.perf_counter() - t0:.1f}s", flush=True)
    assert rest is not None
    return rest


def collect_unique(
    dump: Path,
    *,
    skip_games: int,
    max_games: int,
    max_unique: int,
    progress_every: int,
    stop_when_unique_full: bool = False,
    include_startpos: bool = True,
    dropout: float = 0.0,
    seed: int = 0,
    max_moves: int = 0,
) -> tuple[list[dict], dict]:
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if max_moves < 0:
        raise ValueError("max_moves must be >= 0")
    store: dict[str, dict] = {}
    games = 0
    plies = 0
    broken = 0
    considered = 0
    dropped = 0
    t0 = time.perf_counter()
    unlimited = max_unique is None or max_unique <= 0
    rng = random.Random(int(seed))
    ply_cap = 2 * int(max_moves) if max_moves > 0 else None

    def _count(board: chess.Board) -> None:
        nonlocal considered, dropped
        considered += 1
        if dropout > 0.0 and rng.random() < dropout:
            dropped += 1
            return
        key = _epd_key(board)
        rec = store.get(key)
        if rec is not None:
            rec["visits"] += 1
        elif unlimited or len(store) < max_unique:
            store[key] = {"fen": board.fen(), "visits": 1}

    class Visitor(chess.pgn.BaseVisitor):
        def begin_game(self) -> None:
            self.board = chess.Board()
            self.ok = True
            self.done = False
            if include_startpos:
                _count(self.board)

        def begin_variation(self) -> object:
            return chess.pgn.SKIP

        def visit_move(self, board: chess.Board, move: chess.Move) -> None:
            nonlocal plies
            if not self.ok or self.done:
                return
            try:
                self.board.push(move)
            except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
                self.ok = False
                return
            if ply_cap is not None and self.board.ply() > ply_cap:
                self.done = True
                return
            plies += 1
            _count(self.board)

        def result(self) -> bool:
            return self.ok

    handle = _open_pgn(dump)
    disable = not progress_every
    try:
        handle = skip_pgn_games(handle, skip_games, progress_every=progress_every)
        with _progress_bar(
            total=max_games,
            desc="extract games",
            unit="game",
            disable=disable,
        ) as bar:
            while games < max_games:
                try:
                    ok = chess.pgn.read_game(handle, Visitor=Visitor)
                except Exception as exc:  # noqa: BLE001
                    broken += 1
                    print(f"warn parse game {games}: {exc}", file=sys.stderr)
                    continue
                if ok is None:
                    break
                games += 1
                if not ok:
                    broken += 1
                bar.set_postfix(
                    unique=len(store),
                    plies=plies,
                    kept=considered - dropped,
                    broken=broken,
                    refresh=False,
                )
                bar.update(1)
                if (
                    stop_when_unique_full
                    and not unlimited
                    and len(store) >= max_unique
                ):
                    print(
                        f"unique cap {max_unique:,} reached after {games:,} games "
                        f"({plies:,} plies); stopping parse",
                        flush=True,
                    )
                    break
    finally:
        handle.close()

    rows = sorted(store.values(), key=lambda r: (-int(r["visits"]), str(r["fen"])))
    stats = {
        "dump": str(dump),
        "games_read": games,
        "skip_games": skip_games,
        "plies_seen": plies,
        "games_broken": broken,
        "unique": len(rows),
        "visits_sum": int(sum(int(r["visits"]) for r in rows)),
        "max_games": max_games,
        "max_unique": max_unique,
        "stop_when_unique_full": stop_when_unique_full,
        "include_startpos": include_startpos,
        "dropout": float(dropout),
        "seed": int(seed),
        "plies_considered": considered,
        "plies_dropped": dropped,
        "plies_kept": considered - dropped,
        "max_moves": int(max_moves),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    return rows, stats


def label_extract(
    extract: Path,
    labeled: Path,
    *,
    batch: int,
    network: Path,
    progress: bool = True,
    force: bool = False,
) -> int:
    df = pd.read_parquet(extract)
    n = len(df)
    if (
        not force
        and "expected_reward" in df.columns
        and df["expected_reward"].notna().all()
        and all(col in df.columns for col in ("wdl_win", "wdl_draw", "wdl_loss"))
    ):
        print(f"already labeled: {labeled}")
        return n
    start = 0
    if labeled.exists() and not force:
        prev = pd.read_parquet(labeled)
        start = len(prev)
        print(f"resume labels at {start:,}/{n:,}")
        if start >= n:
            return start
        df_done = prev
    else:
        df_done = None

    teacher_name = network.name
    labeled.parent.mkdir(parents=True, exist_ok=True)
    with Lc0Teacher(weights=str(network)) as teacher:
        with _progress_bar(
            total=n,
            desc="label positions",
            unit="pos",
            disable=not progress,
            initial=start,
        ) as bar:
            while start < n:
                end = min(n, start + batch)
                chunk = df.iloc[start:end]
                wdls: list[tuple[float, float, float]] = []
                rewards: list[float] = []
                for fen in chunk["fen"].astype(str).tolist():
                    board = chess.Board(fen)
                    wdl = teacher.evaluate_stm_wdl(board)
                    payload = labeled_payload(fen, wdl, 1)
                    wdls.append(tuple(payload["wdl"]))
                    rewards.append(float(payload["value"]))
                    bar.update(1)
                piece = chunk.copy()
                piece["wdl_win"] = [row[0] for row in wdls]
                piece["wdl_draw"] = [row[1] for row in wdls]
                piece["wdl_loss"] = [row[2] for row in wdls]
                piece["expected_reward"] = rewards
                piece["teacher_network"] = teacher_name
                piece["source"] = "lichess"
                df_done = piece if df_done is None else pd.concat([df_done, piece], ignore_index=True)
                df_done.to_parquet(labeled, index=False)
                start = end
    return n


def wdl_from_labeled_row(row: dict) -> tuple[float, float, float]:
    if all(key in row and row[key] is not None for key in ("wdl_win", "wdl_draw", "wdl_loss")):
        return (float(row["wdl_win"]), float(row["wdl_draw"]), float(row["wdl_loss"]))
    return stm_wdl_from_white_value(str(row["fen"]), float(row["expected_reward"]))


def write_json(
    labeled: Path,
    json_path: Path,
    *,
    max_draw: float | None = None,
) -> int:
    """Write JSON/parquet. ``max_draw`` keeps rows with STM ``D < max_draw``."""
    df = pd.read_parquet(labeled)
    payload = []
    dropped = 0
    for row in df.sort_values(["visits", "fen"], ascending=[False, True]).to_dict(
        orient="records"
    ):
        wdl = wdl_from_labeled_row(row)
        if max_draw is not None and wdl[1] >= float(max_draw):
            dropped += 1
            continue
        payload.append(labeled_payload(str(row["fen"]), wdl, int(row["visits"])))
    if not payload:
        raise ValueError(
            f"no rows left after max-draw filter (labeled {len(df):,}, dropped {dropped:,})"
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(dumps_labeled_json(payload), encoding="utf-8")
    pq = json_path.with_suffix(".parquet")
    pd.DataFrame(payload).to_parquet(pq, index=False)
    extra = ""
    if max_draw is not None:
        extra = f" (kept D<{max_draw:g}, dropped {dropped:,} draws)"
    print(f"JSON {len(payload):,} → {json_path} + {pq.name}{extra}")
    return len(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lichess dump games [n, m) → {fen, wdl, value, visits} JSON + features.npz"
    )
    parser.add_argument(
        "n",
        type=int,
        help="First game (0 = start of dump; 1-based if n>=1), inclusive",
    )
    parser.add_argument("m", type=int, help="End game number (exclusive)")
    parser.add_argument("--input", type=Path, default=DEFAULT_DUMP)
    parser.add_argument(
        "--max-unique",
        type=int,
        default=0,
        help="Cap unique EPDs (0 = keep all from games [n, m))",
    )
    parser.add_argument(
        "--stop-when-unique-full",
        action="store_true",
        help="Stop parsing once --max-unique EPDs are collected (visits from that prefix only)",
    )
    parser.add_argument("--no-startpos", action="store_true", help="Do not record the initial FEN")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.90,
        help="Drop each ply independently with this probability (default 0.90 → keep 1/10, stem _d90)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for ply dropout (default 0)",
    )
    parser.add_argument(
        "--max-draw",
        type=float,
        default=None,
        help="Keep only positions with STM draw proba D < P (decisive slice; stem _draw30 if P=0.30)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=0,
        help="Keep startpos + first N full moves (2N plies). 0 = whole game. 10 → stem _m10",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="0 disables progress bars; any positive value enables ~1s bars with ETA",
    )
    parser.add_argument("--batch", type=int, default=20_000, help="Label checkpoint size")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-label", action="store_true")
    parser.add_argument(
        "--skip-encode",
        action="store_true",
        help="Do not write features.npz after the JSON (default: encode 844 + STM WDL)",
    )
    parser.add_argument(
        "--force-label",
        action="store_true",
        help="Re-run Lc0 even if a labeled parquet already exists (needed for WDL rebuild)",
    )
    parser.add_argument(
        "--extract",
        type=Path,
        default=None,
        help="Extract parquet (default: data/raw/lichess/<dump>_<n>-<m>_extract.parquet)",
    )
    parser.add_argument(
        "--labeled",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path under processed/board_eval/fen_value_visits/",
    )
    args = parser.parse_args(argv)

    try:
        skip_games, max_games = game_range_to_skip_max(args.n, args.m)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.dropout < 0.0 or args.dropout >= 1.0:
        print("--dropout must be in [0, 1)", file=sys.stderr)
        return 2
    if args.max_draw is not None and args.max_draw < 0.0:
        print("--max-draw must be >= 0", file=sys.stderr)
        return 2
    if args.max_moves < 0:
        print("--max-moves must be >= 0", file=sys.stderr)
        return 2

    dump = args.input if args.input.is_absolute() else (PROJECT_ROOT / args.input)
    extract = args.extract or (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "lichess"
        / slice_extract_name(
            dump, args.n, args.m, dropout=args.dropout, max_moves=args.max_moves
        )
    )
    if args.extract and not args.extract.is_absolute():
        extract = PROJECT_ROOT / args.extract
    labeled = args.labeled or (
        PROCESSED_DATA_DIR
        / "labeled"
        / slice_labeled_name(
            dump, args.n, args.m, dropout=args.dropout, max_moves=args.max_moves
        )
    )
    if args.labeled and not Path(args.labeled).is_absolute():
        labeled = PROJECT_ROOT / args.labeled
    json_path = args.output or fen_value_visits_slice_path(
        PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME,
        slice_json_name(
            dump,
            args.n,
            args.m,
            dropout=args.dropout,
            max_moves=args.max_moves,
            max_draw=args.max_draw,
        ),
    )
    if args.output and not Path(args.output).is_absolute():
        json_path = PROJECT_ROOT / args.output

    dump = dump.resolve()
    extract = extract.resolve()
    labeled = labeled.resolve()
    json_path = json_path.resolve()

    if not args.skip_extract:
        if not dump.is_file():
            print(f"dump not found: {dump}", file=sys.stderr)
            return 1
        print(
            f"extracting games [{args.n:,}, {args.m:,}) from {dump} "
            f"(skip={skip_games:,}, count={max_games:,}, dropout={args.dropout:g}, "
            f"max_moves={args.max_moves}) …",
            flush=True,
        )
        rows, stats = collect_unique(
            dump,
            skip_games=skip_games,
            max_games=max_games,
            max_unique=args.max_unique,
            progress_every=args.progress_every,
            stop_when_unique_full=args.stop_when_unique_full,
            include_startpos=not args.no_startpos,
            dropout=args.dropout,
            seed=args.seed,
            max_moves=args.max_moves,
        )
        stats["n"] = args.n
        stats["m"] = args.m
        extract.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(extract, index=False)
        stats_path = extract.with_name(extract.stem + ".stats.json")
        try:
            stats["extract"] = str(extract.relative_to(PROJECT_ROOT))
        except ValueError:
            stats["extract"] = str(extract)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"extract {len(rows):,} unique → {extract}", flush=True)
        print(json.dumps(stats, indent=2), flush=True)

    if not extract.is_file():
        print(f"extract missing: {extract}", file=sys.stderr)
        return 1

    if not args.skip_label:
        print(f"labeling {extract} …", flush=True)
        label_extract(
            extract,
            labeled,
            batch=args.batch,
            network=LC0_NETWORK_DEFAULT.resolve(),
            progress=bool(args.progress_every),
            force=bool(args.force_label),
        )

    if not labeled.is_file():
        print(f"labeled missing: {labeled} (use --skip-label only after labels exist)", file=sys.stderr)
        return 1

    try:
        write_json(labeled, json_path, max_draw=args.max_draw)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not args.skip_encode:
        from encode_slice_features import encode_slice_folder

        encode_slice_folder(
            json_path.parent,
            rebuild=True,
            progress=bool(args.progress_every),
        )
    try:
        rel = json_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = json_path
    print(f"done → {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
