#!/usr/bin/env python3
"""Stream a Lichess monthly ``.pgn.zst`` dump → unique {fen, value, visits} JSON.

Takes 1-based inclusive game numbers ``n`` and ``m``. Games ``n`` through ``m``
are parsed (skip is ``[Event `` header count, no chess parse). Unique EPDs are
kept unless ``--max-unique`` is set; later games in the range only increment
visits for positions already in that set.

Teacher: Lc0 WDL → White-POV expected reward. Output (and parquet twin) under
``data/processed/board_eval/fen_value_visits/``:

    fen_value_visits_lichess_db_standard_rated_2026-07_<n>-<m>.json

Example (first 10 games)::

    py -3.12 scripts/lichess_dump_to_fen_value_visits.py 1 10
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.engine.eval_lc0 import Lc0Teacher, wdl_to_expected_reward_white


def _terminal_wdl_and_reward(board: chess.Board) -> tuple[tuple[int, int, int], float] | None:
    if board.is_checkmate():
        return (0, 0, 1000), (-1.0 if board.turn == chess.WHITE else 1.0)
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_threefold_repetition()
        or board.can_claim_fifty_moves()
    ):
        return (0, 1000, 0), 0.0
    return None

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
    """1-based inclusive ``n``..``m`` → (skip_games, max_games)."""
    if n < 1:
        raise ValueError("n must be >= 1 (1-based game numbers)")
    if m < n:
        raise ValueError("m must be >= n")
    return n - 1, m - n + 1


def slice_json_name(dump: Path, n: int, m: int) -> str:
    return f"fen_value_visits_{dump_month_id(dump)}_{n}-{m}.json"


def slice_extract_name(dump: Path, n: int, m: int) -> str:
    return f"{dump_month_id(dump)}_{n}-{m}_extract.parquet"


def slice_labeled_name(dump: Path, n: int, m: int) -> str:
    return f"{dump_month_id(dump)}_{n}-{m}.parquet"


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
) -> tuple[list[dict], dict]:
    store: dict[str, dict] = {}
    games = 0
    plies = 0
    broken = 0
    t0 = time.perf_counter()
    unlimited = max_unique is None or max_unique <= 0

    def _count(board: chess.Board) -> None:
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
            if include_startpos:
                _count(self.board)

        def begin_variation(self) -> object:
            return chess.pgn.SKIP

        def visit_move(self, board: chess.Board, move: chess.Move) -> None:
            nonlocal plies
            if not self.ok:
                return
            try:
                self.board.push(move)
            except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
                self.ok = False
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
                bar.set_postfix(unique=len(store), plies=plies, broken=broken, refresh=False)
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
) -> int:
    df = pd.read_parquet(extract)
    n = len(df)
    if "expected_reward" in df.columns and df["expected_reward"].notna().all():
        print(f"already labeled: {labeled}")
        return n
    start = 0
    if labeled.exists():
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
                rewards: list[float] = []
                for fen in chunk["fen"].astype(str).tolist():
                    board = chess.Board(fen)
                    terminal = _terminal_wdl_and_reward(board)
                    if terminal is not None:
                        (_w, _d, _l), reward = terminal
                    else:
                        win, draw, loss = teacher.evaluate_wdl(board)
                        reward = wdl_to_expected_reward_white(board, win, draw, loss)
                    rewards.append(float(reward))
                    bar.update(1)
                piece = chunk.copy()
                piece["expected_reward"] = rewards
                piece["teacher_network"] = teacher_name
                piece["source"] = "lichess"
                df_done = piece if df_done is None else pd.concat([df_done, piece], ignore_index=True)
                df_done.to_parquet(labeled, index=False)
                start = end
    return n


def write_json(labeled: Path, json_path: Path) -> None:
    df = pd.read_parquet(labeled)
    payload = [
        {
            "fen": str(row["fen"]),
            "value": float(row["expected_reward"]),
            "visits": int(row["visits"]),
        }
        for row in df.sort_values(["visits", "fen"], ascending=[False, True]).to_dict(
            orient="records"
        )
    ]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pq = json_path.with_suffix(".parquet")
    pd.DataFrame(payload).to_parquet(pq, index=False)
    print(f"JSON {len(payload):,} → {json_path} + {pq.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lichess dump games n..m → fen-value-visits JSON")
    parser.add_argument("n", type=int, help="First game number (1-based, inclusive)")
    parser.add_argument("m", type=int, help="Last game number (1-based, inclusive)")
    parser.add_argument("--input", type=Path, default=DEFAULT_DUMP)
    parser.add_argument(
        "--max-unique",
        type=int,
        default=0,
        help="Cap unique EPDs (0 = keep all from games n..m)",
    )
    parser.add_argument(
        "--stop-when-unique-full",
        action="store_true",
        help="Stop parsing once --max-unique EPDs are collected (visits from that prefix only)",
    )
    parser.add_argument("--no-startpos", action="store_true", help="Do not record the initial FEN")
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

    dump = args.input if args.input.is_absolute() else (PROJECT_ROOT / args.input)
    extract = args.extract or (
        PROJECT_ROOT / "data" / "raw" / "lichess" / slice_extract_name(dump, args.n, args.m)
    )
    if args.extract and not args.extract.is_absolute():
        extract = PROJECT_ROOT / args.extract
    labeled = args.labeled or (PROCESSED_DATA_DIR / "labeled" / slice_labeled_name(dump, args.n, args.m))
    if args.labeled and not Path(args.labeled).is_absolute():
        labeled = PROJECT_ROOT / args.labeled
    json_path = args.output or (
        PROCESSED_DATA_DIR
        / BOARD_EVAL_DIR_NAME
        / FEN_VALUE_VISITS_DIR_NAME
        / slice_json_name(dump, args.n, args.m)
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
            f"extracting games {args.n:,}–{args.m:,} from {dump} "
            f"(skip={skip_games:,}, count={max_games:,}) …",
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
        )

    if not labeled.is_file():
        print(f"labeled missing: {labeled} (use --skip-label only after labels exist)", file=sys.stderr)
        return 1

    write_json(labeled, json_path)
    try:
        rel = json_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = json_path
    print(f"done → {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
