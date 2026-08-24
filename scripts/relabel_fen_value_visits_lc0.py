#!/usr/bin/env python3
"""Recompute ``value`` in a {fen, value, visits} JSON at Lc0 ``go depth N``.

Keeps ``fen`` and ``visits``. ``value`` is White-POV expected reward in [-1, +1],
rounded to 3 decimal digits.

Lc0 ``go depth N`` is MCTS *average* tree depth (not Stockfish αβ). ``--depth``
is required.

The input JSON may be a list of objects, JSONL, parquet, or a slightly broken
list (missing ``{`` / extra commas). Invalid FENs are skipped with a warning.

Examples::

    py -3.12 -u scripts/relabel_fen_value_visits_lc0.py path/to/slice.json --depth 2
    py -3.12 -u scripts/relabel_fen_value_visits_lc0.py path/to/slice.json --depth 4 --in-place
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import pandas as pd
from tqdm import tqdm

from tinymlinternship.config.settings import (
    LC0_NETWORK_DEFAULT,
    LC0_NETWORK_PRESETS,
    PROJECT_ROOT,
)
from tinymlinternship.engine.eval_lc0 import (
    Lc0Teacher,
    go_command,
    wdl_to_expected_reward_white,
)

VALUE_DIGITS = 3
PROGRESS_INTERVAL_S = 1.0
CHECKPOINT_EVERY = 50

FEN_RE = re.compile(r'"fen"\s*:\s*"([^"]+)"')
VALUE_RE = re.compile(r'"value"\s*:\s*([-+0-9.eE]+)')
VISITS_RE = re.compile(r'"visits"\s*:\s*(-?\d+)')


def _progress_bar(*, total: int, desc: str, disable: bool, initial: int = 0) -> tqdm:
    return tqdm(
        total=total,
        desc=desc,
        unit="pos",
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


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def default_output(input_path: Path, *, depth: int, nodes: int | None) -> Path:
    extra = f"_depth{depth}"
    if nodes is not None:
        extra += f"_nodes{nodes}"
    return input_path.with_name(input_path.stem + extra + ".json")


def _as_row(obj: object) -> dict | None:
    if not isinstance(obj, dict) or "fen" not in obj:
        return None
    visits = obj.get("visits", 1)
    value = obj.get("value", 0.0)
    try:
        visits_i = int(visits)
    except (TypeError, ValueError):
        visits_i = 1
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        value_f = 0.0
    return {"fen": str(obj["fen"]), "value": value_f, "visits": visits_i}


def _first(pattern: re.Pattern[str], after: str, before: str) -> re.Match[str] | None:
    return pattern.search(after) or pattern.search(before)


def recover_rows(text: str) -> list[dict]:
    """Pull {fen, value, visits} even if a brace or comma is missing."""
    rows: list[dict] = []
    for match in FEN_RE.finditer(text):
        after = text[match.end() : match.end() + 240]
        before = text[max(0, match.start() - 80) : match.start()]
        val = _first(VALUE_RE, after, before)
        vis = _first(VISITS_RE, after, before)
        rows.append(
            {
                "fen": match.group(1),
                "value": float(val.group(1)) if val else 0.0,
                "visits": int(vis.group(1)) if vis else 1,
            }
        )
    return rows


def load_json_rows(path: Path) -> tuple[list[dict], str]:
    """Return (rows, how). ``how`` is ``json``, ``jsonl``, or ``recovered``."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        rows = [r for r in (_as_row(x) for x in payload) if r is not None]
        if rows:
            return rows, "json"
    if isinstance(payload, dict) and _as_row(payload) is not None:
        row = _as_row(payload)
        assert row is not None
        return [row], "json"

    jsonl: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip().rstrip(",")
        if stripped in ("", "[", "]"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        row = _as_row(obj)
        if row is not None:
            jsonl.append(row)
    if jsonl:
        return jsonl, "jsonl"

    recovered = recover_rows(text)
    if recovered:
        return recovered, "recovered"
    raise ValueError(
        f"{path} is not a fen-value-visits JSON list (and no fen/value/visits records could be recovered)"
    )


def load_rows(path: Path) -> tuple[list[dict], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        rows = [r for r in (_as_row(x) for x in df.to_dict(orient="records")) if r is not None]
        return rows, "parquet"
    return load_json_rows(path)


def _terminal_reward(board: chess.Board) -> float | None:
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_threefold_repetition()
        or board.can_claim_fifty_moves()
    ):
        return 0.0
    return None


def evaluate_value(teacher: Lc0Teacher, fen: str) -> float:
    board = chess.Board(fen)
    terminal = _terminal_reward(board)
    if terminal is not None:
        return float(terminal)
    win, draw, loss = teacher.evaluate_wdl(board)
    return float(wdl_to_expected_reward_white(board, win, draw, loss))


def write_payload(rows: list[dict], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    parquet_path = json_path.with_suffix(".parquet")
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)


def slim_row(row: dict, value: float) -> dict:
    return {
        "fen": str(row["fen"]),
        "value": round(float(value), VALUE_DIGITS),
        "visits": int(row.get("visits", 1)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute every value in a fen-value-visits JSON via Lc0 go depth N."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Target JSON/parquet with {fen, value, visits} rows",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="Lc0 UCI go depth (required). Depth 2 sees the sample mate-in-2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (parquet twin beside it). Default: <stem>_depthN.json",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input JSON (and write a parquet twin next to it)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Optional extra UCI node cap (stop at depth or nodes, whichever first)",
    )
    parser.add_argument(
        "--network",
        choices=sorted(LC0_NETWORK_PRESETS),
        default=None,
        help=f"Lc0 weights preset (default: {LC0_NETWORK_DEFAULT.name})",
    )
    parser.add_argument("--backend", default="blas")
    parser.add_argument("--limit", type=int, default=None, help="Label only the first N rows")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="Write output every N newly labeled rows (0 = only at the end)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="0 disables the progress bar; any positive value enables ~1s bars",
    )
    args = parser.parse_args(argv)

    input_path = _resolve(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    if args.in_place and args.output is not None:
        print("Use either --in-place or --output, not both.", file=sys.stderr)
        return 1
    if args.depth < 1:
        print("--depth must be >= 1", file=sys.stderr)
        return 1
    if args.nodes is not None and args.nodes < 1:
        print("--nodes must be >= 1", file=sys.stderr)
        return 1

    try:
        rows_in, how = load_rows(input_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if how == "recovered":
        print(
            f"warning: {input_path.name} is not valid JSON; recovered {len(rows_in)} records from fen/value/visits keys",
            file=sys.stderr,
        )
    elif how == "jsonl":
        print(f"loaded {len(rows_in)} JSONL records from {input_path.name}", flush=True)

    if args.limit is not None:
        rows_in = rows_in[: args.limit]
    if not rows_in:
        print("No rows to label.", file=sys.stderr)
        return 1

    if args.in_place:
        output = input_path.with_suffix(".json")
    elif args.output is not None:
        output = _resolve(args.output)
        if output.suffix.lower() == ".parquet":
            output = output.with_suffix(".json")
    else:
        output = default_output(input_path, depth=args.depth, nodes=args.nodes)

    go = go_command(depth=args.depth, nodes=args.nodes)
    weights = LC0_NETWORK_PRESETS[args.network] if args.network else LC0_NETWORK_DEFAULT
    weights = weights.resolve() if weights.exists() else weights

    out_rows: list[dict] = []
    start = 0
    parquet_out = output.with_suffix(".parquet")
    if parquet_out.exists() and not args.in_place:
        prev = pd.read_parquet(parquet_out)
        out_rows = prev.to_dict(orient="records")
        start = len(out_rows)
        if start > len(rows_in):
            print(
                f"checkpoint {parquet_out} has {start} rows > input {len(rows_in)}; refusing resume",
                file=sys.stderr,
            )
            return 1
        print(f"resume at {start:,}/{len(rows_in):,} from {parquet_out}", flush=True)

    checkpoint_every = max(0, int(args.checkpoint_every))
    skipped = 0

    def _flush() -> None:
        write_payload(out_rows, output)

    with Lc0Teacher(
        weights=str(weights),
        backend=args.backend,
        go=go,
        smart_pruning_factor=0.0,
    ) as teacher:
        with _progress_bar(
            total=len(rows_in),
            desc=f"lc0 {go}",
            disable=not args.progress_every,
            initial=start,
        ) as bar:
            for idx in range(start, len(rows_in)):
                row = rows_in[idx]
                try:
                    value = evaluate_value(teacher, str(row["fen"]))
                except (ValueError, chess.InvalidFenError) as exc:
                    skipped += 1
                    print(f"warn row {idx}: {exc}", file=sys.stderr)
                    value = float(row.get("value", 0.0))
                out_rows.append(slim_row(row, value))
                bar.update(1)
                if checkpoint_every and (idx + 1) % checkpoint_every == 0:
                    _flush()

    _flush()
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output),
                "parquet": str(parquet_out),
                "count": len(out_rows),
                "skipped": skipped,
                "go": go,
                "network": weights.name,
                "value_min": min(r["value"] for r in out_rows),
                "value_max": max(r["value"] for r in out_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
