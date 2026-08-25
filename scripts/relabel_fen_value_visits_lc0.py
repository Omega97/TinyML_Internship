#!/usr/bin/env python3
"""Overwrite STM ``wdl`` (and White-POV ``value``) in existing slice JSON via Lc0.

Does **not** re-parse the Lichess dump. Keeps ``fen`` and ``visits``. Faster than
``lichess_dump_to_fen_value_visits.py`` because extract is skipped; Lc0 eval is
still one call per unique EPD.

``wdl`` is ``[W, D, L]`` from the side to move. ``value`` is White-POV
``±(W-L)``, both rounded to 3 decimal digits.

Lc0 ``go depth N`` is MCTS *average* tree depth (not Stockfish αβ). Default
``--depth 1``. Pass a slice JSON, a slice folder, or the whole
``fen_value_visits/`` parent.

Examples::

    py -3.12 -u scripts/relabel_fen_value_visits_lc0.py path/to/slice.json --in-place
    py -3.12 -u scripts/relabel_fen_value_visits_lc0.py path/to/slice.json --depth 2 --in-place
    py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits --in-place
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
from tinymlinternship.data.wdl import dumps_labeled_json, labeled_payload, wdl_from_row
from tinymlinternship.engine.eval_lc0 import Lc0Teacher, go_command

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


def uci_go(*, depth: int, nodes: int | None) -> str:
    """``depth 0`` is value-head only (``go nodes 1``): one NN eval per position."""
    if depth == 0:
        return go_command(nodes=1 if nodes is None else nodes)
    return go_command(depth=depth, nodes=nodes)


def default_output(input_path: Path, *, depth: int, nodes: int | None) -> Path:
    extra = f"_depth{depth}"
    if nodes is not None:
        extra += f"_nodes{nodes}"
    return input_path.with_name(input_path.stem + extra + ".json")


SKIP_JSON_NAMES = {"features.meta.json", "meta.json"}


def discover_targets(path: Path) -> list[Path]:
    """One JSON, a slice folder, or the parent of all slice folders."""
    path = _resolve(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    named = path / f"{path.name}.json"
    if named.is_file():
        return [named]
    found: list[Path] = []
    for json_path in sorted(path.rglob("fen_value_visits_*.json")):
        if json_path.name in SKIP_JSON_NAMES:
            continue
        if json_path.parent.name != json_path.stem:
            continue
        found.append(json_path)
    return found


def _as_row(obj: object) -> dict | None:
    if not isinstance(obj, dict) or "fen" not in obj:
        return None
    visits = obj.get("visits", 1)
    try:
        visits_i = int(visits)
    except (TypeError, ValueError):
        visits_i = 1
    row: dict = {"fen": str(obj["fen"]), "visits": visits_i}
    if "wdl" in obj:
        row["wdl"] = obj["wdl"]
    if "value" in obj:
        try:
            row["value"] = float(obj["value"])
        except (TypeError, ValueError):
            row["value"] = 0.0
    elif "wdl" not in row:
        row["value"] = 0.0
    return row


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


def evaluate_wdl(teacher: Lc0Teacher, fen: str) -> tuple[float, float, float]:
    board = chess.Board(fen)
    return teacher.evaluate_stm_wdl(board)


def write_payload(rows: list[dict], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(dumps_labeled_json(rows), encoding="utf-8")
    parquet_path = json_path.with_suffix(".parquet")
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)


def slim_row(row: dict, wdl: tuple[float, float, float]) -> dict:
    return labeled_payload(str(row["fen"]), wdl, int(row.get("visits", 1)))


def relabel_file(
    input_path: Path,
    output: Path,
    *,
    teacher: Lc0Teacher,
    limit: int | None,
    checkpoint_every: int,
    progress: bool,
    resume: bool,
) -> dict:
    rows_in, how = load_rows(input_path)
    if how == "recovered":
        print(
            f"warning: {input_path.name} is not valid JSON; recovered {len(rows_in)} records from fen/value/visits keys",
            file=sys.stderr,
        )
    elif how == "jsonl":
        print(f"loaded {len(rows_in)} JSONL records from {input_path.name}", flush=True)
    if limit is not None:
        rows_in = rows_in[:limit]
    if not rows_in:
        raise ValueError(f"No rows to label in {input_path}")

    out_rows: list[dict] = []
    start = 0
    parquet_out = output.with_suffix(".parquet")
    if resume and parquet_out.exists():
        prev = pd.read_parquet(parquet_out)
        out_rows = prev.to_dict(orient="records")
        start = len(out_rows)
        if start > len(rows_in):
            raise ValueError(
                f"checkpoint {parquet_out} has {start} rows > input {len(rows_in)}; refusing resume"
            )
        print(f"resume at {start:,}/{len(rows_in):,} from {parquet_out}", flush=True)

    skipped = 0

    def _flush() -> None:
        write_payload(out_rows, output)

    with _progress_bar(
        total=len(rows_in),
        desc=f"{input_path.stem} {teacher.go}",
        disable=not progress,
        initial=start,
    ) as bar:
        for idx in range(start, len(rows_in)):
            row = rows_in[idx]
            try:
                wdl = evaluate_wdl(teacher, str(row["fen"]))
            except (ValueError, chess.InvalidFenError) as exc:
                skipped += 1
                print(f"warn row {idx}: {exc}", file=sys.stderr)
                wdl = wdl_from_row(row)
            out_rows.append(slim_row(row, wdl))
            bar.update(1)
            if checkpoint_every and (idx + 1) % checkpoint_every == 0:
                _flush()
    _flush()
    return {
        "input": str(input_path),
        "output": str(output),
        "parquet": str(parquet_out),
        "count": len(out_rows),
        "skipped": skipped,
        "go": teacher.go,
        "value_min": min(r["value"] for r in out_rows),
        "value_max": max(r["value"] for r in out_rows),
        "wdl": "STM [W, D, L]",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Overwrite STM wdl in existing fen-value-visits JSON via Lc0 go depth N."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Slice JSON/parquet, slice folder, or fen_value_visits/ parent",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Lc0 UCI go depth (default: 1). 0 = one NN eval (go nodes 1). Depth 2 sees short mates.",
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
    if args.depth < 0:
        print("--depth must be >= 0 (0 = go nodes 1, one eval per position)", file=sys.stderr)
        return 1
    if args.nodes is not None and args.nodes < 1:
        print("--nodes must be >= 1", file=sys.stderr)
        return 1

    targets = discover_targets(input_path)
    if not targets:
        print(f"no fen-value-visits JSON in {input_path}", file=sys.stderr)
        return 1
    if args.output is not None and len(targets) > 1:
        print("--output is only valid for a single JSON file.", file=sys.stderr)
        return 1

    in_place = bool(args.in_place) or (input_path.is_dir() and args.output is None)
    go = uci_go(depth=args.depth, nodes=args.nodes)
    weights = LC0_NETWORK_PRESETS[args.network] if args.network else LC0_NETWORK_DEFAULT
    weights = weights.resolve() if weights.exists() else weights
    checkpoint_every = max(0, int(args.checkpoint_every))

    print(f"{len(targets)} file(s), {go}", flush=True)
    with Lc0Teacher(
        weights=str(weights),
        backend=args.backend,
        go=go,
        smart_pruning_factor=0.0,
    ) as teacher:
        summaries: list[dict] = []
        for target in targets:
            if in_place:
                output = target.with_suffix(".json")
            elif args.output is not None:
                output = _resolve(args.output)
                if output.suffix.lower() == ".parquet":
                    output = output.with_suffix(".json")
            else:
                output = default_output(target, depth=args.depth, nodes=args.nodes)
            try:
                summary = relabel_file(
                    target,
                    output,
                    teacher=teacher,
                    limit=args.limit,
                    checkpoint_every=checkpoint_every,
                    progress=bool(args.progress_every),
                    resume=not in_place,
                )
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            summary["network"] = weights.name
            summaries.append(summary)
            print(json.dumps(summary, indent=2), flush=True)
    if len(summaries) > 1:
        print(json.dumps({"files": len(summaries), "go": go}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
