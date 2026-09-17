#!/usr/bin/env python3
"""Overwrite STM ``wdl`` (and White-POV ``value``) in existing slice JSON via Lc0.

Fully robust, production-grade rewrite optimized for Nvidia DGX GPUs. Features
pre-validation of FEN strings, fault-tolerant worker execution with automatic 
engine process recovery, and seamless checkpoint resumption.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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
SKIP_JSON_NAMES = {"features.meta.json", "meta.json"}


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
    if depth == 0:
        return go_command(nodes=1 if nodes is None else nodes)
    return go_command(depth=depth, nodes=nodes)


def default_output(input_path: Path, *, depth: int, nodes: int | None) -> Path:
    extra = f"_depth{depth}"
    if nodes is not None:
        extra += f"_nodes{nodes}"
    return input_path.with_name(input_path.stem + extra + ".json")


def discover_targets(path: Path) -> list[Path]:
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
        f"{path} is not a valid fen-value-visits JSON list and records could not be recovered."
    )


def load_rows(path: Path) -> tuple[list[dict], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        rows = [r for r in (_as_row(x) for x in df.to_dict(orient="records")) if r is not None]
        return rows, "parquet"
    return load_json_rows(path)


def write_payload(rows: list[dict], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(dumps_labeled_json(rows), encoding="utf-8")
    parquet_path = json_path.with_suffix(".parquet")
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)


def slim_row(row: dict, wdl: tuple[float, float, float]) -> dict:
    return labeled_payload(str(row["fen"]), wdl, int(row.get("visits", 1)))


def process_chunk(args: tuple) -> tuple:
    """Worker process payload evaluating a chunk of FENs with fault-tolerant engine recovery."""
    chunk, weights_path, backend, go = args
    results = []
    skipped = 0
    errors = []
    
    teacher = None
    
    def _init_teacher() -> Lc0Teacher:
        t = Lc0Teacher(weights=weights_path, backend=backend, go=go, smart_pruning_factor=0.0)
        t.start()
        return t

    try:
        teacher = _init_teacher()
    except Exception as exc:
        for idx, row in chunk:
            skipped += 1
            errors.append((idx, f"Critical: Failed to launch Lc0 engine: {exc}"))
            results.append((idx, slim_row(row, wdl_from_row(row))))
        return results, skipped, errors

    for idx, row in chunk:
        fen_str = str(row["fen"])
        
        # Step 1: Pre-validate FEN string locally before touching the engine pipe
        try:
            board = chess.Board(fen_str)
        except Exception as fen_exc:
            skipped += 1
            errors.append((idx, f"Invalid FEN format: {fen_exc}"))
            results.append((idx, slim_row(row, wdl_from_row(row))))
            continue

        # Step 2: Evaluate via Lc0 with built-in retry and process recovery
        success = False
        wdl = None
        for attempt in range(2):
            try:
                wdl = teacher.evaluate_stm_wdl(board)
                success = True
                break
            except Exception as engine_exc:
                # If pipe breaks or engine faults, attempt a single restart
                if attempt == 0 and isinstance(engine_exc, (BrokenPipeError, OSError, EOFError, RuntimeError)):
                    try:
                        teacher.close()
                    except Exception:
                        pass
                    try:
                        teacher = _init_teacher()
                        continue
                    except Exception:
                        pass
                
                # Second failure or unrecoverable error
                skipped += 1
                errors.append((idx, f"Engine evaluation error: {engine_exc}"))
                wdl = wdl_from_row(row)
                success = True
                break
        
        if success and wdl is not None:
            results.append((idx, slim_row(row, wdl)))

    if teacher:
        try:
            teacher.close()
        except Exception:
            pass

    return results, skipped, errors


def relabel_file(
    input_path: Path,
    output: Path,
    *,
    weights_path: str,
    backend: str,
    go: str,
    workers: int,
    limit: int | None,
    checkpoint_every: int,
    progress: bool,
    resume: bool,
) -> dict:
    rows_in, how = load_rows(input_path)
    if how == "recovered":
        print(f"warning: {input_path.name} is not valid JSON; recovered {len(rows_in)} records", file=sys.stderr)
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
        try:
            prev = pd.read_parquet(parquet_out)
            out_rows = prev.to_dict(orient="records")
            start = len(out_rows)
            if start > len(rows_in):
                raise ValueError(f"checkpoint {parquet_out} has {start} rows > input {len(rows_in)}")
            print(f"resuming at position {start:,}/{len(rows_in):,} from {parquet_out}", flush=True)
        except Exception as exc:
            print(f"warning: failed to load checkpoint ({exc}); starting from scratch.", file=sys.stderr)
            out_rows = []
            start = 0

    skipped = 0
    rows_to_process = rows_in[start:]
    
    if not rows_to_process:
        print(f"File already fully processed: {input_path.name}", flush=True)
        return {
            "input": str(input_path),
            "output": str(output),
            "parquet": str(parquet_out),
            "count": len(out_rows),
            "skipped": 0,
            "go": go,
            "value_min": min((r["value"] for r in out_rows), default=0.0),
            "value_max": max((r["value"] for r in out_rows), default=0.0),
            "wdl": "STM [W, D, L]",
        }

    # Divide work into evenly distributed chunks
    chunk_size = max(1, len(rows_to_process) // (workers * 4))
    chunks = []
    current_chunk = []
    
    for i, row in enumerate(rows_to_process):
        current_chunk.append((start + i, row))
        if len(current_chunk) == chunk_size:
            chunks.append((current_chunk, weights_path, backend, go))
            current_chunk = []
    if current_chunk:
        chunks.append((current_chunk, weights_path, backend, go))

    def _flush() -> None:
        write_payload(out_rows, output)

    next_checkpoint = start + checkpoint_every if checkpoint_every else float('inf')

    with ProcessPoolExecutor(max_workers=workers) as executor:
        with _progress_bar(
            total=len(rows_in),
            desc=f"{input_path.stem} {go}",
            disable=not progress,
            initial=start,
        ) as bar:
            for res, skips, errs in executor.map(process_chunk, chunks):
                skipped += skips
                for idx, err_msg in errs:
                    print(f"warn row {idx}: {err_msg}", file=sys.stderr)
                
                for idx, out_row in res:
                    out_rows.append(out_row)
                
                bar.update(len(res))
                
                if len(out_rows) >= next_checkpoint:
                    _flush()
                    next_checkpoint += checkpoint_every

    _flush()
    return {
        "input": str(input_path),
        "output": str(output),
        "parquet": str(parquet_out),
        "count": len(out_rows),
        "skipped": skipped,
        "go": go,
        "value_min": min((r["value"] for r in out_rows), default=0.0),
        "value_max": max((r["value"] for r in out_rows), default=0.0),
        "wdl": "STM [W, D, L]",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Overwrite STM wdl in existing fen-value-visits JSON via Lc0 (DGX Optimized)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Slice JSON/parquet, slice folder, or parent directory",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Lc0 UCI go depth (default: 1). 0 = one NN eval (go nodes 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: <stem>_depthN.json",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input JSON directly",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Optional extra UCI node cap",
    )
    parser.add_argument(
        "--network",
        choices=sorted(LC0_NETWORK_PRESETS),
        default=None,
        help=f"Lc0 weights preset (default: {LC0_NETWORK_DEFAULT.name})",
    )
    parser.add_argument(
        "--backend", 
        default="cudnn", 
        help="Lc0 backend (default: cudnn)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=min(32, max(1, mp.cpu_count() // 2)), 
        help="Number of concurrent Lc0 worker processes"
    )
    parser.add_argument("--limit", type=int, default=None, help="Label only first N rows")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="Write output every N newly labeled rows",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="0 disables progress bar; positive enables updates",
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
        print("--depth must be >= 0", file=sys.stderr)
        return 1
    if args.nodes is not None and args.nodes < 1:
        print("--nodes must be >= 1", file=sys.stderr)
        return 1

    targets = discover_targets(input_path)
    if not targets:
        print(f"no valid target JSON found in {input_path}", file=sys.stderr)
        return 1
    if args.output is not None and len(targets) > 1:
        print("--output is only valid for a single target file.", file=sys.stderr)
        return 1

    in_place = bool(args.in_place) or (input_path.is_dir() and args.output is None)
    go = uci_go(depth=args.depth, nodes=args.nodes)
    weights = LC0_NETWORK_PRESETS[args.network] if args.network else LC0_NETWORK_DEFAULT
    weights = weights.resolve() if weights.exists() else weights
    checkpoint_every = max(0, int(args.checkpoint_every))

    print(f"Target files: {len(targets)} | Engine command: '{go}' | Workers: {args.workers}", flush=True)

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
                weights_path=str(weights),
                backend=args.backend,
                go=go,
                workers=args.workers,
                limit=args.limit,
                checkpoint_every=checkpoint_every,
                progress=bool(args.progress_every),
                resume=not in_place,
            )
        except Exception as exc:
            print(f"Error processing {target.name}: {exc}", file=sys.stderr)
            return 1
            
        summary["network"] = weights.name
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    if len(summaries) > 1:
        print(json.dumps({"files": len(summaries), "go": go}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())