#!/usr/bin/env python3
"""Build ``boards.json``: dict board_hash → board state from position parquets.

Default sources (Lc0 + Lichess human extract)::

    data/processed/lc0/positions.parquet
    data/processed/lichess/positions.parquet

Also merges FENs from labeled blocks if present.

Example::

    py -3.12 scripts/build_boards_json.py
    py -3.12 scripts/build_boards_json.py --output data/processed/board_eval/boards.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import (
    BOARD_EVAL_DIR_NAME,
    BOARDS_JSON_NAME,
    merge_board_into,
    save_boards_json,
)

DEFAULT_OUTPUT = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / BOARDS_JSON_NAME

DEFAULT_SOURCES = [
    PROCESSED_DATA_DIR / "lc0" / "positions.parquet",
    PROCESSED_DATA_DIR / "lichess" / "positions.parquet",
    PROCESSED_DATA_DIR / "labeled" / "lc0_labeled.parquet",
    PROCESSED_DATA_DIR / "labeled" / "lichess_labeled.parquet",
    PROCESSED_DATA_DIR / "labeled" / "train.parquet",
    PROCESSED_DATA_DIR / "labeled" / "val.parquet",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build boards.json hash → board state")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Parquet with a fen column (repeatable). Default: Lc0+Lichess processed + labeled",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a FEN does not parse",
    )
    args = parser.parse_args(argv)

    inputs = args.input if args.input else DEFAULT_SOURCES
    store: dict[str, dict] = {}
    stats = {
        "files_read": 0,
        "files_missing": 0,
        "rows_seen": 0,
        "rows_skipped": 0,
        "errors": 0,
        "per_file": {},
    }

    for path in inputs:
        path = Path(path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            stats["files_missing"] += 1
            print(f"skip missing: {path}", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        if "fen" not in df.columns:
            print(f"skip no fen column: {path}", file=sys.stderr)
            stats["files_missing"] += 1
            continue
        stats["files_read"] += 1
        before = len(store)
        n_ok = 0
        n_skip = 0
        for row in df.to_dict(orient="records"):
            stats["rows_seen"] += 1
            fen = row.get("fen")
            if not fen or not isinstance(fen, str):
                n_skip += 1
                stats["rows_skipped"] += 1
                continue
            try:
                merge_board_into(
                    store,
                    fen,
                    source=str(row["source"]) if row.get("source") is not None else None,
                    game_id=str(row["game_id"]) if row.get("game_id") is not None else None,
                    ply=int(row["ply"]) if row.get("ply") is not None else None,
                )
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 — collect bad FENs
                stats["errors"] += 1
                n_skip += 1
                stats["rows_skipped"] += 1
                if args.strict:
                    raise
                print(f"warn bad fen in {path.name}: {exc}", file=sys.stderr)
        stats["per_file"][str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)] = {
            "rows": len(df),
            "accepted": n_ok,
            "skipped": n_skip,
            "new_hashes": len(store) - before,
        }
        print(
            f"{path.name}: rows={len(df)} accepted={n_ok} "
            f"new_hashes={len(store) - before} total_unique={len(store)}"
        )

    total_obs = sum(int(v.get("count", 1)) for v in store.values())
    max_count = max((int(v.get("count", 1)) for v in store.values()), default=0)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Board hash → state (FEN/EPD) + observation count. "
            "Join with values.json on hash for (board, eval). "
            "Use count for sample weighting / multiplicity later."
        ),
        "hash_algorithm": "sha256(epd)",
        "n_boards": len(store),
        "n_observations": total_obs,
        "max_count": max_count,
        "stats": stats,
        "output": str(args.output),
    }
    out = save_boards_json(store, args.output, meta=meta)
    # Sidecar compact stats for logs
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(store):,} unique boards → {out}")
    print(f"Stats → {stats_path}")
    return 0 if len(store) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
