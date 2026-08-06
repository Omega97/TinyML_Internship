#!/usr/bin/env python3
"""Build unified dataset.json: hash → {fen, stm_white, count, source, ply, reward, wdl}.

1) Count observations from **raw extracts only** (not labeled re-exports — avoids
   double-counting every labeled FEN).
2) Merge teacher labels from labeled parquets (running mean if multi-label).
3) Write only positions that have expected_reward.

Example::

    py -3.12 scripts/build_dataset_json.py
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
    CANONICAL_TAGS,
    DATASET_JSON_NAME,
    bump_count,
    merge_label,
    save_dataset_json,
)
from tinymlinternship.data.schema import LABEL_FORMULA

DEFAULT_OUTPUT = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / DATASET_JSON_NAME

# Position extracts → count only
COUNT_SOURCES = [
    PROCESSED_DATA_DIR / "lc0" / "positions.parquet",
    PROCESSED_DATA_DIR / "lichess" / "positions.parquet",
    PROCESSED_DATA_DIR / "lichess" / "kaggle_games_positions.parquet",
]

# Labeled → count + values
LABEL_SOURCES = [
    PROCESSED_DATA_DIR / "labeled" / "lc0_labeled.parquet",
    PROCESSED_DATA_DIR / "labeled" / "lichess_labeled.parquet",
    PROCESSED_DATA_DIR / "labeled" / "lc0_large_25k.parquet",
    PROCESSED_DATA_DIR / "labeled" / "lichess_kaggle_10k.parquet",
    PROCESSED_DATA_DIR / "labeled" / "train.parquet",
    PROCESSED_DATA_DIR / "labeled" / "val.parquet",
]


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build unified board+eval dataset.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--count-input",
        type=Path,
        action="append",
        default=None,
        help="Parquet with fen for count only (repeatable)",
    )
    parser.add_argument(
        "--label-input",
        type=Path,
        action="append",
        default=None,
        help="Labeled parquet fen+expected_reward+wdl (repeatable)",
    )
    args = parser.parse_args(argv)

    count_paths = [_resolve(p) for p in (args.count_input or COUNT_SOURCES)]
    label_paths = [_resolve(p) for p in (args.label_input or LABEL_SOURCES)]

    store: dict[str, dict] = {}
    stats = {"count_rows": 0, "label_rows": 0, "files": {}}

    # Pass 1: observation counts from **raw extracts only** (not labeled re-exports).
    # Counting labeled parquets as well double-counted every labeled FEN (count≥2).
    for path in count_paths:
        if not path.exists():
            print(f"skip missing: {path}", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        if "fen" not in df.columns:
            continue
        n = 0
        for row in df.to_dict(orient="records"):
            fen = row.get("fen")
            if not fen:
                continue
            try:
                bump_count(
                    store,
                    str(fen),
                    source=str(row["source"]) if row.get("source") is not None else None,
                    ply=int(row["ply"]) if row.get("ply") is not None else None,
                )
                n += 1
                stats["count_rows"] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"warn count {path.name}: {exc}", file=sys.stderr)
        stats["files"][str(path)] = {"count_rows": n}
        print(f"count {path.name}: {n}  store_size={len(store)}")

    # Pass 2: labels only (reward/WDL). Does not re-count extract duplicates.
    # If a hash was never in extracts, merge_label sets count=1 once.
    for path in label_paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "fen" not in df.columns or "expected_reward" not in df.columns:
            print(f"skip labels (need fen+reward): {path}", file=sys.stderr)
            continue
        n = 0
        for row in df.to_dict(orient="records"):
            fen = row.get("fen")
            rew = row.get("expected_reward")
            if fen is None or rew is None:
                continue
            ww = row.get("wdl_win", 0)
            wd = row.get("wdl_draw", 0)
            wl = row.get("wdl_loss", 0)
            if ww is None or wd is None or wl is None:
                continue
            try:
                merge_label(
                    store,
                    str(fen),
                    expected_reward=float(rew),
                    wdl_win=float(ww),
                    wdl_draw=float(wd),
                    wdl_loss=float(wl),
                    source=str(row["source"]) if row.get("source") is not None else None,
                    ply=int(row["ply"]) if row.get("ply") is not None else None,
                )
                n += 1
                stats["label_rows"] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"warn label {path.name}: {exc}", file=sys.stderr)
        stats["files"].setdefault(str(path), {})["label_rows"] = n
        print(f"label {path.name}: {n}")

    labeled = sum(1 for r in store.values() if "expected_reward" in r)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Unified hash → board+eval (canonical tags only).",
        "hash_algorithm": "sha256(epd)",
        "label_formula": LABEL_FORMULA,
        "canonical_tags": list(CANONICAL_TAGS),
        "teacher_network": "791556.pb.gz",
        "stats": stats,
        "n_shells_before_filter": len(store),
        "n_labeled": labeled,
    }
    out = save_dataset_json(store, args.output, meta=meta, labeled_only=True)
    # re-read size
    doc = json.loads(out.read_text(encoding="utf-8"))
    n = doc["meta"]["n_positions"]
    print(f"Wrote {n:,} labeled positions → {out}")
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(doc["meta"], indent=2), encoding="utf-8")
    print(f"Stats → {stats_path}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
