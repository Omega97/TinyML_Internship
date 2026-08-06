#!/usr/bin/env python3
"""Build ``values.json``: board_hash → teacher eval (expected_reward).

Reads labeled parquets (must have ``fen`` + ``expected_reward``). Aggregates
multiple labels per hash with a running mean; keeps last teacher id.

Example::

    py -3.12 scripts/build_values_json.py
    py -3.12 scripts/build_values_json.py --input data/processed/labeled/train.parquet
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
    VALUES_JSON_NAME,
    board_hash,
)
from tinymlinternship.data.schema import LABEL_FORMULA

DEFAULT_OUTPUT = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / VALUES_JSON_NAME

DEFAULT_SOURCES = [
    PROCESSED_DATA_DIR / "labeled" / "lc0_labeled.parquet",
    PROCESSED_DATA_DIR / "labeled" / "lichess_labeled.parquet",
    PROCESSED_DATA_DIR / "labeled" / "train.parquet",
    PROCESSED_DATA_DIR / "labeled" / "val.parquet",
]


def merge_value(
    store: dict[str, dict],
    fen: str,
    reward: float,
    *,
    teacher_network: str | None = None,
    source: str | None = None,
    wdl: tuple[float, float, float] | None = None,
) -> str:
    """Running mean of expected_reward per hash; increments n_labels."""
    key = board_hash(fen)
    r = float(reward)
    if key not in store:
        rec: dict = {
            "expected_reward": r,
            "n_labels": 1,
            "reward_sum": r,
            "reward_min": r,
            "reward_max": r,
        }
        if teacher_network:
            rec["teacher_network"] = teacher_network
        if source:
            rec["sources"] = [source]
        if wdl is not None:
            rec["wdl_win"], rec["wdl_draw"], rec["wdl_loss"] = wdl
        store[key] = rec
        return key
    rec = store[key]
    n = int(rec.get("n_labels", 1)) + 1
    s = float(rec.get("reward_sum", rec["expected_reward"])) + r
    rec["n_labels"] = n
    rec["reward_sum"] = s
    rec["expected_reward"] = s / n
    rec["reward_min"] = min(float(rec.get("reward_min", r)), r)
    rec["reward_max"] = max(float(rec.get("reward_max", r)), r)
    if teacher_network:
        rec["teacher_network"] = teacher_network
    if source:
        srcs = rec.setdefault("sources", [])
        if source not in srcs:
            srcs.append(source)
    return key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build values.json hash → eval")
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    inputs = args.input if args.input else DEFAULT_SOURCES
    store: dict[str, dict] = {}
    stats = {"files_read": 0, "rows": 0, "skipped": 0, "per_file": {}}

    for path in inputs:
        path = Path(path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            print(f"skip missing: {path}", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        if "fen" not in df.columns or "expected_reward" not in df.columns:
            print(f"skip need fen+expected_reward: {path}", file=sys.stderr)
            continue
        stats["files_read"] += 1
        before = len(store)
        n_ok = 0
        for row in df.to_dict(orient="records"):
            stats["rows"] += 1
            fen = row.get("fen")
            rew = row.get("expected_reward")
            if fen is None or rew is None or (isinstance(rew, float) and rew != rew):
                stats["skipped"] += 1
                continue
            wdl = None
            if all(k in row and row[k] is not None for k in ("wdl_win", "wdl_draw", "wdl_loss")):
                wdl = (float(row["wdl_win"]), float(row["wdl_draw"]), float(row["wdl_loss"]))
            merge_value(
                store,
                str(fen),
                float(rew),
                teacher_network=str(row["teacher_network"])
                if row.get("teacher_network") is not None
                else None,
                source=str(row["source"]) if row.get("source") is not None else None,
                wdl=wdl,
            )
            n_ok += 1
        stats["per_file"][str(path)] = {
            "rows": len(df),
            "accepted": n_ok,
            "new_hashes": len(store) - before,
        }
        print(f"{path.name}: rows={len(df)} labeled={n_ok} unique_values={len(store)}")

    # Drop helper sum before write (keep mean + n_labels)
    for rec in store.values():
        rec.pop("reward_sum", None)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Board hash → teacher value. Join boards.json for (board, eval).",
        "label_formula": LABEL_FORMULA,
        "hash_algorithm": "sha256(epd)",
        "n_values": len(store),
        "stats": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "values": store}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(store):,} values → {args.output}")
    return 0 if store else 1


if __name__ == "__main__":
    raise SystemExit(main())
