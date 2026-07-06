#!/usr/bin/env python3
"""Convert ChessBench state_value .bag → SARDINE NNUE training parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tinymlinternship.config.settings import CHESSBENCH_PROCESSED_DIR, CHESSBENCH_RAW_DIR
from tinymlinternship.data.chessbench_preprocess import (
    iter_state_value_records,
    parse_chessbench_row,
    row_to_dict,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ChessBench state_value .bag → sparse 716 features + expected_reward"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CHESSBENCH_RAW_DIR / "test" / "state_value_data.bag",
        help="ChessBench state_value .bag (Research format)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHESSBENCH_PROCESSED_DIR,
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Max rows (debug)")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        print("Run: py -3.12 scripts/download_chessbench.py", file=sys.stderr)
        return 1

    rows: list[dict] = []
    skipped = 0
    for i, (fen, win_prob) in enumerate(iter_state_value_records(args.input)):
        if args.limit is not None and i >= args.limit:
            break
        parsed = parse_chessbench_row(fen, win_prob)
        if parsed is None:
            skipped += 1
            continue
        rows.append(row_to_dict(parsed))

    if not rows:
        print("No valid rows produced.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_val = max(1, int(len(df) * args.val_fraction))
    val_df = df.iloc[:n_val].copy()
    train_df = df.iloc[n_val:].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_path = args.output_dir / "positions.parquet"
    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "val.parquet"

    df.to_parquet(all_path, index=False)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    manifest = {
        "source": str(args.input.resolve()),
        "format": "ChessBench state_value (SF16 win_prob → expected_reward)",
        "label_formula": "expected_reward = clip(2 * win_prob - 1, -1, +1)  # STM POV",
        "features": "encode_dual() → white_features, black_features (sparse 716 indices)",
        "rows_total": len(df),
        "rows_train": len(train_df),
        "rows_val": len(val_df),
        "rows_skipped": skipped,
        "expected_reward_min": float(df["expected_reward"].min()),
        "expected_reward_max": float(df["expected_reward"].max()),
        "expected_reward_mean": float(df["expected_reward"].mean()),
        "bucket_counts": df["bucket_id"].value_counts().sort_index().to_dict(),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Rows: {len(df):,} (train {len(train_df):,}, val {len(val_df):,}, skipped {skipped})")
    print(f"expected_reward: [{manifest['expected_reward_min']:.4f}, {manifest['expected_reward_max']:.4f}] "
          f"mean={manifest['expected_reward_mean']:.4f}")
    print(f"Saved: {all_path}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())