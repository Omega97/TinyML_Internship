#!/usr/bin/env python3
"""Join boards.json + values.json → train/val parquet for NNUE training.

Sample weight column ``count`` from boards (observation multiplicity).

Example::

    py -3.12 scripts/export_board_eval_pairs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tinymlinternship.config.settings import PROCESSED_DATA_DIR
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, BOARDS_JSON_NAME, VALUES_JSON_NAME
from tinymlinternship.features.bucket import bucket_id, has_queen, piece_count
import chess

DEFAULT_BOARDS = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / BOARDS_JSON_NAME
DEFAULT_VALUES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / VALUES_JSON_NAME
DEFAULT_OUT = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "pairs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export joined board+eval pairs")
    parser.add_argument("--boards", type=Path, default=DEFAULT_BOARDS)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    boards_doc = json.loads(args.boards.read_text(encoding="utf-8"))
    values_doc = json.loads(args.values.read_text(encoding="utf-8"))
    boards = boards_doc.get("boards", boards_doc)
    values = values_doc.get("values", values_doc)

    rows = []
    missing_board = 0
    for h, v in values.items():
        b = boards.get(h)
        if b is None:
            missing_board += 1
            continue
        fen = b["fen"]
        board = chess.Board(fen)
        rows.append(
            {
                "board_hash": h,
                "fen": fen,
                "expected_reward": float(v["expected_reward"]),
                "count": int(b.get("count", 1)),
                "stm_white": bool(b.get("stm_white", board.turn == chess.WHITE)),
                "bucket_id": int(bucket_id(board)),
                "piece_count": int(piece_count(board)),
                "has_queen": bool(has_queen(board)),
                "source": (b.get("sources") or [b.get("source") or "unknown"])[0],
                "teacher_network": v.get("teacher_network"),
                "n_labels": int(v.get("n_labels", 1)),
            }
        )

    if not rows:
        print("No joined pairs.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    # Weighted-ish split: shuffle by hash seed
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(len(df) * args.val_fraction))
    val = df.iloc[:n_val].copy()
    train = df.iloc[n_val:].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "boards": str(args.boards),
        "values": str(args.values),
        "n_pairs": len(df),
        "n_train": len(train),
        "n_val": len(val),
        "missing_board_for_value": missing_board,
        "sum_count": int(df["count"].sum()),
        "reward_mean": float(df["expected_reward"].mean()),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"pairs={len(df)} train={len(train)} val={len(val)} "
        f"sum_count={manifest['sum_count']} → {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
