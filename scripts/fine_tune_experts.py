#!/usr/bin/env python3
"""Freeze L1 and fine-tune one expert head per bucket."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.moe_data import slice_folders
from tinymlinternship.nnue.moe_pipeline import configure_torch, fine_tune_experts, load_base_on_device

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_CKPT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument(
        "--expert-labels",
        choices=("dispatcher", "kmeans"),
        default="dispatcher",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(device)
    slices_dir = args.slices_dir if args.slices_dir.is_absolute() else (PROJECT_ROOT / args.slices_dir)
    work_dir = args.input_dir if args.input_dir.is_absolute() else (PROJECT_ROOT / args.input_dir)
    folders = slice_folders(slices_dir)
    base = load_base_on_device(args.checkpoint, device)
    fine_tune_experts(
        base,
        work_dir,
        folders,
        device=device,
        n_clusters=args.n_clusters,
        expert_labels=args.expert_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
