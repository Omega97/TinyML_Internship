#!/usr/bin/env python3
"""Compute L2-normalized, randomly projected per-sample head gradients."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.moe_data import slice_folders
from tinymlinternship.nnue.moe_pipeline import (
    DEFAULT_GRAD_BATCH,
    DEFAULT_REDUCE_DIM,
    compute_gradients,
    configure_torch,
    load_base_on_device,
)
from tinymlinternship.nnue.sample_gradients import max_reduce_dim_for_budget

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_CKPT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_GRAD_BATCH)
    parser.add_argument("--reduce-dim", type=int, default=DEFAULT_REDUCE_DIM)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--proj-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    max_rows = 4096 if args.smoke and args.max_rows == 1_000_000 else args.max_rows
    cap = max_reduce_dim_for_budget()
    if int(args.reduce_dim) > cap:
        print(
            f"WARNING: --reduce-dim {args.reduce_dim} stores "
            f">{100:.0f} MiB per million rows (cap {cap} for float16)",
            file=sys.stderr,
        )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(device)
    slices_dir = args.slices_dir if args.slices_dir.is_absolute() else (PROJECT_ROOT / args.slices_dir)
    folders = slice_folders(slices_dir)
    if not folders:
        print(f"no slice folders in {slices_dir}", file=sys.stderr)
        return 1
    work_dir = args.output_dir or (PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe" / "default")
    work_dir = work_dir if work_dir.is_absolute() else (PROJECT_ROOT / work_dir)
    model = load_base_on_device(args.checkpoint, device)
    compute_gradients(
        model,
        folders,
        work_dir,
        device=device,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        subset_seed=args.subset_seed,
        max_rows=max_rows,
        batch_size=args.batch_size,
        reduce_dim=args.reduce_dim,
        proj_seed=args.proj_seed,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
