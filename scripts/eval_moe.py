#!/usr/bin/env python3
"""Evaluate base vs dispatcher-routed MoE (and k-means oracle) on holdout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.moe import DualHiddenMoE
from tinymlinternship.nnue.moe_data import slice_folders
from tinymlinternship.nnue.moe_pipeline import (
    configure_torch,
    evaluate_moe,
    load_base_on_device,
    write_plots,
)

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_CKPT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
PLOTS_DIR = PROJECT_ROOT / "plots" / "MoE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--moe", type=Path, default=None)
    parser.add_argument("--max-test", type=int, default=50_000)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--no-oracle", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(device)
    slices_dir = args.slices_dir if args.slices_dir.is_absolute() else (PROJECT_ROOT / args.slices_dir)
    work_dir = args.input_dir if args.input_dir.is_absolute() else (PROJECT_ROOT / args.input_dir)
    plots_dir = args.plots_dir if args.plots_dir.is_absolute() else (PROJECT_ROOT / args.plots_dir)
    folders = slice_folders(slices_dir)
    base = load_base_on_device(args.checkpoint, device)
    moe_path = args.moe or (work_dir / "moe.pt")
    moe = DualHiddenMoE.load(moe_path, device=device)
    evaluate_moe(
        base,
        moe,
        folders,
        work_dir,
        device=device,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        max_test=args.max_test,
        batch_size=args.batch_size,
        oracle=not args.no_oracle,
    )
    write_plots(work_dir, plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
