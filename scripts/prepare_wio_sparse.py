#!/usr/bin/env python3
"""
Wio Terminal — sparse int8 value-net export (80% weight sparsity).

Applies global magnitude pruning so 80% of weight values are zero (20% non-zero),
then quantizes and emits a C header for Arduino/Wio_TinyValueTest/.

Default architecture: MediumValueMLP (768→128→64→1). Use --model to pick another tier.

Also saves: models/checkpoints/<name>.pt and models/exported/<name>_int8.bin

Usage (from project root):

  py -3.12 scripts/prepare_wio_sparse.py

  py -3.12 scripts/prepare_wio_sparse.py --model small --sparsity 0.8

  py -3.12 scripts/prepare_wio_sparse.py --train --epochs 2 --max-games 600
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import torch

from tinymlinternship.config.settings import CHECKPOINTS_DIR, WIO_SKETCH_DIR
from tinymlinternship.models.value import (
    BigValueMLP,
    HugeValueMLP,
    MediumValueMLP,
    SmallValueMLP,
    TinyValueMLP,
    UltraTinyValueMLP,
)
from wio_int8_common import (
    apply_weight_sparsity,
    ensure_dirs,
    export_compact_blob,
    export_sparse_csr_c_header,
    quantize_to_int8,
    run_training_if_requested,
)

DEFAULT_SPARSITY = 0.8
HEADER_NAME = "wio_int8_weights_sparse80.h"
HEADER_GUARD = "WIO_INT8_WEIGHTS_SPARSE80_H"
DEFAULT_CKPT_NAME = "tiny_value_wio_sparse80_int8"

MODEL_CHOICES = {
    "nano": (UltraTinyValueMLP, "768→16→8→1"),
    "tiny": (TinyValueMLP, "768→32→16→1"),
    "small": (SmallValueMLP, "768→64→32→1"),
    "medium": (MediumValueMLP, "768→128→64→1"),
    "big": (BigValueMLP, "768→256→64→1"),
    "huge": (HugeValueMLP, "768→512→64→1"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare sparse (80% zero) int8 value-net header for Wio Terminal"
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES),
        default="medium",
        help="Base architecture before pruning (default: medium)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=DEFAULT_SPARSITY,
        help="Fraction of weight values to zero (default: 0.8 → 20%% non-zero)",
    )
    parser.add_argument("--train", action="store_true", help="Run quick outcome-based training")
    parser.add_argument("--max-games", type=int, default=600, help="Games to sample for training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--name", default=DEFAULT_CKPT_NAME, help="Checkpoint / blob base name")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Load an existing .pt checkpoint (skip prune; re-export header only)",
    )
    args = parser.parse_args()

    ensure_dirs()
    WIO_SKETCH_DIR.mkdir(parents=True, exist_ok=True)

    model_cls, arch_label = MODEL_CHOICES[args.model]
    model = model_cls().eval()
    print(f"Using {model_cls.__name__} ({arch_label})")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}  (int8 ≈ {total_params/1024:.1f} KB dense)")

    ckpt_path = CHECKPOINTS_DIR / f"{args.name}.pt"

    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")
        pruned_sd = model.state_dict()
        weight_keys = [k for k in pruned_sd if "weight" in k]
        total = sum(pruned_sd[k].numel() for k in weight_keys)
        nonzero = sum((pruned_sd[k] != 0).sum().item() for k in weight_keys)
        stats = {
            "sparsity_target": args.sparsity,
            "sparsity_actual": 1.0 - (nonzero / total) if total else 0.0,
            "nonzero_weights": nonzero,
            "total_weights": total,
            "zeroed": total - nonzero,
        }
    else:
        run_training_if_requested(model, args.train, args.max_games, args.epochs)
        sd = model.state_dict()
        pruned_sd, stats = apply_weight_sparsity(sd, sparsity=args.sparsity)
        model.load_state_dict(pruned_sd)

    print(
        f"Sparsity: target={stats['sparsity_target']:.0%}  "
        f"actual={stats['sparsity_actual']:.1%}  "
        f"non-zero={stats['nonzero_weights']:,}/{stats['total_weights']:,}"
    )

    if args.checkpoint is None:
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved pruned checkpoint: {ckpt_path}")

    q_sd, scales = quantize_to_int8(pruned_sd)
    export_compact_blob(q_sd, scales, model_name=args.name)

    header_path = WIO_SKETCH_DIR / HEADER_NAME
    export_sparse_csr_c_header(
        model, q_sd, scales, header_path, HEADER_GUARD, sparsity=args.sparsity
    )

    hsize = header_path.stat().st_size
    nnz = stats["nonzero_weights"]
    dense_bytes = stats["total_weights"]
    csr_bytes = nnz * 3 + (model.fc1.out_features + model.fc2.out_features + model.fc3.out_features + 3) * 2
    print(f"\n✅ Generated sparse CSR sketch header: {header_path}")
    print(
        f"   Size: {hsize / 1024:.1f} KB  |  CSR payload ≈ {csr_bytes / 1024:.1f} KB "
        f"vs {dense_bytes / 1024:.1f} KB dense  |  {stats['sparsity_actual']:.0%} sparse"
    )


if __name__ == "__main__":
    main()