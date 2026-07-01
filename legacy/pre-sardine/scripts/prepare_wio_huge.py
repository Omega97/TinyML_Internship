#!/usr/bin/env python3
"""
Wio Terminal — HugeValueMLP (768→512→64→1) → int8 C header for Arduino.

Largest model that fits the Wio's ~512 KB internal flash (~95% of 507904 bytes
with ~60 KB sketch overhead). Produces wio_int8_weights_huge.h in
Arduino/Wio_TinyValueTest/ with FC*_IN/OUT_DIM macros, int8 weight/bias arrays,
and per-tensor scales.

Also saves: models/checkpoints/<name>.pt and models/exported/<name>_int8.bin

Usage (from project root):

  py -3.12 scripts/prepare_wio_huge.py

  py -3.12 scripts/prepare_wio_huge.py --name tiny_value_wio_huge_int8

  py -3.12 scripts/prepare_wio_huge.py --train --epochs 2 --max-games 600
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: E402, F401
sys.path.insert(0, str(Path(__file__).parent))

import torch

from tinymlinternship.config.settings import CHECKPOINTS_DIR, WIO_SKETCH_DIR
from tinymlinternship.models.value import HugeValueMLP
from wio_int8_common import (
    ensure_dirs,
    export_compact_blob,
    export_structured_c_header,
    quantize_to_int8,
    run_training_if_requested,
)

HEADER_NAME = "wio_int8_weights_huge.h"
HEADER_GUARD = "WIO_INT8_WEIGHTS_HUGE_H"
DEFAULT_CKPT_NAME = "tiny_value_wio_huge_int8"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HugeValueMLP (768→512→64→1) int8 header for Wio Terminal"
    )
    parser.add_argument("--train", action="store_true", help="Run quick outcome-based training")
    parser.add_argument("--max-games", type=int, default=600, help="Games to sample for training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--name", default=DEFAULT_CKPT_NAME, help="Checkpoint / blob base name")
    args = parser.parse_args()

    ensure_dirs()
    WIO_SKETCH_DIR.mkdir(parents=True, exist_ok=True)

    model = HugeValueMLP().eval()
    print("Using HugeValueMLP (768→512→64→1)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}  (int8 ≈ {total_params/1024:.1f} KB)")

    run_training_if_requested(model, args.train, args.max_games, args.epochs)

    ckpt_path = CHECKPOINTS_DIR / f"{args.name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved full checkpoint: {ckpt_path}")

    sd = model.state_dict()
    q_sd, scales = quantize_to_int8(sd)
    export_compact_blob(q_sd, scales, model_name=args.name)

    header_path = WIO_SKETCH_DIR / HEADER_NAME
    export_structured_c_header(model, q_sd, scales, header_path, HEADER_GUARD)

    hsize = header_path.stat().st_size
    print(f"\n✅ Generated sketch header: {header_path}")
    print(f"   Size: {hsize / 1024:.1f} KB (FC*_IN/OUT_DIM macros + int8 PROGMEM arrays)")


if __name__ == "__main__":
    main()