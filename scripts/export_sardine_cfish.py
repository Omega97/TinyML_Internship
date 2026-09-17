#!/usr/bin/env python3
"""Export DualHidden weights to a binary Cfish can mmap/fread."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch

MAGIC = b"SRDN"
VERSION = 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("models/checkpoints/nnue/dual_h128_H256_spark_32x4096/best.pt"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/omar/jupyterlab/lichess-bot/engines/sardine.bin"),
    )
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = payload["model_state_dict"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    hidden = int(payload.get("hidden_dim", sd["l1.weight"].shape[0]))
    hidden2 = int(payload.get("hidden2_dim", sd["l2.weight"].shape[0]))
    feat = int(sd["l1.weight"].shape[1])
    blobs = [
        sd["l1.weight"].float().contiguous().numpy(),
        sd["l1.bias"].float().contiguous().numpy(),
        sd["l2.weight"].float().contiguous().numpy(),
        sd["l2.bias"].float().contiguous().numpy(),
        sd["head.weight"].float().contiguous().numpy(),
        sd["head.bias"].float().contiguous().numpy(),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, feat, hidden, hidden2))
        for arr in blobs:
            f.write(arr.tobytes(order="C"))
    print(
        f"wrote {args.out} feat={feat} W={hidden} H={hidden2} "
        f"test_ce={payload.get('test_ce')} epoch={payload.get('epoch')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
