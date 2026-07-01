#!/usr/bin/env python3
"""
Count parameters for all Wio value-net model variants.

Reports architecture, parameter count, memory sizes (float32 / int8), and
estimated flash usage on the Wio Terminal (507904-byte budget, ~60 KB overhead).

Usage (from project root):

  py -3.12 scripts/count_model_params.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: E402, F401

import torch
import torch.nn as nn

from tinymlinternship.config.settings import CHECKPOINTS_DIR, WIO_SKETCH_DIR
from tinymlinternship.models.policy import TinyPolicy
from tinymlinternship.models.value import (
    BigValueMLP,
    HugeValueMLP,
    MediumValueMLP,
    SmallValueMLP,
    TinyValueMLP,
    UltraTinyValueMLP,
)

WIO_FLASH_BUDGET = 507_904
WIO_SKETCH_OVERHEAD = 60_000


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def architecture_str(model: nn.Module) -> str:
    if hasattr(model, "hidden1"):
        return f"{model.input_size}→{model.hidden1}→{model.hidden2}→1"
    if isinstance(model, UltraTinyValueMLP):
        return "768→16→8→1"
    if isinstance(model, TinyPolicy):
        return "12×8×8 conv→4096"
    return "?"


def parse_header_dims(header_path: Path) -> dict[str, int]:
    text = header_path.read_text(encoding="utf-8")
    dims = {}
    for name in ("FC1_IN_DIM", "FC1_OUT_DIM", "FC2_IN_DIM", "FC2_OUT_DIM", "FC3_IN_DIM", "FC3_OUT_DIM"):
        match = re.search(rf"#define\s+{name}\s+(\d+)", text)
        if match:
            dims[name] = int(match.group(1))
    return dims


def params_from_dims(dims: dict[str, int]) -> int:
    fc1_in, fc1_out = dims["FC1_IN_DIM"], dims["FC1_OUT_DIM"]
    fc2_in, fc2_out = dims["FC2_IN_DIM"], dims["FC2_OUT_DIM"]
    fc3_in = dims["FC3_IN_DIM"]
    return (
        fc1_in * fc1_out + fc1_out
        + fc2_in * fc2_out + fc2_out
        + fc3_in + 1
    )


def flash_estimate_bytes(params: int) -> tuple[int, float]:
    used = params + WIO_SKETCH_OVERHEAD
    pct = 100.0 * used / WIO_FLASH_BUDGET
    return used, pct


VALUE_MODELS = [
    ("nano", UltraTinyValueMLP),
    ("tiny", TinyValueMLP),
    ("small", SmallValueMLP),
    ("medium", MediumValueMLP),
    ("big", BigValueMLP),
    ("huge", HugeValueMLP),
]


def print_table(title: str, rows: list[tuple], headers: tuple[str, ...] | None = None) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if headers is None:
        headers = ("name", "architecture", "params", "float32 KB", "int8 KB", "flash est.", "flash %")
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    value_rows = []
    for name, model_cls in VALUE_MODELS:
        model = model_cls().eval()
        params = count_params(model)
        used, pct = flash_estimate_bytes(params)
        value_rows.append(
            (
                name,
                architecture_str(model),
                f"{params:,}",
                f"{params * 4 / 1024:.1f}",
                f"{params / 1024:.1f}",
                f"{used:,} B",
                f"{pct:.1f}%",
            )
        )

    print_table("Value models (from Python classes)", value_rows)

    header_rows = []
    for header_path in sorted(WIO_SKETCH_DIR.glob("wio_int8_weights_*.h")):
        name = header_path.stem.replace("wio_int8_weights_", "")
        dims = parse_header_dims(header_path)
        if not dims:
            continue
        params = params_from_dims(dims)
        arch = f"{dims['FC1_IN_DIM']}→{dims['FC1_OUT_DIM']}→{dims['FC2_OUT_DIM']}→1"
        used, pct = flash_estimate_bytes(params)
        header_rows.append(
            (
                name,
                arch,
                f"{params:,}",
                f"{params * 4 / 1024:.1f}",
                f"{params / 1024:.1f}",
                f"{used:,} B",
                f"{pct:.1f}%",
                f"{header_path.stat().st_size / 1024:.1f} KB",
            )
        )

    if header_rows:
        print_table(
            "Generated Arduino headers (parsed FC*_DIM macros)",
            header_rows,
            headers=("name", "architecture", "params", "float32 KB", "int8 KB", "flash est.", "flash %", ".h size"),
        )
        print("  (.h source size is C text formatting; int8 KB column is actual weight bytes)")

    ckpt_rows = []
    for ckpt_path in sorted(CHECKPOINTS_DIR.glob("*.pt")):
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            params = count_params(obj)
        elif isinstance(obj, dict):
            params = sum(v.numel() for v in obj.values() if torch.is_tensor(v))
        else:
            continue
        ckpt_rows.append((ckpt_path.name, f"{params:,}", f"{params / 1024:.1f}"))

    if ckpt_rows:
        print_table(
            "Saved checkpoints",
            [(n, p, f"{i} KB int8") for n, p, i in ckpt_rows],
            headers=("checkpoint", "params", "int8 size"),
        )

    policy = TinyPolicy().eval()
    policy_params = count_params(policy)
    print_table(
        "Other models",
        [
            (
                "TinyPolicy",
                architecture_str(policy),
                f"{policy_params:,}",
                f"{policy_params * 4 / 1024:.1f}",
                f"{policy_params / 1024:.1f}",
                "n/a",
                "n/a",
            )
        ],
    )

    print(f"\nWio flash budget: {WIO_FLASH_BUDGET:,} bytes | sketch overhead estimate: {WIO_SKETCH_OVERHEAD:,} bytes")


if __name__ == "__main__":
    main()