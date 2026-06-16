#!/usr/bin/env python3
"""
Re-runnable script to generate wio_weights.h from a tiny value net checkpoint.

This turns the Python -c one-liner into a proper, editable, re-runnable Python file
so you can easily re-generate the weights header whenever needed (e.g. after retraining).

Usage (from project root, inside your .venv or with the right Python):
    python scripts/generate_wio_weights.py
    python scripts/generate_wio_weights.py \
        --checkpoint models/checkpoints/my_tiny_model.pt \
        --output Arduino/Wio_TinyValueTest/wio_weights.h

The output is a C header with PROGMEM arrays for the UltraTinyValueMLP
(768 -> 16 -> 8 -> 1) that can be #included directly in an Arduino/Wio sketch.
"""

import argparse
import sys
from pathlib import Path

# Make the package importable when running the script directly
# (consistent with run_model.py, run_pipeline.py, etc.)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from tinymlinternship.models.value import UltraTinyValueMLP


def main():
    parser = argparse.ArgumentParser(
        description="Generate wio_weights.h (C header) from a trained UltraTinyValueMLP checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/tiny_value_wio.pt"),
        help="Path to the .pt checkpoint (default: tiny_value_wio.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Arduino/Wio_TinyValueTest/wio_weights.h"),
        help="Output header filename (default: Arduino/Wio_TinyValueTest/wio_weights.h)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading model from {args.checkpoint}...")
    model = UltraTinyValueMLP()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    def c_array(name: str, tensor: torch.Tensor) -> str:
        """Convert a tensor to a C array literal (float, PROGMEM for Arduino)."""
        flat = tensor.detach().cpu().float().flatten().tolist()
        lines = []
        for i in range(0, len(flat), 8):
            chunk = ", ".join(f"{v:.8f}f" for v in flat[i : i + 8])
            lines.append("  " + chunk)
        body = ",\n".join(lines)
        return f"const float {name}[] PROGMEM = {{\n{body}\n}};"

    lines = [
        "// Auto-generated weights for UltraTinyValueMLP (768 -> 16 -> 8 -> 1)",
        "// For Wio Terminal (SAMD51) - hand-written forward pass in Arduino sketch",
        f"// Source: {args.checkpoint}",
        "",
        c_array("fc1_w", model.fc1.weight),
        c_array("fc1_b", model.fc1.bias),
        c_array("fc2_w", model.fc2.weight),
        c_array("fc2_b", model.fc2.bias),
        c_array("fc3_w", model.fc3.weight),
        c_array("fc3_b", model.fc3.bias),
        "",
        "// Minimal forward pass (float) - copy into your .ino if needed:",
        "// float forward(const float* x) {",
        "//   float h1[16];",
        "//   for (int i = 0; i < 16; i++) {",
        "//     float sum = fc1_b[i];",
        "//     for (int j = 0; j < 768; j++) sum += x[j] * fc1_w[i*768 + j];",
        "//     h1[i] = (sum > 0) ? sum : 0;  // ReLU",
        "//   }",
        "//   float h2[8];",
        "//   for (int i = 0; i < 8; i++) {",
        "//     float sum = fc2_b[i];",
        "//     for (int j = 0; j < 16; j++) sum += h1[j] * fc2_w[i*16 + j];",
        "//     h2[i] = (sum > 0) ? sum : 0;",
        "//   }",
        "//   float out = fc3_b[0];",
        "//   for (int j = 0; j < 8; j++) out += h2[j] * fc3_w[j];",
        "//   return tanhf(out);  // #include <math.h>",
        "// }",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    size_kb = args.output.stat().st_size / 1024
    print(f"✅ Wrote {args.output} ({size_kb:.1f} KB)")
    print("   You can now #include it in your Wio Terminal sketch.")


if __name__ == "__main__":
    main()