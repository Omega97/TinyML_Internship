#!/usr/bin/env python3
"""
Small utility: Convert any binary file (model, .tflite, onnx data, TorchScript, etc.)
into a compact C header suitable for embedding in Arduino / ESP32 firmware.

Usage:
    py -3.12 scripts/bin_to_c_header.py models/exported/tiny_chess_policy_lab.ts.pt \
        --var-name g_tiny_chess_model \
        --out models/arduino/models/tiny_chess_policy_model.h

This is the standard, flash-efficient way to ship models to microcontrollers.
"""

import argparse
from pathlib import Path


def binary_to_c_array(data: bytes, var_name: str, cols: int = 12) -> str:
    lines = []
    for i in range(0, len(data), cols):
        chunk = data[i : i + cols]
        hex_line = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append("  " + hex_line)
    return ",\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Binary file → C header for embedded use")
    parser.add_argument("input", type=Path, help="Input binary file (e.g. .ts.pt, .tflite, onnx.data)")
    parser.add_argument("--var-name", default="g_model", help="C variable name")
    parser.add_argument("--out", type=Path, default=None, help="Output .h path (default: <input>.h next to input)")
    parser.add_argument("--cols", type=int, default=12, help="Bytes per line")
    args = parser.parse_args()

    data = args.input.read_bytes()
    array_init = binary_to_c_array(data, args.var_name, args.cols)

    if args.out is None:
        args.out = args.input.with_suffix(".h")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    guard = args.var_name.upper().replace(" ", "_") + "_H"
    content = f'''// Auto-generated with scripts/bin_to_c_header.py
// Source: {args.input.name}
// Model data size: {len(data)} bytes

#ifndef {guard}
#define {guard}

const unsigned char {args.var_name}[] = {{
{array_init}
}};

const unsigned int {args.var_name}_len = {len(data)};

#endif  // {guard}
'''

    args.out.write_text(content)
    print(f"✅ Wrote {args.out}")
    print(f"   Variable: {args.var_name}")
    print(f"   Length  : {len(data)} bytes ({len(data)/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
