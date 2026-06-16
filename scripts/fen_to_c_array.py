#!/usr/bin/env python3
"""
Small helper to turn a FEN into a C header file containing the 768-element
input array for the tiny value nets (ready for Arduino/Wio).

This avoids pasting huge arrays into your .ino. Just re-run the script with
a new FEN and --output pointing inside your sketch folder.

Usage:
    python scripts/fen_to_c_array.py "FEN" --output Arduino/Wio_TinyValueTest/fen_input.h
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.datasets.featurizer import fen_to_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Generate a .h file with the 768 board feature array from a FEN."
    )
    parser.add_argument("fen", help="FEN string of the chess position")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fen_input.h"),
        help="Output .h file path (default: fen_input.h). "
             "Point this inside your Arduino sketch folder, e.g. "
             "Arduino/Wio_TinyValueTest/fen_input.h",
    )
    args = parser.parse_args()

    vec = fen_to_tensor(args.fen, flatten=True)

    lines = [
        f"// Auto-generated from FEN: {args.fen}",
        "// 768-element board vector for UltraTinyValueMLP / tiny value nets.",
        "// Include this in your .ino with: #include \"fen_input.h\"",
        "// Then use: forward(input);  (or whatever your array name is)",
        "",
        "const float input[768] PROGMEM = {",
    ]

    for i in range(0, 768, 16):
        chunk = ", ".join(f"{x:.1f}f" for x in vec[i : i + 16])
        comma = "," if i + 16 < 768 else ""
        lines.append(f"  {chunk}{comma}")

    lines.append("};")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✅ Wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")
    print("   Now #include it in your sketch and use the 'input' array.")


if __name__ == "__main__":
    main()