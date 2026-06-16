#!/usr/bin/env python3
"""
Wio Terminal D51R - specific tiny chess model pipeline.

Hardware target:
  - 512 KB internal flash
  - 192 KB RAM
  - 4 MB external SPI flash (QSPI)

Goal: Produce a model + C header small enough to actually fit and run inference
      (ideally the weights live in external flash or comfortably in internal).

Recommended models:
  - TinyValueMLP (768→32→16→1)   ~25k params → ~25 KB int8
  - UltraTinyValueMLP (768→16→8→1) ~12.5k params → ~13 KB int8

These are small enough for:
  - Storing the weights/blob in the 4 MB external flash
  - Very small TFLM arena or even a hand-written forward pass (recommended for 192 KB RAM)
  - Shallow search (depth 2-4) on the device

The script:
  1. Builds the tiny value net
  2. (Optional but recommended) Quick supervised training on game outcomes from the dataset
  3. Post-training quantization to int8 (per-tensor absmax for simplicity)
  4. Exports a compact binary blob (weights + scales)
  5. Generates a C header using bin_to_c_header.py (or direct)
  6. Prints fit analysis for the Wio Terminal

Usage (from project root with good Python):
    py -3.12 scripts/prepare_wio_tiny.py --model ultra --train --samples 800

After this you will have a header you can actually flash.
"""

import argparse
import csv
import struct
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Project imports (clean package names + runtime path hack for direct execution)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tinymlinternship.config.settings import (
    CHECKPOINTS_DIR,
    EXPORTED_DIR,
    ARDUINO_MODELS_DIR,
    RAW_DATA_DIR,
)
from tinymlinternship.models.value import TinyValueMLP, UltraTinyValueMLP
from tinymlinternship.datasets.featurizer import fen_to_tensor
import chess


def ensure_dirs():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)
    ARDUINO_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_positions(
    csv_path: Path, max_games: int = 500, positions_per_game: int = 3
) -> List[Tuple[torch.Tensor, float]]:
    """
    Very lightweight data loader.
    Reads the Kaggle games.csv, replays games with python-chess,
    takes a few positions per game + the final outcome as weak supervision.
    Returns list of (768-vector, label in [-1, +1]).
    """
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found. Using purely random weights.")
        return []

    print(f"Loading training positions from {csv_path} (max_games={max_games})...")

    examples: List[Tuple[torch.Tensor, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_games:
                break
            moves_str = row.get("moves", "")
            winner = row.get("winner", "draw").lower()

            if not moves_str:
                continue

            # Map outcome to label for side-to-move (very weak but better than nothing)
            if winner == "white":
                outcome = 1.0
            elif winner == "black":
                outcome = -1.0
            else:
                outcome = 0.0

            try:
                board = chess.Board()
                move_list = moves_str.strip().split()

                # Take a few positions spread through the game + near the end
                n_moves = len(move_list)
                indices = set()
                if n_moves > 0:
                    indices.add(min(3, n_moves - 1))  # early
                if n_moves > 8:
                    indices.add(n_moves // 2)
                for k in range(1, positions_per_game + 1):
                    idx = max(0, n_moves - k * 3)
                    indices.add(min(idx, n_moves - 1))

                for idx in sorted(indices):
                    # Replay up to this move
                    b = chess.Board()
                    for m in move_list[: idx + 1]:
                        try:
                            b.push_san(m)
                        except:
                            break

                    # Featurize to 768 vector (exactly what the tiny MLP expects)
                    vec = fen_to_tensor(b, flatten=True)  # (768,)
                    # Label from the perspective of the side that just moved? Approximate.
                    # For simplicity we use the final game outcome as the label for all sampled positions.
                    label = outcome if b.turn == chess.WHITE else -outcome
                    examples.append((vec, label))
            except Exception:
                continue

    print(f"  Collected {len(examples)} position→outcome examples")
    return examples


def train_tiny_value(
    model: nn.Module,
    examples: List[Tuple[torch.Tensor, float]],
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 0.01,
) -> None:
    """Quick and dirty supervised training on outcomes."""
    if not examples:
        print("No training examples — skipping training (random weights).")
        return

    print(f"Quick training for {epochs} epochs on {len(examples)} examples...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        # Simple batching
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            x = torch.stack([e[0] for e in batch])
            y = torch.tensor([e[1] for e in batch], dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        avg = total_loss / len(examples)
        print(f"  Epoch {ep+1}/{epochs}  loss={avg:.4f}")

    model.eval()
    print("Training done.")


def quantize_to_int8(state_dict: dict) -> Tuple[dict, dict]:
    """
    Very simple per-tensor absmax quantization to int8.
    Returns (quantized_state_dict, scales)
    """
    q_state = {}
    scales = {}
    for name, tensor in state_dict.items():
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        abs_max = tensor.abs().max().clamp(min=1e-8)
        scale = abs_max / 127.0
        q = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
        q_state[name] = q
        scales[name] = scale.item()
    return q_state, scales


def export_compact_blob(
    model: nn.Module, scales: dict | None = None, model_name: str = "tiny_value_wio"
) -> Path:
    """
    Export a very small binary: int8 weights + optional per-tensor scales (float32).
    This is tiny and easy to load from external flash on the Wio.
    """
    ensure_dirs()
    blob_path = EXPORTED_DIR / f"{model_name}_int8.bin"

    sd = model.state_dict()
    q_sd, scales = quantize_to_int8(sd)

    with blob_path.open("wb") as f:
        # Simple header: magic + num_tensors + per-tensor metadata
        f.write(b"WIO1")  # magic
        f.write(struct.pack("<I", len(q_sd)))

        for name, q_tensor in q_sd.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            shape = q_tensor.shape
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", d))
            data = q_tensor.cpu().numpy().tobytes()
            f.write(struct.pack("<I", len(data)))
            f.write(data)
            sc = scales.get(name, 1.0)
            f.write(struct.pack("<f", sc))

    print(f"Exported compact int8 blob: {blob_path} ({blob_path.stat().st_size} bytes)")
    return blob_path


def main():
    parser = argparse.ArgumentParser(description="Prepare ultra-tiny chess value net for Wio Terminal D51R")
    parser.add_argument("--model", choices=["tiny", "ultra"], default="ultra",
                        help="tiny=768-32-16-1 (~25k params), ultra=768-16-8-1 (~12.5k params)")
    parser.add_argument("--train", action="store_true", help="Run quick outcome-based training")
    parser.add_argument("--max-games", type=int, default=600, help="How many games to sample for training data")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--name", default="tiny_value_wio")
    args = parser.parse_args()

    ensure_dirs()

    # 1. Build the model sized for the hardware
    if args.model == "ultra":
        model = UltraTinyValueMLP().eval()
        print("Using UltraTinyValueMLP (768→16→8→1) — extremely MCU friendly")
    else:
        model = TinyValueMLP(hidden1=32, hidden2=16).eval()
        print("Using TinyValueMLP (768→32→16→1)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}  (int8 ≈ {total_params/1024:.1f} KB)")

    # 2. Optional training on real(ish) data
    examples = []
    if args.train:
        csv_path = RAW_DATA_DIR / "games.csv"
        examples = load_training_positions(csv_path, max_games=args.max_games)
        train_tiny_value(model, examples, epochs=args.epochs)

    # 3. Save full precision checkpoint (for further work)
    ckpt_path = CHECKPOINTS_DIR / f"{args.name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved full checkpoint: {ckpt_path}")

    # 4. Quantize + export tiny binary blob
    blob_path = export_compact_blob(model, model_name=args.name)

    # 5. Produce the C header that you can actually include / store on external flash
    # We embed the compact binary blob (best for Wio)
    import subprocess
    header_path = ARDUINO_MODELS_DIR / f"{args.name}_model.h"
    try:
        subprocess.check_call(
            [
                sys.executable.replace("python.exe", "py.exe") if "python.exe" in sys.executable else "py",
                "-3.12",
                "scripts/bin_to_c_header.py",
                str(blob_path),
                "--var-name",
                "g_wio_chess_model",
                "--out",
                str(header_path),
            ]
        )
    except Exception:
        # Fallback: direct generation
        data = blob_path.read_bytes()
        lines = []
        for i in range(0, len(data), 12):
            chunk = data[i : i + 12]
            lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
        header_path.write_text(
            f"// Wio Terminal tiny chess model\n"
            f"const unsigned char g_wio_chess_model[] = {{\n{chr(10).join(lines)}\n}};\n"
            f"const unsigned int g_wio_chess_model_len = {len(data)};\n"
        )

    hsize = header_path.stat().st_size
    print(f"\n✅ Generated hardware header: {header_path}")
    print(f"   Source size: {hsize / 1024:.1f} KB  (embeds {blob_path.stat().st_size} bytes of model)")

    # 6. Fit report for the actual board
    print("\n" + "=" * 60)
    print("WIO TERMINAL D51R FIT ANALYSIS")
    print("=" * 60)
    print(f"Model params (int8): ~{total_params/1024:.1f} KB")
    print(f"Model binary blob  : {blob_path.stat().st_size / 1024:.1f} KB")
    print(f"C header (source)  : {hsize / 1024:.1f} KB (this is text; the data in flash will be the blob size)")
    print()
    print("Internal flash: 512 KB total")
    print("  → Your firmware + Arduino libs + this header data easily fits if you put the blob")
    print("    in the 4 MB external QSPI flash (recommended pattern for Wio Terminal).")
    print()
    print("RAM (192 KB):")
    print("  → Activations for this net are tiny (a few hundred bytes).")
    print("  → You can run a hand-written forward pass in < 5-10 KB RAM or use stripped TFLM.")
    print("  → Plenty of room left for a small negamax search + board representation.")
    print()
    print("Recommended next steps for the Wio:")
    print("  - Store the model blob on the external 4 MB flash (use Seeed QSPI or SdFat + SPI flash lib).")
    print("  - Implement a tiny C++ forward pass (or port a minimal TFLM).")
    print("  - Use the value for shallow search (depth 2-3 is realistic at 120 MHz).")
    print("  - See NOTES/hardware.md for Wio specific notes and the general load-model-howto.")
    print("=" * 60)


if __name__ == "__main__":
    main()
