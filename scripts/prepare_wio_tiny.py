#!/usr/bin/env python3
"""
Wio Terminal D51R - specific tiny chess model pipeline.

Hardware target:
  - 512 KB internal flash
  - 192 KB RAM
  - 4 MB external SPI flash (QSPI)

Goal: Produce a model + C header small enough to actually fit and run inference.
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
    WIO_SKETCH_DIR,
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

            if winner == "white":
                outcome = 1.0
            elif winner == "black":
                outcome = -1.0
            else:
                outcome = 0.0

            try:
                board = chess.Board()
                move_list = moves_str.strip().split()

                n_moves = len(move_list)
                indices = set()
                if n_moves > 0:
                    indices.add(min(3, n_moves - 1))
                if n_moves > 8:
                    indices.add(n_moves // 2)
                for k in range(1, positions_per_game + 1):
                    idx = max(0, n_moves - k * 3)
                    indices.add(min(idx, n_moves - 1))

                for idx in sorted(indices):
                    b = chess.Board()
                    for m in move_list[: idx + 1]:
                        try:
                            b.push_san(m)
                        except:
                            break

                    vec = fen_to_tensor(b, flatten=True)
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
    """Very simple per-tensor absmax quantization to int8."""
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
    model: nn.Module, q_sd: dict, scales: dict, model_name: str = "tiny_value_wio"
) -> Path:
    """Export standard raw binary fallback metadata."""
    ensure_dirs()
    blob_path = EXPORTED_DIR / f"{model_name}_int8.bin"

    with blob_path.open("wb") as f:
        f.write(b"WIO1")
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

    print(f"Exported fallback int8 blob: {blob_path} ({blob_path.stat().st_size} bytes)")
    return blob_path


def export_structured_c_header(
    model: nn.Module, q_sd: dict, scales: dict, header_path: Path
) -> None:
    """
    Generates a structured C header containing both network architecture dimensions
    and the quantized model parameters (weights, biases, and scales).
    """
    sd = model.state_dict()
    weight_keys = [k for k in sd.keys() if "weight" in k]
    bias_keys = [k for k in sd.keys() if "bias" in k]

    if len(weight_keys) != 3 or len(bias_keys) != 3:
        raise ValueError(f"Expected a 3-layer MLP architecture, found {len(weight_keys)} layers.")

    # Dynamically read layer shapes from the PyTorch model
    fc1_out, fc1_in = sd[weight_keys[0]].shape
    fc2_out, fc2_in = sd[weight_keys[1]].shape
    fc3_out, fc3_in = sd[weight_keys[2]].shape

    lines = [
        "// Auto-generated int8 weights + scales with architecture layout",
        "#ifndef WIO_INT8_WEIGHTS_H",
        "#define WIO_INT8_WEIGHTS_H",
        "",
        "#include <avr/pgmspace.h>",
        "",
        "// ============================================================================",
        "// Network Architecture Dimensions",
        "// ============================================================================",
        f"#define FC1_IN_DIM    {fc1_in}",
        f"#define FC1_OUT_DIM   {fc1_out}",
        "",
        f"#define FC2_IN_DIM    {fc2_in}",
        f"#define FC2_OUT_DIM   {fc2_out}",
        "",
        f"#define FC3_IN_DIM    {fc3_in}",
        f"#define FC3_OUT_DIM   {fc3_out}",
        "",
        "// ============================================================================",
        "// Quantized Tensors",
        "// ============================================================================",
    ]

    def format_int8_array(c_name, tensor):
        flat_data = tensor.cpu().flatten().tolist()
        array_lines = []
        for i in range(0, len(flat_data), 16):
            chunk = flat_data[i : i + 16]
            array_lines.append("  " + ", ".join(str(x) for x in chunk))
        return f"const int8_t {c_name}[] PROGMEM = {{\n" + ",\n".join(array_lines) + "\n};"

    # Map parameters to clean structured arrays
    c_variable_pairs = [
        ("fc1_w", "fc1_b", weight_keys[0], bias_keys[0]),
        ("fc2_w", "fc2_b", weight_keys[1], bias_keys[1]),
        ("fc3_w", "fc3_b", weight_keys[2], bias_keys[2]),
    ]

    for idx, (w_name, b_name, w_key, b_key) in enumerate(c_variable_pairs, 1):
        lines.append(f"// --- Layer {idx} ---")
        lines.append(format_int8_array(w_name, q_sd[w_key]))
        lines.append(f"const float {w_name}_scale = {scales[w_key]:.18f}f;")
        lines.append("")
        lines.append(format_int8_array(b_name, q_sd[b_key]))
        lines.append(f"const float {b_name}_scale = {scales[b_key]:.18f}f;")
        lines.append("")

    lines.append("const float input_scale = 1.0f;")
    lines.append("")
    lines.append("#endif // WIO_INT8_WEIGHTS_H")

    header_path.write_text("\n".join(lines))


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

    if args.model == "ultra":
        model = UltraTinyValueMLP().eval()
        print("Using UltraTinyValueMLP (768→16→8→1) — extremely MCU friendly")
    else:
        model = TinyValueMLP(hidden1=32, hidden2=16).eval()
        print("Using TinyValueMLP (768→32→16→1)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}  (int8 ≈ {total_params/1024:.1f} KB)")

    # 2. Optional training
    if args.train:
        csv_path = RAW_DATA_DIR / "games.csv"
        examples = load_training_positions(csv_path, max_games=args.max_games)
        train_tiny_value(model, examples, epochs=args.epochs)

    # 3. Save full precision checkpoint
    ckpt_path = CHECKPOINTS_DIR / f"{args.name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved full checkpoint: {ckpt_path}")

    # 4. Quantize parameters
    sd = model.state_dict()
    q_sd, scales = quantize_to_int8(sd)

    # Keep backup binary blob file
    blob_path = export_compact_blob(model, q_sd, scales, model_name=args.name)

    # 5. Produce the C header (architecture macros + int8 weight arrays) for the sketch
    # Maps 'ultra' → nano filename, 'tiny' → tiny filename (matches WEIGHTS_FILE in .ino)
    model_suffix = "nano" if args.model == "ultra" else "tiny"
    header_filename = f"wio_int8_weights_{model_suffix}.h"
    WIO_SKETCH_DIR.mkdir(parents=True, exist_ok=True)
    header_path = WIO_SKETCH_DIR / header_filename

    export_structured_c_header(model, q_sd, scales, header_path)

    hsize = header_path.stat().st_size
    print(f"\n✅ Generated sketch header: {header_path}")
    print(f"   Size: {hsize / 1024:.1f} KB (FC*_IN/OUT_DIM macros + int8 PROGMEM arrays)")

    # 6. Fit report
    print("\n" + "=" * 60)
    print("WIO TERMINAL D51R FIT ANALYSIS")
    print("=" * 60)
    print(f"Model params (int8): ~{total_params/1024:.1f} KB")
    print(f"C header filename  : {header_filename}")
    print(f"C header size      : {hsize / 1024:.1f} KB")
    print()
    print("Internal flash: 512 KB total")
    print("  → Your dynamic loop structures in the .ino will adapt to this file auto-generatively.")
    print("=" * 60)


if __name__ == "__main__":
    main()