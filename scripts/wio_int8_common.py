"""Shared int8 export helpers for Wio Terminal value-net prepare scripts."""

import csv
import struct
from pathlib import Path
from typing import List, Tuple

import chess
import torch
import torch.nn as nn
import torch.optim as optim

from tinymlinternship.config.settings import (
    ARDUINO_MODELS_DIR,
    CHECKPOINTS_DIR,
    EXPORTED_DIR,
    RAW_DATA_DIR,
)
from tinymlinternship.datasets.featurizer import fen_to_tensor


def ensure_dirs() -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)
    ARDUINO_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_positions(
    csv_path: Path, max_games: int = 500, positions_per_game: int = 3
) -> List[Tuple[torch.Tensor, float]]:
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
                        except Exception:
                            break

                    vec = fen_to_tensor(b, flatten=True)
                    label = outcome if b.turn == chess.WHITE else -outcome
                    examples.append((vec, label))
            except Exception:
                continue

    print(f"  Collected {len(examples)} position→outcome examples")
    return examples


def train_value_net(
    model: nn.Module,
    examples: List[Tuple[torch.Tensor, float]],
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 0.01,
) -> None:
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
    q_sd: dict, scales: dict, model_name: str
) -> Path:
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
    model: nn.Module,
    q_sd: dict,
    scales: dict,
    header_path: Path,
    header_guard: str,
) -> None:
    sd = model.state_dict()
    weight_keys = [k for k in sd.keys() if "weight" in k]
    bias_keys = [k for k in sd.keys() if "bias" in k]

    if len(weight_keys) != 3 or len(bias_keys) != 3:
        raise ValueError(f"Expected a 3-layer MLP architecture, found {len(weight_keys)} layers.")

    fc1_out, fc1_in = sd[weight_keys[0]].shape
    fc2_out, fc2_in = sd[weight_keys[1]].shape
    fc3_out, fc3_in = sd[weight_keys[2]].shape

    lines = [
        "// Auto-generated int8 weights + scales with architecture layout",
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "#include <avr/pgmspace.h>",
        "",
        "// Network Architecture Dimensions",
        f"#define FC1_IN_DIM    {fc1_in}",
        f"#define FC1_OUT_DIM   {fc1_out}",
        "",
        f"#define FC2_IN_DIM    {fc2_in}",
        f"#define FC2_OUT_DIM   {fc2_out}",
        "",
        f"#define FC3_IN_DIM    {fc3_in}",
        f"#define FC3_OUT_DIM   {fc3_out}",
        "",
        "// Quantized Tensors",
    ]

    def format_int8_array(c_name, tensor):
        flat_data = tensor.cpu().flatten().tolist()
        array_lines = []
        for i in range(0, len(flat_data), 16):
            chunk = flat_data[i : i + 16]
            array_lines.append("  " + ", ".join(str(x) for x in chunk))
        return f"const int8_t {c_name}[] PROGMEM = {{\n" + ",\n".join(array_lines) + "\n};"

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
    lines.append(f"#endif // {header_guard}")

    header_path.write_text("\n".join(lines))


def run_training_if_requested(model: nn.Module, train: bool, max_games: int, epochs: int) -> None:
    if not train:
        return
    csv_path = RAW_DATA_DIR / "games.csv"
    examples = load_training_positions(csv_path, max_games=max_games)
    train_value_net(model, examples, epochs=epochs)