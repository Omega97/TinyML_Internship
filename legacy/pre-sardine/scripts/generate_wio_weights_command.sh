#!/bin/bash
# Reproducible one-liner to generate wio_weights.h from the tiny value net checkpoint.
# Run with: bash scripts/generate_wio_weights_command.sh
# Or copy the python -c part and run directly.

python -c '
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
import torch
from tinymlinternship.models.value import UltraTinyValueMLP
model = UltraTinyValueMLP()
model.load_state_dict(torch.load("models/checkpoints/tiny_value_wio.pt", map_location="cpu"))
def c_array(name, tensor):
    flat = tensor.detach().cpu().float().flatten().tolist()
    lines = []
    for i in range(0, len(flat), 8):
        chunk = ", ".join(f"{v:.8f}f" for v in flat[i:i+8])
        lines.append("  " + chunk)
    return f"const float {name}[] PROGMEM = {{\n" + ",\n".join(lines) + "\n};"
print("// UltraTinyValueMLP weights (768->16->8->1) for Wio")
print(c_array("fc1_w", model.fc1.weight))
print(c_array("fc1_b", model.fc1.bias))
print(c_array("fc2_w", model.fc2.weight))
print(c_array("fc2_b", model.fc2.bias))
print(c_array("fc3_w", model.fc3.weight))
print(c_array("fc3_b", model.fc3.bias))
' > wio_weights.h

echo "Generated wio_weights.h"