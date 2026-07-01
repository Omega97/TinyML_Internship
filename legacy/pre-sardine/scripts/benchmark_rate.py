#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: E402, F401

from tinymlinternship.config.settings import CHECKPOINTS_DIR
from tinymlinternship.models.value import UltraTinyValueMLP

model = UltraTinyValueMLP()
model.load_state_dict(torch.load(CHECKPOINTS_DIR / "tiny_value_wio.pt", map_location="cpu"))
model.eval()
x = torch.randn(1, 768)
N = 1000000
t0 = time.time()
for _ in range(N):
    model(x)
dt = time.time() - t0
print(f'PC evals/sec: {N / dt:.0f}')
