#!/usr/bin/env python3
import time
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
from tinymlinternship.models.value import UltraTinyValueMLP
model = UltraTinyValueMLP()
model.load_state_dict(torch.load('models/checkpoints/tiny_value_wio.pt', map_location='cpu'))
model.eval()
x = torch.randn(1, 768)
N = 1000000
t0 = time.time()
for _ in range(N):
    model(x)
dt = time.time() - t0
print(f'PC evals/sec: {N / dt:.0f}')
