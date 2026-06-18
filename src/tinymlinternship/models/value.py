"""
Tiny value networks designed for severely constrained MCUs like the Wio Terminal D51R
(512 KB flash, 192 KB RAM, 4 MB external SPI flash).

These are inspired by the minimal NNUE-style models discussed in PROJECT.md:
- 768 → 32 → 16 → 1  (or even smaller)
- Quantizable to int8
- Very small activation memory
- Suitable for shallow search (negamax/minimax depth 2-4) on the device.

The model can live in internal flash or be loaded from the external 4 MB SPI flash.
"""

import torch
import torch.nn as nn
from typing import Tuple


class TinyValueMLP(nn.Module):
    """
    Extremely small value network for Wio Terminal class hardware.

    Input:
        - 768-dimensional vector (flattened 12x8x8 board from featurizer.flatten=True)
          or any 768 piece-square like features.

    Output:
        - Scalar in approx. [-1, +1] via Tanh (positive = good for side to move).
          Can be scaled to centipawns if desired.

    Sizes (example 768-32-16-1):
        ~25k parameters → ~25 KB in int8 (perfect for 512 KB internal flash).

    Even smaller variant: hidden1=16, hidden2=8 → < 13k params.
    """

    def __init__(
        self,
        input_size: int = 768,
        hidden1: int = 32,
        hidden2: int = 16,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 768)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x.squeeze(-1)  # (B,)

    def get_config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "type": "TinyValueMLP",
        }


def create_tiny_value(
    input_size: int = 768,
    hidden1: int = 32,
    hidden2: int = 16,
    model_name: str = "tiny_value_wio",
) -> Tuple[nn.Module, str]:
    """
    Factory that returns a ready-to-use tiny value net sized for Wio Terminal.
    """
    model = TinyValueMLP(input_size=input_size, hidden1=hidden1, hidden2=hidden2).eval()
    return model, model_name


class SmallValueMLP(TinyValueMLP):
    """768 → 64 → 32 → 1   (~51k params)"""

    def __init__(self, input_size: int = 768):
        super().__init__(input_size=input_size, hidden1=64, hidden2=32)


class MediumValueMLP(TinyValueMLP):
    """768 → 128 → 64 → 1   (~104k params, 2× small)"""

    def __init__(self, input_size: int = 768):
        super().__init__(input_size=input_size, hidden1=128, hidden2=64)


# Even more aggressive ultra-tiny variant (for maximum headroom on 192 KB RAM)
class UltraTinyValueMLP(nn.Module):
    """768 → 16 → 8 → 1   (~12.5k params)"""

    def __init__(self, input_size: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x.squeeze(-1)


if __name__ == "__main__":
    # Quick smoke test
    m, name = create_tiny_value()
    print(f"Created {name}")
    print(m)
    total = sum(p.numel() for p in m.parameters())
    print(f"Params: {total:,} → ~{total/1024:.1f} KB float32, ~{total/1024/4:.1f} KB int8 (ideal for Wio)")
    x = torch.randn(1, 768)
    y = m(x)
    print("Output example:", y)
