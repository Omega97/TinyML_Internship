"""Bucketed dual-perspective NNUE (SARDINE step C)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinymlinternship.features import FEATURE_DIM, NUM_BUCKETS


def crelu(x: torch.Tensor, clip: float = 127.0) -> torch.Tensor:
    return torch.clamp(x, min=0.0, max=clip)


class BucketedNNUE(nn.Module):
    """
    Shared L1 (844 → W) + 8 expert heads (2W → 1).

    Dual perspective: same L1 weights applied to white and black sparse inputs.
    STM ordering for concat; tanh output in [-1, +1] (expected reward).
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        num_buckets: int = NUM_BUCKETS,
        crelu_clip: float = 127.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_buckets = num_buckets
        self.crelu_clip = crelu_clip

        self.l1 = nn.Linear(feature_dim, hidden_dim, bias=True)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, 1, bias=True) for _ in range(num_buckets)]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.l1.weight, a=5**0.5)
        nn.init.zeros_(self.l1.bias)
        for expert in self.experts:
            nn.init.kaiming_uniform_(expert.weight, a=5**0.5)
            nn.init.zeros_(expert.bias)

    def l1_forward(self, features: torch.Tensor) -> torch.Tensor:
        return crelu(self.l1(features), self.crelu_clip)

    def forward(
        self,
        white_features: torch.Tensor,
        black_features: torch.Tensor,
        bucket_ids: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        white_h = self.l1_forward(white_features)
        black_h = self.l1_forward(black_features)

        stm_mask = stm_white.unsqueeze(1)
        stm_h = torch.where(stm_mask, white_h, black_h)
        opp_h = torch.where(stm_mask, black_h, white_h)
        concat = torch.cat([stm_h, opp_h], dim=1)

        expert_outs = torch.stack(
            [torch.tanh(expert(concat)) for expert in self.experts],
            dim=1,
        ).squeeze(-1)
        route = F.one_hot(bucket_ids.long(), num_classes=self.num_buckets).float()
        return (expert_outs * route).sum(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)