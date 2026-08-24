"""Dual-POV two-hidden NNUE (Goal §2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinymlinternship.features import FEATURE_DIM


def crelu(x: torch.Tensor, clip: float = 127.0) -> torch.Tensor:
    return torch.clamp(x, min=0.0, max=clip)


class DualHiddenNNUE(nn.Module):
    """
    Shared L1 ``844 → W`` on each POV, CReLU, concat ``[STM, opp]`` → ``2W``,
    L2 ``2W → H`` CReLU, head ``H → 1`` tanh. Target is White-POV expected reward.
    """

    architecture = "dual_hidden"

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        hidden2_dim: int = 128,
        crelu_clip: float = 127.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden2_dim = hidden2_dim
        self.crelu_clip = crelu_clip
        self.l1 = nn.Linear(feature_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim * 2, hidden2_dim, bias=True)
        self.head = nn.Linear(hidden2_dim, 1, bias=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in (self.l1, self.l2, self.head):
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
            nn.init.zeros_(layer.bias)

    def l1_dense(self, features: torch.Tensor) -> torch.Tensor:
        return crelu(self.l1(features), self.crelu_clip)

    def l1_sparse(self, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``indices`` (B, K) int64, ``mask`` (B, K) bool. Pad slots must be masked."""
        safe = indices.long().clamp(min=0, max=self.feature_dim - 1)
        gathered = F.embedding(safe, self.l1.weight.t())
        gathered = gathered * mask.unsqueeze(-1).to(dtype=gathered.dtype)
        return crelu(gathered.sum(dim=1) + self.l1.bias, self.crelu_clip)

    def _head_from_accumulators(
        self,
        white_h: torch.Tensor,
        black_h: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        stm_mask = stm_white.unsqueeze(1)
        stm_h = torch.where(stm_mask, white_h, black_h)
        opp_h = torch.where(stm_mask, black_h, white_h)
        concat = torch.cat([stm_h, opp_h], dim=1)
        h2 = crelu(self.l2(concat), self.crelu_clip)
        return torch.tanh(self.head(h2)).squeeze(-1)

    def forward(
        self,
        white_features: torch.Tensor,
        black_features: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        return self._head_from_accumulators(
            self.l1_dense(white_features),
            self.l1_dense(black_features),
            stm_white,
        )

    def forward_sparse(
        self,
        white_idx: torch.Tensor,
        white_mask: torch.Tensor,
        black_idx: torch.Tensor,
        black_mask: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        return self._head_from_accumulators(
            self.l1_sparse(white_idx, white_mask),
            self.l1_sparse(black_idx, black_mask),
            stm_white,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
