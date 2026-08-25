"""Dual-POV two-hidden NNUE (Goal §2) with a 3-way STM WDL head."""

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
    L2 ``2W → H`` CReLU, head ``H → 3`` logits. Softmax is STM ``(W, D, L)``.
    """

    architecture = "dual_hidden_wdl"
    n_outputs = 3

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
        self.head = nn.Linear(hidden2_dim, 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
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
        return self.head(h2)  # STM WDL logits; softmax in loss / probabilities()

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

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return self.softmax(logits)

    def stm_value(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self.probabilities(logits)
        return probs[..., 0] - probs[..., 2]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearWDLNNUE(nn.Module):
    """No hidden layers: concat ``[STM ‖ opp]`` ``2×844 → 3`` logits, softmax STM WDL."""

    architecture = "linear_wdl"
    n_outputs = 3

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.head = nn.Linear(feature_dim * 2, 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.kaiming_uniform_(self.head.weight, a=5**0.5)
        nn.init.zeros_(self.head.bias)

    def _stm_opp(
        self,
        white: torch.Tensor,
        black: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = stm_white.unsqueeze(1)
        stm = torch.where(mask, white, black)
        opp = torch.where(mask, black, white)
        return stm, opp

    def forward(
        self,
        white_features: torch.Tensor,
        black_features: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        stm, opp = self._stm_opp(white_features, black_features, stm_white)
        return self.head(torch.cat([stm, opp], dim=1))

    def _sparse_dot(self, indices: torch.Tensor, mask: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        safe = indices.long().clamp(min=0, max=self.feature_dim - 1)
        gathered = F.embedding(safe, weight_t)
        gathered = gathered * mask.unsqueeze(-1).to(dtype=gathered.dtype)
        return gathered.sum(dim=1)

    def forward_sparse(
        self,
        white_idx: torch.Tensor,
        white_mask: torch.Tensor,
        black_idx: torch.Tensor,
        black_mask: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        stm_idx, opp_idx = self._stm_opp(white_idx, black_idx, stm_white)
        stm_mask, opp_mask = self._stm_opp(white_mask, black_mask, stm_white)
        w = self.head.weight
        stm_part = self._sparse_dot(stm_idx, stm_mask, w[:, : self.feature_dim].t())
        opp_part = self._sparse_dot(opp_idx, opp_mask, w[:, self.feature_dim :].t())
        return stm_part + opp_part + self.head.bias

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return self.softmax(logits)

    def stm_value(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self.probabilities(logits)
        return probs[..., 0] - probs[..., 2]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MediumWDLNNUE(nn.Module):
    """One hidden layer: concat ``[STM ‖ opp]`` ``2×844 → H → 3`` logits, CReLU, softmax STM WDL."""

    architecture = "medium_wdl"
    n_outputs = 3

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 20,
        crelu_clip: float = 127.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.crelu_clip = crelu_clip
        self.l1 = nn.Linear(feature_dim * 2, hidden_dim, bias=True)
        self.head = nn.Linear(hidden_dim, 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.kaiming_uniform_(self.l1.weight, a=5**0.5)
        nn.init.zeros_(self.l1.bias)
        nn.init.kaiming_uniform_(self.head.weight, a=5**0.5)
        nn.init.zeros_(self.head.bias)

    def _stm_opp(
        self,
        white: torch.Tensor,
        black: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = stm_white.unsqueeze(1)
        stm = torch.where(mask, white, black)
        opp = torch.where(mask, black, white)
        return stm, opp

    def forward(
        self,
        white_features: torch.Tensor,
        black_features: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        stm, opp = self._stm_opp(white_features, black_features, stm_white)
        hidden = crelu(self.l1(torch.cat([stm, opp], dim=1)), self.crelu_clip)
        return self.head(hidden)

    def _sparse_dot(self, indices: torch.Tensor, mask: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        safe = indices.long().clamp(min=0, max=self.feature_dim - 1)
        gathered = F.embedding(safe, weight_t)
        gathered = gathered * mask.unsqueeze(-1).to(dtype=gathered.dtype)
        return gathered.sum(dim=1)

    def forward_sparse(
        self,
        white_idx: torch.Tensor,
        white_mask: torch.Tensor,
        black_idx: torch.Tensor,
        black_mask: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        stm_idx, opp_idx = self._stm_opp(white_idx, black_idx, stm_white)
        stm_mask, opp_mask = self._stm_opp(white_mask, black_mask, stm_white)
        w = self.l1.weight
        stm_h = self._sparse_dot(stm_idx, stm_mask, w[:, : self.feature_dim].t())
        opp_h = self._sparse_dot(opp_idx, opp_mask, w[:, self.feature_dim :].t())
        hidden = crelu(stm_h + opp_h + self.l1.bias, self.crelu_clip)
        return self.head(hidden)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return self.softmax(logits)

    def stm_value(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self.probabilities(logits)
        return probs[..., 0] - probs[..., 2]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
