"""Dual-perspective NNUE models (SARDINE F3 production + legacy multi-head)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinymlinternship.features import FEATURE_DIM, NUM_BUCKETS

Architecture = Literal["single_head", "bucketed"]


def crelu(x: torch.Tensor, clip: float = 127.0) -> torch.Tensor:
    return torch.clamp(x, min=0.0, max=clip)


def _dual_concat(
    white_features: torch.Tensor,
    black_features: torch.Tensor,
    stm_white: torch.Tensor,
    l1: nn.Linear,
    crelu_clip: float,
) -> torch.Tensor:
    """Shared L1 + CReLU on each POV; concat [STM, opponent] → 2W."""
    white_h = crelu(l1(white_features), crelu_clip)
    black_h = crelu(l1(black_features), crelu_clip)
    stm_mask = stm_white.unsqueeze(1)
    stm_h = torch.where(stm_mask, white_h, black_h)
    opp_h = torch.where(stm_mask, black_h, white_h)
    return torch.cat([stm_h, opp_h], dim=1)


class SingleHeadNNUE(nn.Module):
    """
    F3 production student: shared L1 ``844 → W`` (dual POV) → concat ``2W`` →
    one head ``2W → 1`` → tanh expected reward in ``[-1, +1]``.

    ``bucket_ids`` are accepted for a uniform train/eval call signature but are
    **not** used for routing (metadata only until §D ablation).
    """

    architecture: Architecture = "single_head"

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        crelu_clip: float = 127.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.crelu_clip = crelu_clip

        self.l1 = nn.Linear(feature_dim, hidden_dim, bias=True)
        self.head = nn.Linear(hidden_dim * 2, 1, bias=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.l1.weight, a=5**0.5)
        nn.init.zeros_(self.l1.bias)
        nn.init.kaiming_uniform_(self.head.weight, a=5**0.5)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        white_features: torch.Tensor,
        black_features: torch.Tensor,
        bucket_ids: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        del bucket_ids  # F3: ignore multi-head routing
        concat = _dual_concat(
            white_features, black_features, stm_white, self.l1, self.crelu_clip
        )
        return torch.tanh(self.head(concat)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BucketedNNUE(nn.Module):
    """
    Legacy multi-expert: shared L1 (844 → W) + N expert heads (2W → 1).

    Experimental / pilots only until §D. Prefer :class:`SingleHeadNNUE` for
    production (F3).
    """

    architecture: Architecture = "bucketed"

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
        concat = _dual_concat(
            white_features, black_features, stm_white, self.l1, self.crelu_clip
        )

        expert_outs = torch.stack(
            [torch.tanh(expert(concat)) for expert in self.experts],
            dim=1,
        ).squeeze(-1)
        route = F.one_hot(bucket_ids.long(), num_classes=self.num_buckets).float()
        return (expert_outs * route).sum(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_nnue(
    architecture: Architecture = "single_head",
    *,
    hidden_dim: int = 128,
    num_buckets: int = NUM_BUCKETS,
    feature_dim: int = FEATURE_DIM,
) -> SingleHeadNNUE | BucketedNNUE:
    """Factory for train / eval (F3 default = single head)."""
    if architecture == "single_head":
        return SingleHeadNNUE(feature_dim=feature_dim, hidden_dim=hidden_dim)
    if architecture == "bucketed":
        return BucketedNNUE(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_buckets=num_buckets,
        )
    raise ValueError(f"unknown architecture: {architecture!r}")


def infer_architecture(state_dict: dict[str, torch.Tensor]) -> Architecture:
    """Infer model class from a checkpoint ``model_state_dict``."""
    keys = state_dict.keys()
    if any(k.startswith("experts.") for k in keys):
        return "bucketed"
    if any(k.startswith("head.") for k in keys):
        return "single_head"
    raise ValueError(
        "cannot infer NNUE architecture from state_dict keys "
        f"(sample: {list(keys)[:8]})"
    )