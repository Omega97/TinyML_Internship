"""Linear dispatcher + dual-POV Mixture-of-Experts NNUE (Goal §3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinymlinternship.features import FEATURE_DIM
from tinymlinternship.nnue.model import DualHiddenNNUE, crelu
from tinymlinternship.nnue.sample_gradients import head_parameter_dim


def load_dual_hidden_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
    hidden_dim: int | None = None,
    hidden2_dim: int | None = None,
) -> DualHiddenNNUE:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"{path} is not a DualHidden NNUE checkpoint")
    w = int(payload.get("hidden_dim", hidden_dim or 128))
    h2 = int(payload.get("hidden2_dim", hidden2_dim or 256))
    if hidden_dim is not None:
        w = int(hidden_dim)
    if hidden2_dim is not None:
        h2 = int(hidden2_dim)
    model = DualHiddenNNUE(feature_dim=FEATURE_DIM, hidden_dim=w, hidden2_dim=h2)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device)


def _clone_linear(src: nn.Linear) -> nn.Linear:
    layer = nn.Linear(src.in_features, src.out_features, bias=src.bias is not None)
    layer.load_state_dict(src.state_dict())
    return layer


class LinearDispatcher(nn.Module):
    """``z = W h + b``. Train with softmax CE; infer with argmax (no softmax)."""

    def __init__(self, in_dim: int, n_clusters: int) -> None:
        super().__init__()
        if int(n_clusters) < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        self.in_dim = int(in_dim)
        self.n_clusters = int(n_clusters)
        self.linear = nn.Linear(self.in_dim, self.n_clusters, bias=True)
        nn.init.kaiming_uniform_(self.linear.weight, a=5**0.5)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward(h).argmax(dim=-1)


class MLPDispatcher(nn.Module):
    """Single-hidden MLP ``z = W2 ReLU(W1 h + b1) + b2``. Infer with argmax."""

    def __init__(self, in_dim: int, n_clusters: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if int(n_clusters) < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        self.in_dim = int(in_dim)
        self.n_clusters = int(n_clusters)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_clusters, bias=True)
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(h)))

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward(h).argmax(dim=-1)


class DualHiddenMoE(nn.Module):
    """Shared frozen L1 + linear dispatcher + ``B`` expert (L2, head) blocks."""

    architecture = "dual_hidden_moe_wdl"
    n_outputs = 3

    def __init__(
        self,
        base: DualHiddenNNUE,
        n_experts: int,
        dispatcher: LinearDispatcher | None = None,
    ) -> None:
        super().__init__()
        n_experts = int(n_experts)
        if n_experts < 2:
            raise ValueError(f"n_experts must be >= 2, got {n_experts}")
        self.feature_dim = base.feature_dim
        self.hidden_dim = base.hidden_dim
        self.hidden2_dim = base.hidden2_dim
        self.crelu_clip = base.crelu_clip
        self.n_experts = n_experts
        self.l1 = _clone_linear(base.l1)
        self.experts_l2 = nn.ModuleList(_clone_linear(base.l2) for _ in range(n_experts))
        self.experts_head = nn.ModuleList(_clone_linear(base.head) for _ in range(n_experts))
        in_dim = self.hidden_dim * 2
        self.dispatcher = dispatcher if dispatcher is not None else LinearDispatcher(in_dim, n_experts)
        if self.dispatcher.n_clusters != n_experts or self.dispatcher.in_dim != in_dim:
            raise ValueError("dispatcher dimensions do not match base / n_experts")

    @classmethod
    def from_base(
        cls,
        base: DualHiddenNNUE,
        n_experts: int,
        dispatcher: LinearDispatcher | None = None,
    ) -> DualHiddenMoE:
        return cls(base, n_experts, dispatcher=dispatcher)

    def freeze_l1(self) -> None:
        for param in self.l1.parameters():
            param.requires_grad = False

    def load_expert_from_base(self, expert_id: int, base: DualHiddenNNUE) -> None:
        self.experts_l2[int(expert_id)].load_state_dict(base.l2.state_dict())
        self.experts_head[int(expert_id)].load_state_dict(base.head.state_dict())

    def l1_dense(self, features: torch.Tensor) -> torch.Tensor:
        return crelu(self.l1(features), self.crelu_clip)

    def l1_sparse(self, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe = indices.long().clamp(min=0, max=self.feature_dim - 1)
        features = torch.zeros(
            safe.shape[0],
            self.feature_dim,
            device=safe.device,
            dtype=self.l1.weight.dtype,
        )
        features.scatter_add_(1, safe, mask.to(dtype=features.dtype))
        return crelu(self.l1(features), self.crelu_clip)

    def stm_concat(
        self,
        white_h: torch.Tensor,
        black_h: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        stm_mask = stm_white.unsqueeze(1)
        stm_h = torch.where(stm_mask, white_h, black_h)
        opp_h = torch.where(stm_mask, black_h, white_h)
        return torch.cat([stm_h, opp_h], dim=1)

    def l1_concat(
        self,
        white_features: torch.Tensor,
        black_features: torch.Tensor,
        stm_white: torch.Tensor,
        white_mask: torch.Tensor | None = None,
        black_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (white_mask is None) != (black_mask is None):
            raise ValueError("white_mask and black_mask must both be set or both omitted")
        if white_mask is None:
            white_h = self.l1_dense(white_features)
            black_h = self.l1_dense(black_features)
        else:
            white_h = self.l1_sparse(white_features, white_mask)
            black_h = self.l1_sparse(black_features, black_mask)
        return self.stm_concat(white_h, black_h, stm_white)

    def expert_logits_from_h(self, h: torch.Tensor, expert_id: int) -> torch.Tensor:
        h2 = crelu(self.experts_l2[int(expert_id)](h), self.crelu_clip)
        return self.experts_head[int(expert_id)](h2)

    def all_expert_logits_from_h(self, h: torch.Tensor) -> torch.Tensor:
        """``(B, E, 3)`` logits, one row per expert."""
        parts = [self.expert_logits_from_h(h, i) for i in range(self.n_experts)]
        return torch.stack(parts, dim=1)

    def route_from_h(
        self,
        h: torch.Tensor,
        *,
        expert_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits (B, 3), bucket ids (B,))``."""
        if expert_ids is None:
            expert_ids = self.dispatcher.predict(h)
        else:
            expert_ids = expert_ids.long()
        stacked = self.all_expert_logits_from_h(h)
        gather = expert_ids.view(-1, 1, 1).expand(-1, 1, 3)
        logits = stacked.gather(1, gather).squeeze(1)
        return logits, expert_ids

    def forward(
        self,
        white_features: torch.Tensor,
        black_features: torch.Tensor,
        stm_white: torch.Tensor,
        white_mask: torch.Tensor | None = None,
        black_mask: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.l1_concat(
            white_features, black_features, stm_white, white_mask, black_mask
        )
        logits, _ids = self.route_from_h(h, expert_ids=expert_ids)
        return logits

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def stm_value(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self.probabilities(logits)
        return probs[..., 0] - probs[..., 2]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def state_payload(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "hidden2_dim": self.hidden2_dim,
            "n_experts": self.n_experts,
            "model_state_dict": self.state_dict(),
            "head_parameter_dim": head_parameter_dim(self.hidden_dim, self.hidden2_dim),
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_payload(), path)

    @classmethod
    def load(cls, path: Path | str, *, device: torch.device | str = "cpu") -> DualHiddenMoE:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        hidden_dim = int(payload["hidden_dim"])
        hidden2_dim = int(payload["hidden2_dim"])
        n_experts = int(payload["n_experts"])
        base = DualHiddenNNUE(hidden_dim=hidden_dim, hidden2_dim=hidden2_dim)
        moe = cls(base, n_experts)
        moe.load_state_dict(payload["model_state_dict"])
        return moe.to(device)
