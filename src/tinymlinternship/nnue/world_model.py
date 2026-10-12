"""Self-supervised world-model encoder for the L1 accumulator.

Replaces the CReLU accumulator activation with a contrastively trained latent
``z = silu(W_WM x + b)`` (optionally unit-hypersphere normalized). The positive
pair is the board-mirror symmetry: for one position the white-POV and black-POV
views are ``(x, x+)``. Everything else (the ``l1`` weight matrix, the ``2W``
concat, ``l2``, and the head) is unchanged, so state-dict keys stay identical
and legacy checkpoints keep loading.

Losses:
- InfoNCE (SimCLR-style symmetric contrastive loss) with temperature ``tau``.
- VICReg (invariance on the mirror pair + variance + covariance decorrelation).

Both are exposed by ``train_world_model_encoder``, which trains only ``base.l1``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinymlinternship.nnue.model import DualHiddenNNUE

WM_BATCH = 2048


def _pov_z(
    l1: nn.Linear,
    idx: torch.Tensor,
    mask: torch.Tensor,
    feature_dim: int,
    normalize: bool,
) -> torch.Tensor:
    """World-model latent for one POV from sparse indices (silu + optional norm)."""
    safe = idx.long().clamp(min=0, max=feature_dim - 1)
    features = torch.zeros(
        safe.shape[0], feature_dim, device=safe.device, dtype=l1.weight.dtype
    )
    features.scatter_add_(1, safe, mask.to(dtype=features.dtype))
    z = F.silu(l1(features))
    if normalize:
        z = F.normalize(z, dim=-1, eps=1e-8)
    return z


def encode_mirror_pair(
    base: DualHiddenNNUE,
    batch: dict[str, torch.Tensor],
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(z_white, z_black)`` — the mirror-symmetry positive pair."""
    z_white = _pov_z(base.l1, batch["white_idx"], batch["white_mask"], base.feature_dim, normalize)
    z_black = _pov_z(base.l1, batch["black_idx"], batch["black_mask"], base.feature_dim, normalize)
    return z_white, z_black


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric InfoNCE over the batch (in-batch negatives)."""
    z1 = F.normalize(z1, dim=-1, eps=1e-8)
    z2 = F.normalize(z2, dim=-1, eps=1e-8)
    tau = max(float(tau), 1e-6)
    logits = (z1 @ z2.transpose(0, 1)) / tau
    labels = torch.arange(int(z1.shape[0]), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def _var_loss(z: torch.Tensor, gamma: float) -> torch.Tensor:
    std = z - z.mean(dim=0)
    std = torch.sqrt(std.pow(2).mean(dim=0) + 1e-4)
    return F.relu(gamma - std).mean()


def _cov_loss(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    if n < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    zc = z - z.mean(dim=0)
    cov = (zc.transpose(0, 1) @ zc) / (n - 1)
    off = cov.flatten()[1:].view(d - 1, d + 1)[:, :-1].reshape(-1)
    return off.pow(2).sum() / d


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """VICReg: mirror-pair invariance + variance + covariance decorrelation."""
    inv = F.mse_loss(z1, z2)
    var = _var_loss(z1, gamma) + _var_loss(z2, gamma)
    cov = _cov_loss(z1) + _cov_loss(z2)
    return sim_coeff * inv + var_coeff * var + cov_coeff * cov


def collapse_diagnostics(z: torch.Tensor, *, k: int = 8) -> dict:
    """Singular-value spectrum / effective rank — detects representation collapse."""
    zf = z.float()
    zf = zf - zf.mean(dim=0)
    if zf.shape[0] < 2:
        return {"effective_rank": 0.0, "singular_values": []}
    s = torch.linalg.svdvals(zf)
    total = s.sum() + 1e-8
    p = s / total
    ent = -(p * (p + 1e-12).log()).sum()
    return {
        "effective_rank": float(torch.exp(ent).item()),
        "singular_values": [round(float(x), 4) for x in s[:k].tolist()],
    }


def train_world_model_encoder(
    base: DualHiddenNNUE,
    pack,
    cfg,
    device: torch.device,
    emit,
    cancel,
) -> dict:
    """Train only ``base.l1`` with InfoNCE / VICReg on mirror pairs.

    Sets ``base.encoder = "world_model"`` and ``base.normalize_l1`` so every
    downstream stage (base eval, clustering, dispatcher, experts) uses the
    world-model activation. Returns collapse diagnostics on a held sample.
    """
    base.encoder = "world_model"
    base.normalize_l1 = bool(cfg.wm_normalize)

    params = list(base.l1.parameters())
    opt = torch.optim.Adam(params, lr=cfg.wm_lr)
    rng = torch.Generator(device="cpu").manual_seed(0)
    n = len(pack)

    emit(
        {
            "stage": "world_model",
            "event": "start",
            "loss": cfg.wm_loss,
            "epochs": cfg.wm_epochs,
            "normalize": base.normalize_l1,
            "lr": cfg.wm_lr,
        }
    )

    def _batch(device_idx: torch.Tensor) -> dict:
        from tinymlinternship.nnue.moe_data import batches_to_device

        return batches_to_device(pack.gather(device_idx), device)

    base.train()
    for epoch in range(1, int(cfg.wm_epochs) + 1):
        cancel.check()
        perm = torch.randperm(n, generator=rng, device="cpu")
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, WM_BATCH):
            cancel.check()
            idx = perm[start : start + WM_BATCH]
            batch = _batch(idx)
            z_w, z_b = encode_mirror_pair(base, batch, base.normalize_l1)
            if cfg.wm_loss == "vicreg":
                loss = vicreg_loss(z_w, z_b, gamma=float(getattr(cfg, "wm_vicreg_gamma", 1.0)))
            else:
                loss = info_nce_loss(z_w, z_b, cfg.wm_tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        emit(
            {
                "stage": "world_model",
                "event": "epoch",
                "epoch": epoch,
                "epochs": int(cfg.wm_epochs),
                "loss": total_loss / max(n_batches, 1),
                "progress": epoch / max(int(cfg.wm_epochs), 1),
            }
        )

    base.eval()
    diag: dict = {}
    with torch.inference_mode():
        sample_idx = torch.arange(min(n, WM_BATCH), dtype=torch.long)
        batch = _batch(sample_idx)
        z_w, _z_b = encode_mirror_pair(base, batch, base.normalize_l1)
        diag = collapse_diagnostics(z_w)
    emit({"stage": "world_model", "event": "done", "diagnostics": diag})
    return diag
