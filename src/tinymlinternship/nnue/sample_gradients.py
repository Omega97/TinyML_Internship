"""Per-sample head gradients, L2-normalized and randomly projected.

Full head grads are 66,563-d (W=128, H=256). Storing them is ~127 MiB/1M even
in float16 after a 64-d projection, and ~127 GiB at native width. This module
never materializes the 66k-d vector: the Gaussian projection is fused into the
outer products, and the original L2 norm is recovered from Frobenius identities
so direction is preserved.

Default ``reduce_dim=48`` float16 is **91.6 MiB per million positions** (<100 MiB).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinymlinternship.nnue.model import DualHiddenNNUE, crelu

EPS = 1e-8
BYTES_PER_MILLION_LIMIT = 100 * 1024 * 1024


def head_parameter_dim(hidden_dim: int, hidden2_dim: int) -> int:
    """``P_head = 2W·H + H + H·3 + 3``."""
    two_w = int(hidden_dim) * 2
    h2 = int(hidden2_dim)
    return two_w * h2 + h2 + h2 * 3 + 3


def bytes_per_million(reduce_dim: int, dtype: np.dtype = np.float16) -> int:
    item = int(np.dtype(dtype).itemsize)
    return int(1_000_000 * int(reduce_dim) * item)


def max_reduce_dim_for_budget(
    *,
    dtype: np.dtype = np.float16,
    budget: int = BYTES_PER_MILLION_LIMIT,
) -> int:
    item = int(np.dtype(dtype).itemsize)
    return max(1, int(budget) // (1_000_000 * item))


def make_projection_matrix(
    head_dim: int,
    reduce_dim: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``R ~ N(0, 1/sqrt(D))`` of shape ``(D, P)`` so ``E[||R Δ||²] ≈ ||Δ||²``."""
    if int(reduce_dim) < 1:
        raise ValueError(f"reduce_dim must be >= 1, got {reduce_dim}")
    if int(head_dim) < 1:
        raise ValueError(f"head_dim must be >= 1, got {head_dim}")
    device = torch.device(device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    cpu = torch.randn(int(reduce_dim), int(head_dim), generator=gen, dtype=torch.float32)
    cpu.mul_(1.0 / (float(reduce_dim) ** 0.5))
    return cpu.to(device=device, dtype=dtype)


def _crelu_mask(pre: torch.Tensor, clip: float) -> torch.Tensor:
    return (pre > 0) & (pre < float(clip))


def head_grad_norm_sq(
    g_pre: torch.Tensor,
    h: torch.Tensor,
    g_logits: torch.Tensor,
    h2: torch.Tensor,
) -> torch.Tensor:
    """``||Δ||²`` from outer-product identities; no ``(B, P)`` materialization."""
    g_pre_sq = (g_pre * g_pre).sum(dim=-1)
    h_sq = (h * h).sum(dim=-1)
    g_log_sq = (g_logits * g_logits).sum(dim=-1)
    h2_sq = (h2 * h2).sum(dim=-1)
    return g_pre_sq * h_sq + g_pre_sq + g_log_sq * h2_sq + g_log_sq


def analytic_head_grad_flat(
    g_pre: torch.Tensor,
    h: torch.Tensor,
    g_logits: torch.Tensor,
    h2: torch.Tensor,
) -> torch.Tensor:
    """Materialize ``(B, P)`` head grads. Tests / tiny nets only."""
    w_l2 = g_pre.unsqueeze(2) * h.unsqueeze(1)
    w_out = g_logits.unsqueeze(2) * h2.unsqueeze(1)
    return torch.cat(
        [
            w_l2.reshape(g_pre.shape[0], -1),
            g_pre,
            w_out.reshape(g_logits.shape[0], -1),
            g_logits,
        ],
        dim=1,
    )


def project_head_grads(
    g_pre: torch.Tensor,
    h: torch.Tensor,
    g_logits: torch.Tensor,
    h2: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """``R @ vec(Δ)`` with ``Δ`` the concatenated head gradient of each row.

    ``projection`` is ``(D, P)`` with P = H·2W + H + 3·H + 3, layout matching
    ``torch.nn.Linear`` row-major ``weight.reshape(-1)`` then ``bias``.
    """
    batch, hidden2 = g_pre.shape
    two_w = h.shape[1]
    reduce_dim, head_dim = projection.shape
    expected = hidden2 * two_w + hidden2 + 3 * hidden2 + 3
    if int(head_dim) != expected:
        raise ValueError(f"projection width {head_dim} != head dim {expected}")
    if g_logits.shape != (batch, 3) or h2.shape != (batch, hidden2):
        raise ValueError("g_logits / h2 batch shapes do not match g_pre")

    n_wl2 = hidden2 * two_w
    n_bl2 = hidden2
    n_wout = 3 * hidden2
    r_wl2 = projection[:, :n_wl2].reshape(reduce_dim, hidden2, two_w)
    r_bl2 = projection[:, n_wl2 : n_wl2 + n_bl2]
    off = n_wl2 + n_bl2
    r_wout = projection[:, off : off + n_wout].reshape(reduce_dim, 3, hidden2)
    r_bout = projection[:, off + n_wout :]

    proj = torch.einsum("dhw,bh,bw->bd", r_wl2, g_pre, h)
    proj = proj + g_pre @ r_bl2.transpose(0, 1)
    proj = proj + torch.einsum("doh,bo,bh->bd", r_wout, g_logits, h2)
    proj = proj + g_logits @ r_bout.transpose(0, 1)
    return proj


def normalize_projected(
    projected: torch.Tensor,
    norm_sq: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """``(R Δ) / (||Δ|| + eps)`` — unit direction in the original space."""
    scale = norm_sq.clamp_min(0.0).sqrt() + float(eps)
    return projected / scale.unsqueeze(1)


def sample_head_gradients(
    model: DualHiddenNNUE,
    batch: dict[str, torch.Tensor],
    projection: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """L2-normalized, projected per-sample head grads. No autograd graph."""
    model.eval()
    white_idx = batch["white_idx"]
    black_idx = batch["black_idx"]
    stm = batch["stm_white"]
    white_mask = batch.get("white_mask")
    black_mask = batch.get("black_mask")
    target = batch["target"].float()
    denom = target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target = target / denom

    with torch.no_grad():
        h = model.l1_concat(white_idx, black_idx, stm, white_mask, black_mask)
        pre_l2 = model.l2(h)
        h2 = crelu(pre_l2, model.crelu_clip)
        logits = model.head(h2)
        probs = torch.softmax(logits.float(), dim=-1)
        g_logits = (probs - target).to(dtype=h.dtype)
        g_h2 = g_logits @ model.head.weight.to(dtype=g_logits.dtype)
        g_pre = g_h2 * _crelu_mask(pre_l2, model.crelu_clip).to(dtype=g_h2.dtype)
        h_f = h.to(dtype=g_pre.dtype)
        h2_f = h2.to(dtype=g_pre.dtype)
        proj = project_head_grads(g_pre, h_f, g_logits, h2_f, projection.to(dtype=g_pre.dtype))
        norm_sq = head_grad_norm_sq(g_pre, h_f, g_logits, h2_f)
        return normalize_projected(proj, norm_sq, eps=eps)


def flatten_head_grad_autograd(
    model: DualHiddenNNUE,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Reference per-sample head grads via autograd (tests / tiny batches)."""
    n = int(batch["target"].shape[0])
    rows: list[torch.Tensor] = []
    params = (model.l2.weight, model.l2.bias, model.head.weight, model.head.bias)
    for i in range(n):
        model.zero_grad(set_to_none=True)
        sl = {key: tensor[i : i + 1] for key, tensor in batch.items()}
        logits = model(
            sl["white_idx"],
            sl["black_idx"],
            sl["stm_white"],
            sl.get("white_mask"),
            sl.get("black_mask"),
        )
        target = sl["target"].float()
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        log_p = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        loss = -(target * log_p).sum()
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows, dim=0)


def open_float16_store(
    path: Path,
    n_rows: int,
    n_dim: int,
    *,
    resume: bool = False,
) -> np.memmap:
    """Pre-allocate ``(n_rows, n_dim)`` float16 mmap. Resume keeps existing bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (int(n_rows), int(n_dim))
    if resume and path.is_file():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape or arr.dtype != np.float16:
            raise ValueError(f"{path} shape/dtype {arr.shape} {arr.dtype} != {shape} float16")
        return arr
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


def write_grad_meta(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_grad_meta(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
