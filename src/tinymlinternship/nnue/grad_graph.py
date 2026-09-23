"""Map a per-sample NNUE gradient and activations onto the four DualHidden layers.

Stored MoE vectors are a random projection of the *head* gradient
(``L2`` + output), so this module either unpacks a full head vector or
recomputes analytic ``dL/dW`` from the position. Layout of a flattened
head vector matches ``analytic_head_grad_flat`` / ``nn.Linear`` row-major
``weight.reshape(-1)`` then ``bias``: ``W_l2, b_l2, W_out, b_out``.

Layer sizes are ``(in=844, hidden1=2W, hidden2=H, out=3)``. Shared L1 is
drawn twice: STM occupies hidden1 ``[0, W)``, opponent ``[W, 2W)``.
A connection from layer ``l`` neuron ``i`` to layer ``l+1`` neuron ``j``
sits at ``(i / h_l, l / 3) → (j / h_{l+1}, (l+1) / 3)``.

Activations sit on those neurons: STM input features, CReLU concat ``h``,
CReLU ``h2``, and the three WDL logits. Colored with a matplotlib colormap
(default ``managua``); pass ``cmap=`` to ``activation_rgba`` / the painter
to swap it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinymlinternship.features import FEATURE_DIM
from tinymlinternship.nnue.sample_gradients import _crelu_mask, head_parameter_dim

N_NET_LAYERS = 4
GRAD_MIN_ABS = 0.05
GRAD_MAX_EDGES_PER_LAYER = 12_000
GRAD_SIGMA_SCALE = 10.0
POS_RGB = (40, 110, 255)
NEG_RGB = (230, 45, 45)
ACT_CMAP_NAME = "managua"
ACT_SIGMA_SCALE = 2.0
_CMAP_LUTS: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class NnueWeightGrads:
    """Unnormalized ``dL/dW`` for the three DualHidden maps."""

    w01: np.ndarray  # (2W, in) STM then opp
    w12: np.ndarray  # (H, 2W)
    w23: np.ndarray  # (3, H)
    hidden_dim: int
    hidden2_dim: int
    feature_dim: int = FEATURE_DIM

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        return (int(self.feature_dim), int(self.hidden_dim) * 2, int(self.hidden2_dim), 3)

    @property
    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.w01, self.w12, self.w23)


@dataclass(frozen=True)
class NnueActivations:
    """Per-neuron activations aligned with ``NnueWeightGrads.sizes``.

    ``x0`` is the STM 844-d feature vector (the graph has one input rail).
    ``h1`` is STM-ordered CReLU concat, ``h2`` is CReLU(L2), ``out`` is logits.
    ``wdl`` is softmax (win, draw, loss) for the side to move. ``stm_white``
    is that side: bar colors are white-grey-black when it is True.
    """

    x0: np.ndarray  # (in,)
    h1: np.ndarray  # (2W,)
    h2: np.ndarray  # (H,)
    out: np.ndarray  # (3,) logits
    hidden_dim: int
    hidden2_dim: int
    feature_dim: int = FEATURE_DIM
    wdl: np.ndarray | None = None  # (3,) probabilities; None → softmax(out)
    stm_white: bool = True

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        return (int(self.feature_dim), int(self.hidden_dim) * 2, int(self.hidden2_dim), 3)

    @property
    def layers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.x0, self.h1, self.h2, self.out)

    def probabilities(self) -> np.ndarray:
        """STM (win, draw, loss), length 3, summing to 1."""
        if self.wdl is not None:
            raw = np.asarray(self.wdl, dtype=np.float64).reshape(-1)
        else:
            raw = _softmax1d(self.out)
        out = np.zeros(3, dtype=np.float64)
        n = min(3, int(raw.size))
        if n:
            out[:n] = np.clip(raw[:n], 0.0, None)
        total = float(out.sum())
        if total <= 1e-12:
            return np.full(3, 1.0 / 3.0, dtype=np.float64)
        return out / total


def _softmax1d(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if z.size == 0:
        return np.zeros(0, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    total = float(np.sum(e))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(z.shape, 1.0 / float(z.size), dtype=np.float64)
    return e / total


def neuron_xy(layer: int, index: int, n: int) -> tuple[float, float]:
    """Unit-square position of neuron ``index`` on layer ``layer`` in ``0..3``."""
    n = max(int(n), 1)
    x = (float(index) + 0.5) / (float(n))
    y = 1 - float(layer) / 3.0
    return x, y


def edge_rgba(z: float, *, sigma_scale: float = GRAD_SIGMA_SCALE) -> tuple[int, int, int, int]:
    """Blue if ``z>0``, red if ``z<0``, alpha → 0 as ``z→0`` (σ=1 units)."""
    mag = abs(float(z))
    alpha = 0 if mag <= 0.0 else min(1.0, mag / max(float(sigma_scale), 1e-8))
    rgb = POS_RGB if z >= 0.0 else NEG_RGB
    return rgb[0], rgb[1], rgb[2], int(round(255.0 * alpha))


def activation_lut(cmap: str = ACT_CMAP_NAME) -> np.ndarray:
    """256×3 uint8 samples of a matplotlib colormap (``cmap`` is the name)."""
    name = str(cmap or ACT_CMAP_NAME)
    lut = _CMAP_LUTS.get(name)
    if lut is None:
        import matplotlib.pyplot as plt

        mpl_cmap = plt.get_cmap(name)
        t = np.linspace(0.0, 1.0, 256)
        rgb = np.asarray(mpl_cmap(t), dtype=np.float64)[:, :3]
        lut = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        _CMAP_LUTS[name] = lut
    return lut


def activation_rgba(
    z: float,
    *,
    cmap: str = ACT_CMAP_NAME,
    vmax: float = ACT_SIGMA_SCALE,
) -> tuple[int, int, int, int]:
    """Map σ=1 activation ``z`` through matplotlib ``cmap``. Opaque.

    ``t = 0.5 + 0.5 clip(z/vmax)`` so 0 sits at the colormap center.
    Swap palettes with ``cmap='managua'``, ``cmap='RdYlBu'``, …
    """
    scale = max(float(vmax), 1e-8)
    t = 0.5 + 0.5 * float(np.clip(float(z) / scale, -1.0, 1.0))
    lut = activation_lut(cmap)
    idx = int(round(t * (len(lut) - 1)))
    idx = 0 if idx < 0 else (len(lut) - 1 if idx >= len(lut) else idx)
    r, g, b = (int(lut[idx, 0]), int(lut[idx, 1]), int(lut[idx, 2]))
    return r, g, b, 255


def split_head_grad_vector(
    vec: np.ndarray,
    hidden_dim: int,
    hidden2_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invert ``analytic_head_grad_flat``: ``W_l2, b_l2, W_out, b_out``."""
    two_w = int(hidden_dim) * 2
    h2 = int(hidden2_dim)
    expected = head_parameter_dim(two_w // 2, h2)
    flat = np.asarray(vec, dtype=np.float32).reshape(-1)
    if int(flat.size) != expected:
        raise ValueError(f"head grad length {flat.size} != {expected} for W={hidden_dim} H={h2}")
    n_wl2 = h2 * two_w
    n_bl2 = h2
    n_wout = 3 * h2
    w_l2 = flat[:n_wl2].reshape(h2, two_w)
    b_l2 = flat[n_wl2 : n_wl2 + n_bl2]
    off = n_wl2 + n_bl2
    w_out = flat[off : off + n_wout].reshape(3, h2)
    b_out = flat[off + n_wout :]
    return w_l2, b_l2, w_out, b_out


def weight_grads_from_head_vector(
    vec: np.ndarray,
    hidden_dim: int,
    hidden2_dim: int,
    feature_dim: int = FEATURE_DIM,
) -> NnueWeightGrads:
    """Head-only vector: L2 and output filled, L1 zeros (not in the head grad)."""
    w_l2, _b_l2, w_out, _b_out = split_head_grad_vector(vec, hidden_dim, hidden2_dim)
    w01 = np.zeros((int(hidden_dim) * 2, int(feature_dim)), dtype=np.float32)
    return NnueWeightGrads(
        w01=w01,
        w12=np.ascontiguousarray(w_l2, dtype=np.float32),
        w23=np.ascontiguousarray(w_out, dtype=np.float32),
        hidden_dim=int(hidden_dim),
        hidden2_dim=int(hidden2_dim),
        feature_dim=int(feature_dim),
    )


def standardize_weight_grads(grads: NnueWeightGrads, eps: float = 1e-8) -> NnueWeightGrads:
    """Scale all three maps jointly so the concatenated gradient has σ=1."""
    flat = np.concatenate([np.ravel(m) for m in grads.matrices]).astype(np.float64, copy=False)
    std = float(np.std(flat))
    scale = std if std > float(eps) else 1.0
    return NnueWeightGrads(
        w01=(grads.w01 / scale).astype(np.float32, copy=False),
        w12=(grads.w12 / scale).astype(np.float32, copy=False),
        w23=(grads.w23 / scale).astype(np.float32, copy=False),
        hidden_dim=grads.hidden_dim,
        hidden2_dim=grads.hidden2_dim,
        feature_dim=grads.feature_dim,
    )


def standardize_activations(acts: NnueActivations, eps: float = 1e-8) -> NnueActivations:
    """Scale each layer by its own σ so dots use the full colormap range."""
    scaled: list[np.ndarray] = []
    for layer in acts.layers:
        flat = np.asarray(layer, dtype=np.float64).reshape(-1)
        std = float(np.std(flat))
        scale = std if std > float(eps) else 1.0
        scaled.append((flat / scale).astype(np.float32, copy=False))
    return NnueActivations(
        x0=scaled[0],
        h1=scaled[1],
        h2=scaled[2],
        out=scaled[3],
        hidden_dim=acts.hidden_dim,
        hidden2_dim=acts.hidden2_dim,
        feature_dim=acts.feature_dim,
        wdl=acts.wdl,
        stm_white=acts.stm_white,
    )


def layer_edges(
    weight: np.ndarray,
    *,
    min_abs: float = GRAD_MIN_ABS,
    max_edges: int = GRAD_MAX_EDGES_PER_LAYER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(src_i, dst_j, z)`` for ``weight[j, i]``, weakest-first, |z| ≥ min_abs."""
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or w.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty, np.zeros(0, dtype=np.float32)
    abs_w = np.abs(w)
    js, i_s = np.nonzero(abs_w >= float(min_abs))
    zs = w[js, i_s]
    if zs.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty, np.zeros(0, dtype=np.float32)
    cap = int(max_edges)
    if cap > 0 and int(zs.size) > cap:
        keep = np.argpartition(np.abs(zs), -cap)[-cap:]
        i_s = i_s[keep]
        js = js[keep]
        zs = zs[keep]
    order = np.argsort(np.abs(zs), kind="stable")
    return (
        i_s[order].astype(np.int32, copy=False),
        js[order].astype(np.int32, copy=False),
        zs[order].astype(np.float32, copy=False),
    )


def toy_weight_grads(
    seed: int,
    *,
    hidden_dim: int = 128,
    hidden2_dim: int = 256,
    feature_dim: int = FEATURE_DIM,
) -> NnueWeightGrads:
    """Deterministic stand-in when no checkpoint is loaded (demo / tests)."""
    rng = np.random.RandomState(int(seed) + 91)
    w = int(hidden_dim)
    h2 = int(hidden2_dim)
    inn = int(feature_dim)
    w01 = rng.normal(0.0, 0.4, size=(2 * w, inn)).astype(np.float32)
    w01 *= rng.random(w01.shape) < 0.035
    w12 = rng.normal(0.0, 0.25, size=(h2, 2 * w)).astype(np.float32)
    w12 *= rng.random(w12.shape) < 0.12
    w23 = rng.normal(0.0, 0.5, size=(3, h2)).astype(np.float32)
    return NnueWeightGrads(
        w01=w01, w12=w12, w23=w23, hidden_dim=w, hidden2_dim=h2, feature_dim=inn
    )


def toy_activations(
    seed: int,
    *,
    hidden_dim: int = 128,
    hidden2_dim: int = 256,
    feature_dim: int = FEATURE_DIM,
) -> NnueActivations:
    """Deterministic stand-in activations matching ``toy_weight_grads`` sizes."""
    rng = np.random.RandomState(int(seed) + 17)
    w = int(hidden_dim)
    h2 = int(hidden2_dim)
    inn = int(feature_dim)
    x0 = (rng.random(inn) < 0.04).astype(np.float32)
    if inn > 0:
        x0[0] = 1.0
        if inn > 3:
            x0[inn // 2] = 1.0
    h1 = np.clip(rng.normal(8.0, 12.0, size=2 * w), 0.0, 127.0).astype(np.float32)
    h2v = np.clip(rng.normal(6.0, 10.0, size=h2), 0.0, 127.0).astype(np.float32)
    out = rng.normal(0.0, 1.2, size=3).astype(np.float32)
    return NnueActivations(
        x0=x0,
        h1=h1,
        h2=h2v,
        out=out,
        hidden_dim=w,
        hidden2_dim=h2,
        feature_dim=inn,
        wdl=_softmax1d(out).astype(np.float32),
    )


def _dense_from_sparse(idx, mask, n_feat: int, dtype):
    import torch

    safe = idx.long().clamp(min=0, max=int(n_feat) - 1)
    x = torch.zeros(safe.shape[0], int(n_feat), device=safe.device, dtype=dtype)
    x.scatter_add_(1, safe, mask.to(dtype=dtype))
    return x


def batch_from_fen(
    fen: str,
    *,
    eval_target: float | None = None,
    device: str = "cpu",
):
    """Sparse DualHidden batch for one FEN. Target is max-entropy WDL of ``eval_target``."""
    import chess
    import torch

    from tinymlinternship.data.wdl import value_to_wdl
    from tinymlinternship.features import encode_dual

    board = chess.Board(fen)
    white_idx, black_idx = encode_dual(board)
    width = max(len(white_idx), len(black_idx), 1)

    def _pack(idxs: list[int]):
        t = torch.zeros(1, width, dtype=torch.long)
        m = torch.zeros(1, width, dtype=torch.bool)
        if idxs:
            n = min(len(idxs), width)
            t[0, :n] = torch.tensor(idxs[:n], dtype=torch.long)
            m[0, :n] = True
        return t, m

    wi, wm = _pack(list(white_idx))
    bi, bm = _pack(list(black_idx))
    if eval_target is None or not np.isfinite(eval_target):
        wdl = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    else:
        wdl = value_to_wdl(float(eval_target))
    return {
        "white_idx": wi.to(device),
        "white_mask": wm.to(device),
        "black_idx": bi.to(device),
        "black_mask": bm.to(device),
        "stm_white": torch.tensor([board.turn == chess.WHITE], device=device),
        "target": torch.tensor([wdl], dtype=torch.float32, device=device),
    }


def sample_network_maps(model, batch: dict) -> tuple[NnueWeightGrads, NnueActivations]:
    """Analytic per-row ``dL/dW`` plus the four layer activations.

    Takes the first row of ``batch``. Matches ``analytic_head_grad_flat`` on
    L2/head; L1 is the two POV outer products stacked, not the shared sum.
    Input activations are the STM feature vector (one 844-d rail).
    """
    import torch

    from tinymlinternship.nnue.model import DualHiddenNNUE, crelu

    if not isinstance(model, DualHiddenNNUE):
        raise TypeError(f"expected DualHiddenNNUE, got {type(model)!r}")
    model.eval()
    feature_dim = int(model.feature_dim)
    w = int(model.hidden_dim)
    h2 = int(model.hidden2_dim)
    with torch.no_grad():
        white_idx = batch["white_idx"]
        black_idx = batch["black_idx"]
        stm = batch["stm_white"]
        white_mask = batch.get("white_mask")
        black_mask = batch.get("black_mask")
        target = batch["target"].float()
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        dtype = model.l1.weight.dtype
        device = model.l1.weight.device
        if white_mask is None:
            x_white = white_idx.to(device=device, dtype=dtype)
            x_black = black_idx.to(device=device, dtype=dtype)
        else:
            x_white = _dense_from_sparse(white_idx.to(device), white_mask.to(device), feature_dim, dtype)
            x_black = _dense_from_sparse(black_idx.to(device), black_mask.to(device), feature_dim, dtype)

        stm_bool = stm.to(device=device).unsqueeze(1).bool()
        x_stm = torch.where(stm_bool, x_white, x_black)[0]

        pre_white = model.l1(x_white)
        pre_black = model.l1(x_black)
        h_white = crelu(pre_white, model.crelu_clip)
        h_black = crelu(pre_black, model.crelu_clip)
        h = model.stm_concat(h_white, h_black, stm.to(device))
        pre_l2 = model.l2(h)
        hidden2 = crelu(pre_l2, model.crelu_clip)
        logits = model.head(hidden2)
        probs = torch.softmax(logits.float(), dim=-1)
        stm_white = bool(stm.detach().flatten()[0].item())
        wdl = np.ascontiguousarray(probs[0].detach().cpu().numpy(), dtype=np.float32)
        g_logits = (probs - target.to(device=device)).to(dtype=h.dtype)
        g_h2 = g_logits @ model.head.weight.to(dtype=g_logits.dtype)
        g_pre = g_h2 * _crelu_mask(pre_l2, model.crelu_clip).to(dtype=g_h2.dtype)

        w12 = (g_pre.unsqueeze(2) * h.unsqueeze(1))[0]
        w23 = (g_logits.unsqueeze(2) * hidden2.unsqueeze(1))[0]

        g_h = g_pre @ model.l2.weight.to(dtype=g_pre.dtype)
        g_h_stm = g_h[:, :w]
        g_h_opp = g_h[:, w:]
        g_h_white = torch.where(stm_bool, g_h_stm, g_h_opp)
        g_h_black = torch.where(stm_bool, g_h_opp, g_h_stm)
        g_pre_white = g_h_white * _crelu_mask(pre_white, model.crelu_clip).to(dtype=g_h_white.dtype)
        g_pre_black = g_h_black * _crelu_mask(pre_black, model.crelu_clip).to(dtype=g_h_black.dtype)
        w_l1_white = g_pre_white.unsqueeze(2) * x_white.unsqueeze(1)
        w_l1_black = g_pre_black.unsqueeze(2) * x_black.unsqueeze(1)
        stm_m3 = stm_bool.unsqueeze(2)
        w_l1_stm = torch.where(stm_m3, w_l1_white, w_l1_black)
        w_l1_opp = torch.where(stm_m3, w_l1_black, w_l1_white)
        w01 = torch.cat([w_l1_stm, w_l1_opp], dim=1)[0]

    grads = NnueWeightGrads(
        w01=np.ascontiguousarray(w01.detach().cpu().numpy(), dtype=np.float32),
        w12=np.ascontiguousarray(w12.detach().cpu().numpy(), dtype=np.float32),
        w23=np.ascontiguousarray(w23.detach().cpu().numpy(), dtype=np.float32),
        hidden_dim=w,
        hidden2_dim=h2,
        feature_dim=feature_dim,
    )
    acts = NnueActivations(
        x0=np.ascontiguousarray(x_stm.detach().cpu().numpy(), dtype=np.float32),
        h1=np.ascontiguousarray(h[0].detach().cpu().numpy(), dtype=np.float32),
        h2=np.ascontiguousarray(hidden2[0].detach().cpu().numpy(), dtype=np.float32),
        out=np.ascontiguousarray(logits[0].detach().cpu().numpy(), dtype=np.float32),
        hidden_dim=w,
        hidden2_dim=h2,
        feature_dim=feature_dim,
        wdl=wdl,
        stm_white=stm_white,
    )
    return grads, acts


def sample_weight_grads(model, batch: dict) -> NnueWeightGrads:
    """Analytic per-row ``dL/dW`` for L1 (STM‖opp), L2, and the WDL head."""
    grads, _acts = sample_network_maps(model, batch)
    return grads
