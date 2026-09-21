"""NNUE sample-gradient unpacking, σ=1, and layer coordinates."""

from __future__ import annotations

import numpy as np
import torch

from tinymlinternship.features import FEATURE_DIM
from tinymlinternship.nnue.grad_graph import (
    N_NET_LAYERS,
    edge_rgba,
    layer_edges,
    neuron_xy,
    sample_weight_grads,
    split_head_grad_vector,
    standardize_weight_grads,
    toy_weight_grads,
    weight_grads_from_head_vector,
)
from tinymlinternship.nnue.model import DualHiddenNNUE
from tinymlinternship.nnue.sample_gradients import analytic_head_grad_flat, head_parameter_dim


def _rand_batch(n: int = 3, width: int = 8, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "white_idx": torch.randint(0, FEATURE_DIM, (n, width), generator=g),
        "white_mask": torch.ones(n, width, dtype=torch.bool),
        "black_idx": torch.randint(0, FEATURE_DIM, (n, width), generator=g),
        "black_mask": torch.ones(n, width, dtype=torch.bool),
        "stm_white": torch.randint(0, 2, (n,), generator=g).bool(),
        "target": torch.softmax(torch.randn(n, 3, generator=g), dim=-1),
        "weight": torch.ones(n),
    }


def test_neuron_xy_matches_layer_formula():
    assert N_NET_LAYERS == 4
    assert neuron_xy(0, 0, 844) == (0.0, 0.0)
    assert neuron_xy(1, 0, 256) == (0.0, 1.0 / 3.0)
    x, y = neuron_xy(2, 127, 256)
    assert x == 127 / 256
    assert y == 2.0 / 3.0
    assert neuron_xy(3, 2, 3) == (2.0 / 3.0, 1.0)


def test_edge_rgba_red_blue_and_fade():
    r, g, b, a = edge_rgba(-2.0)
    assert r > b and a >= 250
    r, g, b, a = edge_rgba(2.0)
    assert b > r and a >= 250
    _r, _g, _b, a0 = edge_rgba(0.0)
    assert a0 == 0
    _r, _g, _b, a_small = edge_rgba(0.2)
    _r, _g, _b, a_big = edge_rgba(1.5)
    assert a_small < a_big


def test_split_head_grad_inverts_analytic_flat():
    torch.manual_seed(0)
    model = DualHiddenNNUE(hidden_dim=4, hidden2_dim=6)
    model.eval()
    batch = _rand_batch(2, seed=1)
    with torch.no_grad():
        h = model.l1_concat(
            batch["white_idx"],
            batch["black_idx"],
            batch["stm_white"],
            batch["white_mask"],
            batch["black_mask"],
        )
        pre = model.l2(h)
        from tinymlinternship.nnue.model import crelu
        from tinymlinternship.nnue.sample_gradients import _crelu_mask

        h2 = crelu(pre, model.crelu_clip)
        logits = model.head(h2)
        target = batch["target"].float()
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        g_logits = (torch.softmax(logits.float(), dim=-1) - target).to(dtype=h.dtype)
        g_h2 = g_logits @ model.head.weight.to(dtype=g_logits.dtype)
        g_pre = g_h2 * _crelu_mask(pre, model.crelu_clip).to(dtype=g_h2.dtype)
        flat = analytic_head_grad_flat(g_pre, h, g_logits, h2)
    w_l2, b_l2, w_out, b_out = split_head_grad_vector(
        flat[0].numpy(), model.hidden_dim, model.hidden2_dim
    )
    assert w_l2.shape == (6, 8)
    assert b_l2.shape == (6,)
    assert w_out.shape == (3, 6)
    assert b_out.shape == (3,)
    np.testing.assert_allclose(w_l2, (g_pre[0].unsqueeze(1) * h[0].unsqueeze(0)).numpy(), atol=1e-5)
    np.testing.assert_allclose(w_out, (g_logits[0].unsqueeze(1) * h2[0].unsqueeze(0)).numpy(), atol=1e-5)
    np.testing.assert_allclose(b_l2, g_pre[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(b_out, g_logits[0].numpy(), atol=1e-5)
    rebuilt = weight_grads_from_head_vector(flat[0].numpy(), 4, 6)
    np.testing.assert_allclose(rebuilt.w12, w_l2)
    np.testing.assert_allclose(rebuilt.w23, w_out)
    assert rebuilt.w01.shape == (8, FEATURE_DIM)
    assert np.allclose(rebuilt.w01, 0.0)
    assert rebuilt.sizes == (FEATURE_DIM, 8, 6, 3)


def test_sample_weight_grads_match_head_and_l1_sum():
    torch.manual_seed(2)
    model = DualHiddenNNUE(hidden_dim=4, hidden2_dim=6)
    model.eval()
    batch = _rand_batch(1, width=10, seed=3)
    grads = sample_weight_grads(model, batch)
    assert grads.sizes == (FEATURE_DIM, 8, 6, 3)
    with torch.no_grad():
        h = model.l1_concat(
            batch["white_idx"],
            batch["black_idx"],
            batch["stm_white"],
            batch["white_mask"],
            batch["black_mask"],
        )
        pre = model.l2(h)
        from tinymlinternship.nnue.model import crelu
        from tinymlinternship.nnue.sample_gradients import _crelu_mask

        h2 = crelu(pre, model.crelu_clip)
        logits = model.head(h2)
        target = batch["target"].float()
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        g_logits = (torch.softmax(logits.float(), dim=-1) - target).to(dtype=h.dtype)
        g_h2 = g_logits @ model.head.weight.to(dtype=g_logits.dtype)
        g_pre = g_h2 * _crelu_mask(pre, model.crelu_clip).to(dtype=g_h2.dtype)
        flat = analytic_head_grad_flat(g_pre, h, g_logits, h2)
    w_l2, _b, w_out, _bo = split_head_grad_vector(flat[0].numpy(), 4, 6)
    np.testing.assert_allclose(grads.w12, w_l2, atol=1e-4)
    np.testing.assert_allclose(grads.w23, w_out, atol=1e-4)

    sl = {key: tensor[:1] for key, tensor in batch.items()}
    model.zero_grad(set_to_none=True)
    logits = model(
        sl["white_idx"],
        sl["black_idx"],
        sl["stm_white"],
        sl["white_mask"],
        sl["black_mask"],
    )
    target = sl["target"].float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    loss = -(target * torch.nn.functional.log_softmax(logits.float(), dim=-1)).sum()
    (g_l1,) = torch.autograd.grad(loss, model.l1.weight, retain_graph=False)
    stm_block = grads.w01[:4]
    opp_block = grads.w01[4:]
    np.testing.assert_allclose(
        (stm_block + opp_block),
        g_l1.detach().cpu().numpy(),
        atol=1e-4,
        rtol=1e-4,
    )


def test_standardize_has_unit_sigma():
    g = toy_weight_grads(7, hidden_dim=8, hidden2_dim=10, feature_dim=16)
    z = standardize_weight_grads(g)
    flat = np.concatenate([z.w01.ravel(), z.w12.ravel(), z.w23.ravel()])
    np.testing.assert_allclose(np.std(flat), 1.0, atol=1e-5)


def test_layer_edges_skip_near_zero_and_keep_sign():
    w = np.array([[0.0, 3.0], [-2.0, 0.01]], dtype=np.float32)
    i_s, js, zs = layer_edges(w, min_abs=0.05)
    pairs = sorted(zip(i_s.tolist(), js.tolist(), zs.tolist()))
    assert pairs[0][0:2] == (0, 1) and pairs[0][2] < 0
    assert pairs[1][0:2] == (1, 0) and pairs[1][2] > 0
    assert len(pairs) == 2


def test_head_parameter_dim_still_matches_layout():
    assert head_parameter_dim(4, 6) == 4 * 2 * 6 + 6 + 3 * 6 + 3
