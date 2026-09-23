"""NNUE sample-gradient unpacking, σ=1, and layer coordinates."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tinymlinternship.features import FEATURE_DIM
from tinymlinternship.nnue.grad_graph import (
    ACT_CMAP_NAME,
    N_NET_LAYERS,
    activation_rgba,
    edge_rgba,
    layer_edges,
    neuron_xy,
    sample_network_maps,
    sample_weight_grads,
    split_head_grad_vector,
    standardize_activations,
    standardize_weight_grads,
    toy_activations,
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
    x0, y0 = neuron_xy(0, 0, 844)
    assert x0 == pytest.approx(0.5 / 844.0)
    assert y0 == pytest.approx(1.0)
    x1, y1 = neuron_xy(1, 0, 256)
    assert x1 == pytest.approx(0.5 / 256.0)
    assert y1 == pytest.approx(2.0 / 3.0)
    x, y = neuron_xy(2, 127, 256)
    assert x == pytest.approx(127.5 / 256.0)
    assert y == pytest.approx(1.0 / 3.0)
    x3, y3 = neuron_xy(3, 2, 3)
    assert x3 == pytest.approx(2.5 / 3.0)
    assert y3 == pytest.approx(0.0)


def test_edge_rgba_red_blue_and_fade():
    r, g, b, a = edge_rgba(-20.0)
    assert r > b and a >= 250
    r, g, b, a = edge_rgba(20.0)
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


def test_activation_rgba_respects_cmap_argument():
    assert ACT_CMAP_NAME == "managua"
    assert activation_rgba(-2.0) == activation_rgba(-2.0, cmap="managua")
    r_neg, g_neg, b_neg, a_neg = activation_rgba(-2.0, cmap="managua")
    r_zero, g_zero, b_zero, a_zero = activation_rgba(0.0, cmap="managua")
    r_pos, g_pos, b_pos, a_pos = activation_rgba(2.0, cmap="managua")
    assert a_neg == a_zero == a_pos == 255
    assert r_neg > b_neg and g_neg > 140
    assert b_pos > r_pos
    assert max(r_zero, g_zero, b_zero) < 140

    r_rd, g_rd, b_rd, a_rd = activation_rgba(-2.0, cmap="RdYlBu")
    r_yl, g_yl, b_yl, _a = activation_rgba(0.0, cmap="RdYlBu")
    r_bu, g_bu, b_bu, _a = activation_rgba(2.0, cmap="RdYlBu")
    assert a_rd == 255
    assert r_rd > b_rd
    assert b_bu > r_bu
    assert r_yl > 180 and g_yl > 180
    assert (r_neg, g_neg, b_neg) != (r_rd, g_rd, b_rd)


def test_toy_activations_match_grad_sizes():
    g = toy_weight_grads(7, hidden_dim=8, hidden2_dim=10, feature_dim=16)
    a = toy_activations(7, hidden_dim=8, hidden2_dim=10, feature_dim=16)
    assert a.sizes == g.sizes
    assert a.x0.shape == (16,)
    assert a.h1.shape == (16,)
    assert a.h2.shape == (10,)
    assert a.out.shape == (3,)
    assert a.wdl is not None and a.wdl.shape == (3,)
    np.testing.assert_allclose(float(np.sum(a.wdl)), 1.0, atol=1e-5)


def test_standardize_activations_unit_sigma_per_layer():
    a = toy_activations(7, hidden_dim=8, hidden2_dim=10, feature_dim=16)
    z = standardize_activations(a)
    for layer in z.layers:
        np.testing.assert_allclose(np.std(layer), 1.0, atol=1e-5)
    np.testing.assert_allclose(z.wdl, a.wdl)
    assert z.stm_white is a.stm_white


def test_sample_network_maps_activations_match_forward():
    torch.manual_seed(2)
    model = DualHiddenNNUE(hidden_dim=4, hidden2_dim=6)
    model.eval()
    batch = _rand_batch(1, width=10, seed=3)
    grads, acts = sample_network_maps(model, batch)
    assert acts.sizes == grads.sizes
    with torch.no_grad():
        h = model.l1_concat(
            batch["white_idx"],
            batch["black_idx"],
            batch["stm_white"],
            batch["white_mask"],
            batch["black_mask"],
        )
        logits, _pre, h2 = model.head_from_h(h)
        from tinymlinternship.nnue.grad_graph import _dense_from_sparse

        x_white = _dense_from_sparse(
            batch["white_idx"], batch["white_mask"], FEATURE_DIM, model.l1.weight.dtype
        )
        x_black = _dense_from_sparse(
            batch["black_idx"], batch["black_mask"], FEATURE_DIM, model.l1.weight.dtype
        )
        stm = bool(batch["stm_white"][0].item())
        x_stm = x_white[0] if stm else x_black[0]
    np.testing.assert_allclose(acts.h1, h[0].detach().cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(acts.h2, h2[0].detach().cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(acts.out, logits[0].detach().cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(acts.x0, x_stm.detach().cpu().numpy(), atol=1e-5)
    expected_wdl = torch.softmax(logits.float(), dim=-1)[0]
    np.testing.assert_allclose(acts.wdl, expected_wdl.detach().cpu().numpy(), atol=1e-5)
    assert acts.stm_white is stm
    np.testing.assert_allclose(grads.w12, sample_weight_grads(model, batch).w12, atol=1e-5)
