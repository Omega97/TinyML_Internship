"""Sample-gradient MoE: analytic grads, projection budget, routing, clustering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tinymlinternship.nnue.cluster import (
    cluster_diagnostics,
    cluster_size_histogram,
    fit_minibatch_kmeans,
)
from tinymlinternship.nnue.model import DualHiddenNNUE, crelu
from tinymlinternship.nnue.moe import DualHiddenMoE, LinearDispatcher
from tinymlinternship.nnue.moe_data import subsample_parts
from tinymlinternship.nnue.sample_gradients import (
    EPS,
    analytic_head_grad_flat,
    bytes_per_million,
    flatten_head_grad_autograd,
    head_grad_norm_sq,
    head_parameter_dim,
    make_projection_matrix,
    max_reduce_dim_for_budget,
    normalize_projected,
    open_float16_store,
    project_head_grads,
    sample_head_gradients,
)


def _rand_batch(n: int, width: int = 8, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "white_idx": torch.randint(0, 844, (n, width), generator=g),
        "white_mask": torch.ones(n, width, dtype=torch.bool),
        "black_idx": torch.randint(0, 844, (n, width), generator=g),
        "black_mask": torch.ones(n, width, dtype=torch.bool),
        "stm_white": torch.randint(0, 2, (n,), generator=g).bool(),
        "target": torch.softmax(torch.randn(n, 3, generator=g), dim=-1),
        "weight": torch.ones(n),
    }


def _analytic_pieces(model: DualHiddenNNUE, batch: dict[str, torch.Tensor]):
    target = batch["target"].float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    h = model.l1_concat(
        batch["white_idx"],
        batch["black_idx"],
        batch["stm_white"],
        batch["white_mask"],
        batch["black_mask"],
    )
    pre_l2 = model.l2(h)
    h2 = crelu(pre_l2, model.crelu_clip)
    logits = model.head(h2)
    probs = torch.softmax(logits.float(), dim=-1)
    g_logits = (probs - target).to(dtype=h.dtype)
    g_h2 = g_logits @ model.head.weight.to(dtype=g_logits.dtype)
    mask = (pre_l2 > 0) & (pre_l2 < model.crelu_clip)
    g_pre = g_h2 * mask.to(dtype=g_h2.dtype)
    return g_pre, h, g_logits, h2


def test_head_parameter_dim_matches_spec():
    assert head_parameter_dim(128, 256) == 66_563


def test_bytes_per_million_under_100mib_at_default():
    assert bytes_per_million(48) < 100 * 1024 * 1024
    assert bytes_per_million(64) > 100 * 1024 * 1024
    assert max_reduce_dim_for_budget() == 52


def test_analytic_head_grads_match_autograd():
    torch.manual_seed(0)
    model = DualHiddenNNUE(hidden_dim=4, hidden2_dim=6)
    model.eval()
    batch = _rand_batch(5, width=8, seed=1)
    with torch.no_grad():
        g_pre, h, g_logits, h2 = _analytic_pieces(model, batch)
        flat = analytic_head_grad_flat(g_pre, h, g_logits, h2)
    ref = flatten_head_grad_autograd(model, batch)
    assert flat.shape == ref.shape
    assert torch.allclose(flat, ref, atol=1e-4, rtol=1e-4)


def test_fused_projection_matches_matrix_multiply():
    torch.manual_seed(2)
    model = DualHiddenNNUE(hidden_dim=4, hidden2_dim=6)
    model.eval()
    batch = _rand_batch(4, seed=3)
    p = head_parameter_dim(model.hidden_dim, model.hidden2_dim)
    r = make_projection_matrix(p, reduce_dim=7, seed=4)
    with torch.no_grad():
        g_pre, h, g_logits, h2 = _analytic_pieces(model, batch)
        fused = project_head_grads(g_pre, h, g_logits, h2, r)
        flat = analytic_head_grad_flat(g_pre, h, g_logits, h2)
        naive = flat @ r.T
        norm_sq = head_grad_norm_sq(g_pre, h, g_logits, h2)
        assert torch.allclose(norm_sq, (flat * flat).sum(dim=-1), atol=1e-4)
        assert torch.allclose(fused, naive, atol=1e-4, rtol=1e-4)
        unit = normalize_projected(fused, norm_sq)
        ref_unit = naive / (flat.norm(dim=-1, keepdim=True) + EPS)
        assert torch.allclose(unit, ref_unit, atol=1e-4, rtol=1e-4)


def test_sample_head_gradients_unit_and_deterministic():
    torch.manual_seed(5)
    model = DualHiddenNNUE(hidden_dim=4, hidden2_dim=6)
    model.eval()
    batch = _rand_batch(6, seed=6)
    p = head_parameter_dim(4, 6)
    r = make_projection_matrix(p, 8, seed=7)
    a = sample_head_gradients(model, batch, r)
    b = sample_head_gradients(model, batch, r)
    assert torch.allclose(a, b, atol=1e-6)
    assert a.shape == (6, 8)
    assert not torch.isnan(a).any()


def test_mmap_resume_does_not_clobber(tmp_path: Path):
    path = tmp_path / "grads.npy"
    store = open_float16_store(path, 8, 4, resume=False)
    store[0:3] = np.arange(12, dtype=np.float16).reshape(3, 4)
    store.flush()
    del store
    again = open_float16_store(path, 8, 4, resume=True)
    assert np.allclose(again[1], np.arange(4, 8, dtype=np.float16))
    again[3:5] = 1.0
    again.flush()
    assert np.allclose(again[0], np.arange(4, dtype=np.float16))


def _three_blobs(n_per: int = 80, dim: int = 8, scale: float = 0.05):
    rng = np.random.RandomState(0)
    shifts = (
        np.zeros(dim, dtype=np.float32),
        np.array([5.0] + [0.0] * (dim - 1), dtype=np.float32),
        np.array([0.0, 5.0] + [0.0] * (dim - 2), dtype=np.float32),
    )
    parts = [rng.randn(n_per, dim).astype(np.float32) * scale + shift for shift in shifts]
    return np.concatenate(parts, axis=0)


def test_minibatch_kmeans_recovers_three_blobs():
    rng = np.random.RandomState(0)
    blobs = [
        rng.randn(80, 8) * 0.1 + np.array([0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]),
        rng.randn(80, 8) * 0.1 + np.array([5.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]),
        rng.randn(80, 8) * 0.1 + np.array([0.0, 5.0, 0.0, 0.0, 0, 0, 0, 0]),
    ]
    x = np.concatenate(blobs, axis=0).astype(np.float32)
    km = fit_minibatch_kmeans(x, n_clusters=3, batch_size=40, seed=0)
    labels = km.predict(x)
    sizes = cluster_size_histogram(labels, 3)
    assert sizes.min() >= 40
    diag = cluster_diagnostics(labels, km.cluster_centers_, inertia=float(km.inertia_))
    assert diag["empty"] == 0


def test_dbscan_recovers_blobs_and_assigns_noise():
    from tinymlinternship.nnue.cluster import fit_dbscan, fit_dbscan_for_b

    blobs = _three_blobs()
    outlier = np.zeros((1, blobs.shape[1]), dtype=np.float32)
    outlier[0, 0] = 50.0
    x = np.concatenate([blobs, outlier], axis=0)
    labels, _centroids, diag = fit_dbscan(x, epsilon=0.9, seed=0)
    assert diag["algorithm"] == "dbscan"
    assert diag["n_clusters"] == 3
    assert diag["n_noise"] >= 1
    assert labels.shape == (x.shape[0],)
    assert int(labels.min()) >= 0
    nearest_blob = int(np.bincount(labels[80:160].astype(np.int64)).argmax())
    assert int(labels[-1]) == nearest_blob

    targeted, _c2, diag_b = fit_dbscan_for_b(blobs, n_clusters=3, seed=0)
    assert targeted.shape == (blobs.shape[0],)
    assert diag_b["n_clusters"] == 3
    assert diag_b["dbscan_b_match"] is True
    assert any(trial["n_clusters"] >= 2 for trial in diag_b["dbscan_trials"])


def test_dbscan_fit_cap_labels_every_row():
    from tinymlinternship.nnue.cluster import fit_dbscan_for_b

    x = _three_blobs()
    labels, _centroids, diag = fit_dbscan_for_b(x, n_clusters=3, seed=0, fit_cap=60)
    assert labels.shape == (x.shape[0],)
    assert int(labels.min()) >= 0
    assert diag["dbscan_fit_rows"] == 60
    assert diag["n_rows_full"] == int(x.shape[0])


def test_cluster_gradients_dbscan_roundtrip(tmp_path: Path):
    from tinymlinternship.nnue.moe_pipeline import cluster_gradients

    x = _three_blobs().astype(np.float16)
    np.save(tmp_path / "gradients.npy", x)
    diag = cluster_gradients(
        tmp_path, n_clusters=3, algorithm="dbscan", log=None
    )
    labels = np.load(tmp_path / "labels.npy")
    assert diag["algorithm"] == "dbscan"
    assert labels.shape == (x.shape[0],)
    assert int(labels.min()) >= 0
    assert (tmp_path / "centroids.npy").is_file()
    assert (tmp_path / "diagnostics.json").is_file()


def test_battery_commands_cover_kmeans_and_dbscan():
    from tinymlinternship.nnue.results_battery import (
        format_pipeline_command,
        reference_runs,
        select_runs,
    )

    wave1 = select_runs("1")
    algos = {run.algorithm for run in wave1}
    assert algos == {"minibatch_kmeans", "dbscan"}
    assert {run.n_clusters for run in wave1} == {2, 4, 8}
    assert all(run.reuse_gradients for run in wave1)
    text = format_pipeline_command(reference_runs()[0], python="python")
    assert "--algorithm minibatch_kmeans" in text
    assert "--gradient-cache data/processed/board_eval/moe/moe_b4_2m" in text
    assert "--n-clusters 2" in text
    dbscan = next(run for run in wave1 if run.algorithm == "dbscan" and run.n_clusters == 4)
    dbscan_text = format_pipeline_command(dbscan, python="python")
    assert "--algorithm dbscan" in dbscan_text
    assert "RESULTS/moe/W128_H256/dbscan_b4" in dbscan_text
    probes = select_runs("3")
    assert probes
    assert all(run.reuse_gradients is None for run in probes)
    assert "--gradient-cache" not in format_pipeline_command(probes[0], python="python")


def test_dispatcher_loss_drops_on_linear_h():
    torch.manual_seed(0)
    n, dim, b = 256, 16, 3
    h = torch.randn(n, dim)
    labels = torch.randint(0, b, (n,))
    for i in range(b):
        h[labels == i, i] += 6.0
    disp = LinearDispatcher(dim, b)
    opt = torch.optim.Adam(disp.parameters(), lr=0.05)
    first = last = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(disp(h), labels)
        if first is None:
            first = float(loss.item())
        loss.backward()
        opt.step()
        last = float(loss.item())
    acc = float((disp.predict(h) == labels).float().mean())
    assert last < first
    assert acc > 0.7


def test_mlp_dispatcher_loss_drops_and_predicts():
    from tinymlinternship.nnue.moe import MLPDispatcher

    torch.manual_seed(0)
    n, dim, b = 256, 16, 3
    h = torch.randn(n, dim)
    labels = torch.randint(0, b, (n,))
    for i in range(b):
        h[labels == i, i] += 6.0
    disp = MLPDispatcher(dim, b, hidden_dim=64)
    assert disp.in_dim == dim and disp.hidden_dim == 64 and disp.n_clusters == b
    opt = torch.optim.Adam(disp.parameters(), lr=0.05)
    first = last = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(disp(h), labels)
        if first is None:
            first = float(loss.item())
        loss.backward()
        opt.step()
        last = float(loss.item())
    acc = float((disp.predict(h) == labels).float().mean())
    assert last < first
    assert acc > 0.7


def test_expert_step_does_not_change_l1():
    torch.manual_seed(0)
    base = DualHiddenNNUE(hidden_dim=4, hidden2_dim=6)
    moe = DualHiddenMoE.from_base(base, n_experts=2)
    moe.freeze_l1()
    before = moe.l1.weight.detach().clone()
    batch = _rand_batch(8, seed=9)
    opt = torch.optim.Adam(
        [p for p in moe.parameters() if p.requires_grad],
        lr=1e-2,
    )
    logits = moe(
        batch["white_idx"],
        batch["black_idx"],
        batch["stm_white"],
        batch["white_mask"],
        batch["black_mask"],
    )
    loss = F.cross_entropy(logits, batch["target"].argmax(dim=-1))
    loss.backward()
    opt.step()
    assert torch.equal(moe.l1.weight.detach(), before)


def test_moe_routing_matches_single_expert():
    torch.manual_seed(0)
    base = DualHiddenNNUE(hidden_dim=4, hidden2_dim=6)
    moe = DualHiddenMoE.from_base(base, n_experts=3)
    batch = _rand_batch(10, seed=11)
    h = moe.l1_concat(
        batch["white_idx"],
        batch["black_idx"],
        batch["stm_white"],
        batch["white_mask"],
        batch["black_mask"],
    )
    ids = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2, 1], dtype=torch.long)
    routed, got_ids = moe.route_from_h(h, expert_ids=ids)
    assert torch.equal(got_ids, ids)
    for i, expert in enumerate(ids.tolist()):
        one = moe.expert_logits_from_h(h[i : i + 1], expert)
        assert torch.allclose(routed[i : i + 1], one, atol=1e-5)


def test_assign_to_centroids_matches_kmeans():
    from tinymlinternship.nnue.cluster import assign_to_centroids

    rng = np.random.RandomState(1)
    x = rng.randn(60, 4).astype(np.float32)
    km = fit_minibatch_kmeans(x, n_clusters=3, batch_size=20, seed=1)
    a = km.predict(x)
    b = assign_to_centroids(x, km.cluster_centers_)
    assert np.array_equal(a.astype(np.int16), b)


def test_write_plots_smoke(tmp_path: Path):
    from tinymlinternship.nnue.moe_pipeline import write_plots

    rng = np.random.RandomState(0)
    n, d, b = 180, 8, 2
    grads = rng.randn(n, d).astype(np.float16)
    labels = np.repeat(np.arange(b, dtype=np.int16), n // b)
    disp = labels.copy()
    disp[::7] = 1 - disp[::7]
    centroids = np.stack([grads[labels == i].astype(np.float32).mean(0) for i in range(b)])
    work = tmp_path / "work"
    work.mkdir()
    np.save(work / "gradients.npy", grads)
    np.save(work / "labels.npy", labels)
    np.save(work / "labels_dispatcher.npy", disp)
    np.save(work / "centroids.npy", centroids)
    cosine = np.eye(b, dtype=np.float64)
    (work / "diagnostics.json").write_text(
        json.dumps(
            {
                "n_clusters": b,
                "sizes": [int((labels == 0).sum()), int((labels == 1).sum())],
                "centroid_cosine": cosine.tolist(),
            }
        ),
        encoding="utf-8",
    )
    (work / "dispatcher_history.json").write_text(
        json.dumps(
            {
                "history": [
                    {"epoch": 1, "train_acc": 0.5, "val_acc": 0.4},
                    {"epoch": 2, "train_acc": 0.7, "val_acc": 0.6},
                ]
            }
        ),
        encoding="utf-8",
    )
    (work / "expert_metrics.json").write_text(
        json.dumps(
            {
                "buckets": [
                    {"expert": 0, "base_hold_ce": 0.70, "expert_hold_ce": 0.65, "skipped": 0},
                    {"expert": 1, "base_hold_ce": 0.71, "expert_hold_ce": 0.66, "skipped": 0},
                ]
            }
        ),
        encoding="utf-8",
    )
    (work / "eval.json").write_text(
        json.dumps({"base_ce": 0.70, "moe_ce": 0.68, "base_mae": 0.3, "moe_mae": 0.29}),
        encoding="utf-8",
    )
    plots = tmp_path / "plots"
    written = write_plots(work, plots)
    names = {p.name for p in written}
    assert "cluster_pca.png" in names
    assert "moe_vs_base_ce.png" in names
    assert (plots / "cluster_pca.png").is_file()


def test_subsample_parts_respects_budget():
    parts = [
        np.arange(10),
        np.arange(20, 35),
        np.arange(100, 105),
    ]
    got = subsample_parts(parts, max_rows=8, seed=0)
    assert sum(int(p.size) for p in got) == 8
    full = subsample_parts(parts, max_rows=0, seed=0)
    assert [int(p.size) for p in full] == [10, 15, 5]
