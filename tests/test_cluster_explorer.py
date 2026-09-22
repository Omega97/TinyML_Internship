"""Cluster explorer data model, selection, and offscreen window."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME
from tinymlinternship.nnue.cluster_explore import (
    DEFAULT_POOL_SIZE,
    FenResolver,
    SelectionModel,
    cluster_cache_key,
    cluster_color,
    cluster_palette,
    default_work_dir,
    load_table,
    load_work_dir,
    make_demo_data,
    orbit_project,
    project_gradients,
    recluster,
    reload_pool,
    reproject,
    resolve_position,
    stm_value_from_wdl,
)

SMOKE2 = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe" / "moe_smoke2"


def test_selection_toggle_is_unbounded():
    sel = SelectionModel()
    sel.toggle(10)
    sel.toggle(20)
    sel.toggle(30)
    assert sel.order == [10, 20, 30]
    selected = sel.toggle(40)
    assert selected == [10, 20, 30, 40]
    selected = sel.toggle(30)
    assert selected == [10, 20, 40]
    for i in range(8):
        sel.toggle(100 + i)
    assert len(sel.order) == 11


def test_demo_data_has_legal_fens():
    import chess

    data = make_demo_data(n=40, n_clusters=4, seed=1)
    assert len(data) == 40
    assert data.n_clusters == 4
    assert data.fen is not None
    chess.Board(str(data.fen[0]))
    info = data.row(0)
    assert info.fen
    assert info.eval_target is not None


def test_cluster_palette_matches_plan():
    colors = cluster_palette(4)
    assert colors[0].lower() == "#e74c3c"
    assert colors[1].lower() == "#daa520"
    assert cluster_color(0) == colors[0]


def test_light_theme_is_white_and_does_not_swap_data_colors():
    from tinymlinternship.nnue.cluster_explorer_ui import LIGHT_THEME

    assert LIGHT_THEME.window_bg.lower() == "#ffffff"
    assert LIGHT_THEME.text.lower() == "#111111"
    assert LIGHT_THEME.selected_pen.lower() == "#111111"
    assert LIGHT_THEME.grid_alpha > 0
    assert cluster_color(0).lower() == "#e74c3c"
    assert cluster_color(1).lower() == "#daa520"


def test_stm_value_from_wdl():
    assert stm_value_from_wdl([0.6, 0.3, 0.1]) == pytest.approx(0.5)
    assert stm_value_from_wdl(None) is None


def test_load_table_roundtrip(tmp_path: Path):
    data = make_demo_data(n=12, n_clusters=3, seed=2)
    path = tmp_path / "clusters.parquet"
    pd.DataFrame(
        {
            "sample_id": data.sample_id,
            "coord_x": data.coord_x,
            "coord_y": data.coord_y,
            "cluster_id": data.cluster_id,
            "fen": data.fen,
            "eval_target": data.eval_target,
            "grad_norm": data.grad_norm,
        }
    ).to_parquet(path)
    loaded = load_table(path)
    assert len(loaded) == 12
    assert loaded.fen is not None
    assert str(loaded.fen[0]) == str(data.fen[0])


def test_visible_mask_filters_clusters():
    data = make_demo_data(n=80, n_clusters=4, seed=3)
    mask = data.visible_mask(clusters=[0, 2])
    assert set(np.unique(data.cluster_id[mask])).issubset({0, 2})


def test_project_gradients_all_methods():
    rng = np.random.RandomState(0)
    x = rng.randn(36, 8).astype(np.float32)
    pca = project_gradients(x, method="pca", seed=0)
    assert pca.shape == (36, 2)
    for method in ("tsne", "isomap", "lle"):
        coords = project_gradients(x, method=method, seed=0)
        assert coords.shape == (36, 2), method
        assert not np.allclose(coords, pca, atol=1e-3), method
    try:
        umap_coords = project_gradients(x, method="umap", seed=0)
    except ImportError:
        pytest.skip("umap-learn not installed")
    assert umap_coords.shape == (36, 2)


def test_project_gradients_three_components():
    rng = np.random.RandomState(0)
    x = rng.randn(40, 8).astype(np.float32)
    pca3 = project_gradients(x, method="pca", n_components=3, seed=0)
    assert pca3.shape == (40, 3)
    pca2 = project_gradients(x, method="pca", n_components=2, seed=0)
    np.testing.assert_allclose(pca3[:, :2], pca2, atol=1e-4)
    lle3 = project_gradients(x, method="lle", n_components=3, seed=0)
    assert lle3.shape == (40, 3)


def test_reproject_2d_and_3d_caches_are_independent():
    data = make_demo_data(n=40, n_clusters=3, seed=1)
    x2 = data.coord_x.copy()
    reproject(data, "pca", seed=1, n_components=3)
    assert data.coord_z is not None
    assert data.coord_z.shape == (40,)
    assert not np.allclose(data.coord_x, x2)
    x3 = data.coord_x.copy()
    z3 = data.coord_z.copy()
    reproject(data, "pca", seed=1, n_components=2)
    np.testing.assert_allclose(data.coord_x, x2)
    assert data.coord_z is None
    reproject(data, "pca", seed=1, n_components=3)
    np.testing.assert_allclose(data.coord_x, x3)
    np.testing.assert_allclose(data.coord_z, z3)


def test_orbit_project_keeps_origin_and_rotates():
    x = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    z = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sx, sy = orbit_project(x, y, z, azimuth_deg=0.0, elevation_deg=0.0)
    np.testing.assert_allclose(sx[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(sy[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(sx[1], 0.0, atol=1e-5)
    np.testing.assert_allclose(sy[1], 0.0, atol=1e-5)
    np.testing.assert_allclose(sx[2], 1.0, atol=1e-5)
    sx90, sy90 = orbit_project(x, y, z, azimuth_deg=90.0, elevation_deg=0.0)
    np.testing.assert_allclose(sx90[1], -1.0, atol=1e-5)
    np.testing.assert_allclose(sy90[1], 0.0, atol=1e-5)


def test_reproject_uses_cache():
    data = make_demo_data(n=40, n_clusters=3, seed=1)
    x0 = data.coord_x.copy()
    y0 = data.coord_y.copy()
    reproject(data, "lle", seed=1)
    assert data.method == "lle"
    assert not np.allclose(data.coord_x, x0)
    lle_x = data.coord_x.copy()
    reproject(data, "pca", seed=1)
    assert data.method == "pca"
    np.testing.assert_allclose(data.coord_x, x0)
    np.testing.assert_allclose(data.coord_y, y0)
    reproject(data, "lle", seed=1)
    np.testing.assert_allclose(data.coord_x, lle_x)


def test_recluster_kmedoids_and_dbscan():
    data = make_demo_data(n=80, n_clusters=4, seed=6)
    recluster(data, 5, seed=6, algorithm="kmedoids")
    assert data.algorithm == "kmedoids"
    assert data.n_clusters == 5
    assert int(data.cluster_id.min()) >= 0
    recluster(data, 4, seed=6, algorithm="dbscan")
    assert data.algorithm == "dbscan"
    assert int(data.cluster_id.min()) >= -1
    assert data.dbscan_epsilon == pytest.approx(0.3)
    assert data.diagnostics.get("dbscan_percentile") == pytest.approx(30.0)
    bigger = make_demo_data(n=300, n_clusters=4, seed=6)
    recluster(bigger, 4, seed=6, algorithm="dbscan", dbscan_epsilon=0.5)
    assert bigger.algorithm == "dbscan"
    assert bigger.dbscan_epsilon == pytest.approx(0.5)
    assert bigger.diagnostics.get("dbscan_percentile") == pytest.approx(50.0)


def test_reload_pool_changes_n_not_display_only():
    data = make_demo_data(n=300, n_clusters=3, seed=7)
    bigger = reload_pool(data, 1000, n_clusters=3, algorithm="kmeans")
    assert len(bigger) == 1000
    assert bigger.algorithm == "kmeans"


def test_default_pool_is_one_third_of_previous():
    assert DEFAULT_POOL_SIZE == 10_000


def test_recluster_changes_k():
    data = make_demo_data(n=80, n_clusters=4, seed=5)
    assert data.n_clusters == 4
    recluster(data, 6, seed=5)
    assert data.n_clusters == 6
    assert data.cluster_id.min() >= 0
    assert data.cluster_id.max() < 6
    assert data.diagnostics["n_clusters"] == 6
    assert len(data.diagnostics["sizes"]) == 6


def test_recluster_uses_cache(monkeypatch):
    data = make_demo_data(n=80, n_clusters=4, seed=5)
    labels4 = data.cluster_id.copy()
    assert cluster_cache_key("kmeans", 4) in data.cluster_cache
    recluster(data, 5, seed=5)
    labels5 = data.cluster_id.copy()
    assert not np.array_equal(labels4, labels5)
    import tinymlinternship.nnue.cluster as cluster_mod

    calls = {"n": 0}
    real = cluster_mod.fit_minibatch_kmeans

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cluster_mod, "fit_minibatch_kmeans", wrapped)
    recluster(data, 4, seed=5)
    np.testing.assert_array_equal(data.cluster_id, labels4)
    recluster(data, 5, seed=5)
    np.testing.assert_array_equal(data.cluster_id, labels5)
    assert calls["n"] == 0


def test_cluster_cache_key_dbscan_ignores_k():
    assert cluster_cache_key("dbscan", 3) == cluster_cache_key("DBSCAN", 99)
    assert cluster_cache_key("dbscan", 3) == "dbscan:0.3"
    assert cluster_cache_key("dbscan", 3, dbscan_epsilon=0.3) != cluster_cache_key(
        "dbscan", 3, dbscan_epsilon=0.7
    )
    assert cluster_cache_key("kmeans", 4) != cluster_cache_key("kmeans", 5)


def test_quantize_dbscan_epsilon():
    from tinymlinternship.nnue.cluster_explore import quantize_dbscan_epsilon

    assert quantize_dbscan_epsilon(None) == pytest.approx(0.3)
    assert quantize_dbscan_epsilon(0.32) == pytest.approx(0.3)
    assert quantize_dbscan_epsilon(0.0) == pytest.approx(0.1)
    assert quantize_dbscan_epsilon(1.0) == pytest.approx(0.9)
    assert quantize_dbscan_epsilon(0.55) == pytest.approx(0.6)


def test_dbscan_epsilon_is_cached():
    data = make_demo_data(n=80, n_clusters=4, seed=6)
    recluster(data, 4, seed=6, algorithm="dbscan", dbscan_epsilon=0.3)
    labels_a = data.cluster_id.copy()
    recluster(data, 4, seed=6, algorithm="dbscan", dbscan_epsilon=0.7)
    assert cluster_cache_key("dbscan", 4, dbscan_epsilon=0.3) in data.cluster_cache
    assert cluster_cache_key("dbscan", 4, dbscan_epsilon=0.7) in data.cluster_cache
    recluster(data, 4, seed=6, algorithm="dbscan", dbscan_epsilon=0.3)
    np.testing.assert_array_equal(data.cluster_id, labels_a)
    assert data.dbscan_epsilon == pytest.approx(0.3)


@pytest.mark.skipif(not (SMOKE2 / "gradients.npy").is_file(), reason="moe_smoke2 missing")
def test_load_work_dir_smoke2():
    data = load_work_dir(SMOKE2, max_points=64, method="pca", seed=0, n_clusters=4)
    assert len(data) == 64
    assert data.n_clusters == 4
    assert data.features is not None
    assert data.slice_id is not None
    assert data.local_row is not None
    info = resolve_position(data, 0, resolver=FenResolver(data.folders))
    assert info.fen, "expected a FEN from the slice parquet"
    import chess

    chess.Board(info.fen)
    assert info.eval_target is None or np.isfinite(info.eval_target)
    recluster(data, 8, seed=0)
    assert data.n_clusters == 8
    assert int(data.cluster_id.max()) < 8


def test_default_work_dir_finds_a_run():
    found = default_work_dir(PROJECT_ROOT)
    if found is None:
        pytest.skip("no MoE work dir on disk")
    assert (found / "gradients.npy").is_file()


def test_side_to_move_and_board_frame_colors():
    from tinymlinternship.nnue.cluster_explorer_ui import (
        LIGHT_THEME,
        _BOARD_COLORS_BLACK,
        _BOARD_COLORS_WHITE,
        _board_svg,
        position_detail_text,
        side_to_move_name,
    )

    white_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    black_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert side_to_move_name(white_fen) == "White"
    assert side_to_move_name(black_fen) == "Black"
    white_svg = _board_svg(white_fen).decode("utf-8")
    black_svg = _board_svg(black_fen).decode("utf-8")
    assert _BOARD_COLORS_WHITE["margin"] in white_svg
    assert _BOARD_COLORS_WHITE["coord"] in white_svg
    assert _BOARD_COLORS_BLACK["margin"] in black_svg
    assert _BOARD_COLORS_BLACK["coord"] in black_svg
    data = make_demo_data(n=8, n_clusters=3, seed=0)
    text = position_detail_text(data.row(0))
    assert text.startswith("to play  ")
    assert "FEN  " in text


def _wait_idle(win, timeout_ms: int = 8000) -> None:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QTest

    waited = 0
    while getattr(win, "_compute_busy", False) and waited < timeout_ms:
        QTest.qWait(20)
        waited += 20
    QApplication.processEvents()
    assert not getattr(win, "_compute_busy", False), win.status.text()


def test_offscreen_window_pins_cards():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QLabel

    from tinymlinternship.nnue.cluster_explorer_ui import (
        BOARD_SVG_SIZE,
        DARK_THEME,
        LIGHT_THEME,
        _BOARD_COLORS_WHITE,
        _board_svg,
        ClusterExplorerWindow,
    )

    app = QApplication.instance() or QApplication([])
    data = make_demo_data(n=30, n_clusters=3, seed=4)
    win = ClusterExplorerWindow(data, checkpoint=None)
    win.show()
    app.processEvents()
    assert win.isVisible()
    label_texts = [w.text() for w in win.findChildren(QLabel)]
    assert "Side inspector" not in label_texts
    assert not any("Click a point" in (t or "") for t in label_texts)
    assert not hasattr(win, "max_spin")
    assert not hasattr(win, "slice_edit")
    assert win.light_mode.isChecked() is False
    assert win.theme.window_bg.lower() == "#121418"
    win._toggle(0)
    win._toggle(1)
    win._toggle(2)
    app.processEvents()
    assert len(win.cards) == 3
    assert BOARD_SVG_SIZE == 165
    assert win.cards[0].svg.width() == BOARD_SVG_SIZE
    assert win.cards[0].svg.height() == BOARD_SVG_SIZE
    assert win.cards[0].grad_view.width() == BOARD_SVG_SIZE
    assert win.cards[0].grad_view.height() == BOARD_SVG_SIZE
    assert win.cards[0].grad_view._grads is not None
    assert win.cards[0].grad_view._grads.sizes[0] >= 3
    assert win.cards[0].grad_view._grads.sizes[-1] == 3
    assert win.cards[0].grad_view._acts is not None
    assert win.cards[0].grad_view._acts.sizes == win.cards[0].grad_view._grads.sizes
    assert win.cards[0].grad_view._acts.out.shape == (3,)
    assert DARK_THEME.input_bg.lower() == DARK_THEME.panel_bg.lower()
    assert LIGHT_THEME.input_bg.lower() == LIGHT_THEME.panel_bg.lower()
    assert win.theme.input_bg.lower() == win.theme.panel_bg.lower()
    assert "QSpinBox" in win.styleSheet()
    assert win.theme.panel_bg.lower() in win.styleSheet().lower()
    meta0 = win.cards[0].detail_text
    assert meta0.startswith("to play  ")
    assert "White" in meta0 or "Black" in meta0
    assert not win.cards[0].hover_info.isVisible()
    assert len(win.selection) == 3
    links = list(win.iter_link_geometry())
    assert len(links) == 3
    win._toggle(1)
    app.processEvents()
    assert 1 not in win.cards
    assert len(win.selection) == 2
    win._clear_selection()
    app.processEvents()
    assert len(win.cards) == 0
    win._toggle(0)
    app.processEvents()
    win.k_spin.setValue(6)
    _wait_idle(win)
    assert win.data.n_clusters == 6
    assert not hasattr(win, "cluster_boxes")
    from PyQt6.QtWidgets import QAbstractSpinBox

    assert win.k_spin.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.UpDownArrows
    assert win.k_spin.isVisible()
    assert not win.eps_spin.isVisible()
    assert win.wait_spin.value() == 10
    assert win._timeout_ms == 10_000
    assert 0 in win.cards
    assert "Cluster " in win.cards[0].header_label.text()
    win.proj_buttons["isomap"].click()
    _wait_idle(win)
    assert win.data.method == "isomap"
    labels_before = win.data.cluster_id.copy()
    n_before = len(win._shown)
    win.display_slider.setValue(30)
    app.processEvents()
    assert np.array_equal(win.data.cluster_id, labels_before)
    assert len(win._shown) <= n_before
    win.algo_buttons["kmedoids"].click()
    _wait_idle(win)
    assert win.data.algorithm == "kmedoids"
    win._clear_selection()
    app.processEvents()
    for i in range(6):
        win._toggle(i)
    app.processEvents()
    assert len(win.cards) == 6
    assert len(win.selection) == 6
    win.light_mode.setChecked(True)
    app.processEvents()
    assert win.theme is LIGHT_THEME
    assert win.theme.window_bg.lower() == "#ffffff"
    assert win.theme.text.lower() == "#111111"
    assert "#ffffff" in win.styleSheet().lower()
    assert win.plot.backgroundBrush().color().name().lower() == "#ffffff"
    white_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    svg = _board_svg(white_fen).decode("utf-8")
    assert _BOARD_COLORS_WHITE["margin"] in svg
    win.light_mode.setChecked(False)
    app.processEvents()
    assert win.theme.window_bg.lower() == "#121418"
    assert win.view3d.isChecked() is False
    win.view3d.setChecked(True)
    _wait_idle(win)
    assert win._view3d is True
    assert win.data.coord_z is not None
    assert win.data.coord_z.shape == (len(win.data),)
    x_before = np.asarray(win._disp_x).copy()
    win._azimuth = (win._azimuth + 40.0) % 360.0
    win._update_display_coords()
    win._apply_display_coords()
    app.processEvents()
    assert not np.allclose(win._disp_x, x_before)
    win.view3d.setChecked(False)
    _wait_idle(win)
    assert win._view3d is False
    win.close()


def test_grad_pixmap_activation_dots_use_cmap():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue.cluster_explorer_ui import (
        BOARD_SVG_SIZE,
        GRAD_GRAPH_PAD,
        _render_grad_pixmap,
    )
    from tinymlinternship.nnue.grad_graph import (
        NnueActivations,
        neuron_xy,
        toy_activations,
        toy_weight_grads,
    )

    app = QApplication.instance() or QApplication([])
    assert app is not None
    grads = toy_weight_grads(1, hidden_dim=4, hidden2_dim=6, feature_dim=8)
    base = toy_activations(1, hidden_dim=4, hidden2_dim=6, feature_dim=8)
    acts = NnueActivations(
        x0=np.zeros_like(base.x0),
        h1=np.zeros_like(base.h1),
        h2=np.zeros_like(base.h2),
        out=np.array([2.0, 0.0, -2.0], dtype=np.float32),
        hidden_dim=base.hidden_dim,
        hidden2_dim=base.hidden2_dim,
        feature_dim=base.feature_dim,
    )
    pix = _render_grad_pixmap(grads, BOARD_SVG_SIZE, activations=acts, cmap="managua")
    pix_alt = _render_grad_pixmap(grads, BOARD_SVG_SIZE, activations=acts, cmap="RdYlBu")
    img = pix.toImage()
    pad = float(GRAD_GRAPH_PAD)
    span = float(BOARD_SVG_SIZE) - 2.0 * pad
    n_out = 3

    def sample(image, index: int):
        x, y = neuron_xy(3, index, n_out)
        px = int(round(pad + float(x) * span))
        py = int(round(pad + float(y) * span))
        return image.pixelColor(px, py)

    pos = sample(img, 0)
    neg = sample(img, 2)
    assert pos.alpha() > 0 and pos.blue() > pos.red()
    assert neg.alpha() > 0 and neg.red() > neg.blue()
    alt = sample(pix_alt.toImage(), 0)
    assert (pos.red(), pos.green(), pos.blue()) != (alt.red(), alt.green(), alt.blue())


def test_dbscan_replaces_k_with_epsilon():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue.cluster_explorer_ui import ClusterExplorerWindow

    app = QApplication.instance() or QApplication([])
    data = make_demo_data(n=40, n_clusters=3, seed=17)
    win = ClusterExplorerWindow(data, checkpoint=None, timeout_ms=8000)
    win.show()
    app.processEvents()
    assert win.k_spin.isVisible()
    assert not win.eps_spin.isVisible()
    assert win.cluster_param_label.text() == "Clusters"
    k_before = int(win.k_spin.value())
    win.algo_buttons["dbscan"].click()
    _wait_idle(win)
    assert win.data.algorithm == "dbscan"
    assert not win.k_spin.isVisible()
    assert win.eps_spin.isVisible()
    assert win.cluster_param_label.text() == "ε"
    assert win.eps_spin.value() == pytest.approx(0.3)
    assert win.eps_spin.minimum() == pytest.approx(0.1)
    assert win.eps_spin.maximum() == pytest.approx(0.9)
    assert win.eps_spin.singleStep() == pytest.approx(0.1)
    win.eps_spin.setValue(0.6)
    _wait_idle(win)
    assert win.data.dbscan_epsilon == pytest.approx(0.6)
    assert win.data.diagnostics.get("dbscan_percentile") == pytest.approx(60.0)
    win.algo_buttons["kmeans"].click()
    _wait_idle(win)
    assert win.data.algorithm == "kmeans"
    assert win.k_spin.isVisible()
    assert not win.eps_spin.isVisible()
    assert win.cluster_param_label.text() == "Clusters"
    assert int(win.k_spin.value()) == k_before
    win.close()


def test_projection_timeout_reverts_to_previous(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import time

    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue import cluster_explorer_ui as ui

    app = QApplication.instance() or QApplication([])

    def slow_proj(gradients, *, method="pca", n_components=2, seed=0):
        time.sleep(0.35)
        n = int(np.asarray(gradients).shape[0])
        k = max(2, int(n_components))
        rng = np.random.RandomState(int(seed) + 17)
        return rng.randn(n, k).astype(np.float32)

    monkeypatch.setattr(ui, "project_gradients", slow_proj)
    data = make_demo_data(n=24, n_clusters=3, seed=8)
    win = ui.ClusterExplorerWindow(data, checkpoint=None, timeout_ms=60)
    win.show()
    app.processEvents()
    assert win.data.method == "pca"
    win.proj_buttons["tsne"].click()
    _wait_idle(win, timeout_ms=2000)
    assert win.data.method == "pca"
    assert win.proj_buttons["pca"].isChecked()
    assert not win.proj_buttons["tsne"].isChecked()
    assert "timed out" in win.status.text().lower()
    assert "pca" in win.status.text().lower()
    win.proj_buttons["pca"].click()
    app.processEvents()
    assert win.data.method == "pca"
    win.close()


def test_timeout_does_not_block_pca_or_cached_pool(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import time

    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue import cluster_explorer_ui as ui

    app = QApplication.instance() or QApplication([])
    started = {"n": 0}
    released = {"n": 0}

    def slow_proj(gradients, *, method="pca", n_components=2, seed=0):
        started["n"] += 1
        time.sleep(0.5)
        released["n"] += 1
        n = int(np.asarray(gradients).shape[0])
        k = max(2, int(n_components))
        rng = np.random.RandomState(int(seed) + 23)
        return rng.randn(n, k).astype(np.float32)

    monkeypatch.setattr(ui, "project_gradients", slow_proj)
    data300 = make_demo_data(n=300, n_clusters=3, seed=12)
    data1000 = make_demo_data(n=1000, n_clusters=3, seed=12)
    win = ui.ClusterExplorerWindow(data300, checkpoint=None, timeout_ms=80)
    win._pool_cache[300] = data300
    win._pool_cache[1000] = data1000
    win.show()
    app.processEvents()
    ids300 = win.data.sample_id.copy()
    win.pool_buttons[1000].click()
    app.processEvents()
    assert len(win.data) == 1000
    win.proj_buttons["tsne"].click()
    _wait_idle(win, timeout_ms=2000)
    assert win.data.method == "pca"
    assert started["n"] >= 1
    assert released["n"] == 0
    win.view3d.setChecked(True)
    _wait_idle(win, timeout_ms=2000)
    assert win._view3d is False
    assert not win.view3d.isChecked()
    win.proj_buttons["pca"].click()
    app.processEvents()
    assert win.data.method == "pca"
    win.pool_buttons[300].click()
    app.processEvents()
    np.testing.assert_array_equal(win.data.sample_id, ids300)
    assert len(win.data) == 300
    assert win.data.method == "pca"
    win.close()


def test_fast_projection_still_applies():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue.cluster_explorer_ui import ClusterExplorerWindow

    app = QApplication.instance() or QApplication([])
    data = make_demo_data(n=24, n_clusters=3, seed=9)
    win = ClusterExplorerWindow(data, checkpoint=None, timeout_ms=5000)
    win.show()
    app.processEvents()
    win.proj_buttons["lle"].click()
    _wait_idle(win)
    assert win.data.method == "lle"
    win.close()


def test_window_projection_cache_skips_recompute(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue import cluster_explorer_ui as ui

    app = QApplication.instance() or QApplication([])
    calls = {"n": 0}
    real = ui.project_gradients

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ui, "project_gradients", counted)
    data = make_demo_data(n=24, n_clusters=3, seed=10)
    win = ui.ClusterExplorerWindow(data, checkpoint=None, timeout_ms=5000)
    win.show()
    app.processEvents()
    win.proj_buttons["lle"].click()
    _wait_idle(win)
    assert calls["n"] == 1
    win.proj_buttons["pca"].click()
    _wait_idle(win)
    win.proj_buttons["lle"].click()
    _wait_idle(win)
    assert win.data.method == "lle"
    assert calls["n"] == 1
    win.close()


def test_window_pool_cache_restores_sample_ids(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue import cluster_explorer_ui as ui

    app = QApplication.instance() or QApplication([])
    data = make_demo_data(n=300, n_clusters=3, seed=11)
    win = ui.ClusterExplorerWindow(data, checkpoint=None, timeout_ms=8000)
    win.show()
    app.processEvents()
    assert len(win.data) == 300
    ids300 = win.data.sample_id.copy()
    win.pool_buttons[1000].click()
    _wait_idle(win)
    assert len(win.data) == 1000
    ids1000 = win.data.sample_id.copy()
    calls = {"n": 0}
    real = ui.reload_pool

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ui, "reload_pool", counted)
    win.pool_buttons[300].click()
    _wait_idle(win)
    np.testing.assert_array_equal(win.data.sample_id, ids300)
    win.pool_buttons[1000].click()
    _wait_idle(win)
    np.testing.assert_array_equal(win.data.sample_id, ids1000)
    assert calls["n"] == 0
    win.close()


def test_pool_switch_keeps_projection_method():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue.cluster_explorer_ui import ClusterExplorerWindow

    app = QApplication.instance() or QApplication([])
    data = make_demo_data(n=300, n_clusters=3, seed=13)
    win = ClusterExplorerWindow(data, checkpoint=None, timeout_ms=8000)
    win.show()
    app.processEvents()
    win.proj_buttons["lle"].click()
    _wait_idle(win)
    assert win.data.method == "lle"
    win.pool_buttons[1000].click()
    _wait_idle(win)
    assert len(win.data) == 1000
    assert win.data.method == "lle"
    assert win.proj_buttons["lle"].isChecked()
    win.pool_buttons[300].click()
    _wait_idle(win)
    assert len(win.data) == 300
    assert win.data.method == "lle"
    assert win.proj_buttons["lle"].isChecked()
    win.close()


def test_pool_switch_keeps_clustering():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue.cluster_explorer_ui import ClusterExplorerWindow

    app = QApplication.instance() or QApplication([])
    data = make_demo_data(n=300, n_clusters=3, seed=15)
    win = ClusterExplorerWindow(data, checkpoint=None, timeout_ms=8000)
    win.show()
    app.processEvents()
    win.algo_buttons["kmedoids"].click()
    _wait_idle(win)
    win.k_spin.setValue(6)
    _wait_idle(win)
    assert win.data.algorithm == "kmedoids"
    assert win.data.n_clusters == 6
    win.pool_buttons[1000].click()
    _wait_idle(win)
    assert len(win.data) == 1000
    assert win.data.algorithm == "kmedoids"
    assert win.algo_buttons["kmedoids"].isChecked()
    assert win.k_spin.value() == 6
    assert win.data.n_clusters == 6
    win.pool_buttons[300].click()
    _wait_idle(win)
    assert len(win.data) == 300
    assert win.data.algorithm == "kmedoids"
    assert win.algo_buttons["kmedoids"].isChecked()
    assert win.k_spin.value() == 6
    assert win.data.n_clusters == 6
    win.close()


def test_pool_switch_timeout_keeps_clustering(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import time

    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue import cluster_explorer_ui as ui

    app = QApplication.instance() or QApplication([])
    data300 = make_demo_data(n=300, n_clusters=3, seed=16)
    data1000 = make_demo_data(n=1000, n_clusters=3, seed=16)
    win = ui.ClusterExplorerWindow(data300, checkpoint=None, timeout_ms=80)
    win._pool_cache[300] = data300
    win._pool_cache[1000] = data1000
    win.show()
    app.processEvents()
    win.algo_buttons["kmedoids"].click()
    _wait_idle(win)
    assert win.data.algorithm == "kmedoids"
    assert win.algo_buttons["kmedoids"].isChecked()

    real = ui.compute_cluster_payload

    def slow_cluster(*args, **kwargs):
        time.sleep(0.4)
        return real(*args, **kwargs)

    monkeypatch.setattr(ui, "compute_cluster_payload", slow_cluster)
    win.pool_buttons[1000].click()
    _wait_idle(win, timeout_ms=2000)
    assert len(win.data) == 300
    assert win.pool_buttons[300].isChecked()
    assert not win.pool_buttons[1000].isChecked()
    assert win.data.algorithm == "kmedoids"
    assert win.algo_buttons["kmedoids"].isChecked()
    win.close()
