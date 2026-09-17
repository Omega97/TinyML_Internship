"""Cluster explorer data model, FIFO selection, and offscreen window."""

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
    cluster_color,
    cluster_palette,
    default_work_dir,
    load_table,
    load_work_dir,
    make_demo_data,
    project_gradients,
    recluster,
    reload_pool,
    reproject,
    resolve_position,
    stm_value_from_wdl,
)

SMOKE2 = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe" / "moe_smoke2"


def test_selection_fifo_toggle():
    sel = SelectionModel(max_n=3)
    sel.toggle(10)
    sel.toggle(20)
    sel.toggle(30)
    assert sel.order == [10, 20, 30]
    selected, dropped = sel.toggle(40)
    assert dropped == 10
    assert selected == [20, 30, 40]
    selected, dropped = sel.toggle(30)
    assert dropped is None
    assert selected == [20, 40]
    sel.max_n = 1
    assert sel.order == [40]


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
    assert 3 <= data.n_clusters <= 6
    bigger = make_demo_data(n=300, n_clusters=4, seed=6)
    recluster(bigger, 4, seed=6, algorithm="dbscan")
    assert 3 <= bigger.n_clusters <= 6


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


def test_offscreen_window_pins_cards():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue.cluster_explorer_ui import ClusterExplorerWindow

    app = QApplication.instance() or QApplication([])
    data = make_demo_data(n=30, n_clusters=3, seed=4)
    win = ClusterExplorerWindow(data, max_select=5, checkpoint=None)
    win.show()
    app.processEvents()
    assert win.isVisible()
    win._toggle(0)
    win._toggle(1)
    win._toggle(2)
    app.processEvents()
    assert len(win.cards) == 3
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
    app.processEvents()
    assert win.data.n_clusters == 6
    assert len(win.cluster_boxes) == 6
    assert 0 in win.cards
    assert "Cluster " in win.cards[0].header_label.text()
    win.proj_buttons["isomap"].click()
    app.processEvents()
    assert win.data.method == "isomap"
    labels_before = win.data.cluster_id.copy()
    n_before = len(win._shown)
    win.display_slider.setValue(30)
    app.processEvents()
    assert np.array_equal(win.data.cluster_id, labels_before)
    assert len(win._shown) <= n_before
    win.algo_buttons["kmedoids"].click()
    app.processEvents()
    assert win.data.algorithm == "kmedoids"
    win.close()
