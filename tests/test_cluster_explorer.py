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
    FenResolver,
    SelectionModel,
    cluster_color,
    cluster_palette,
    default_work_dir,
    load_table,
    load_work_dir,
    make_demo_data,
    recluster,
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
    win.close()
