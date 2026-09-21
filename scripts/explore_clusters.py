#!/usr/bin/env python3
"""Interactive explorer for per-sample gradient clusters.

Opens a Qt window: scatter of the reduced gradients, a side inspector with
chess boards plus that sample's NNUE gradient graph, and live connector
lines from selected points to their cards.

    python3.12 scripts/explore_clusters.py
    python3.12 scripts/explore_clusters.py --demo
    python3.12 scripts/explore_clusters.py --work-dir data/processed/board_eval/moe/moe_b3_1m
    python3.12 scripts/explore_clusters.py --table clusters.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROJECT_ROOT
from tinymlinternship.nnue.cluster_explore import (
    DEFAULT_POOL_SIZE,
    default_work_dir,
    load_table,
    load_work_dir,
    make_demo_data,
    recluster,
)

DEFAULT_CKPT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"


def _resolve(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_try = (Path.cwd() / path).resolve()
    if cwd_try.exists():
        return cwd_try
    return (PROJECT_ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="MoE run directory with gradients.npy + labels.npy (default: newest local run)",
    )
    parser.add_argument("--table", type=Path, default=None, help="CSV/parquet with coord_x/coord_y/cluster_id/fen")
    parser.add_argument("--demo", action="store_true", help="Ignore on-disk data and open synthetic clusters")
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help="Working-set size for projection/clustering (default 10000; 0 = all rows)",
    )
    parser.add_argument(
        "--method",
        choices=("pca", "tsne", "umap", "isomap", "lle"),
        default="pca",
        help="Initial projection (changeable in the window; 3D is a live toggle)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="k-means cluster count B (default 4). Change it live in the window to re-fit.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="NNUE checkpoint for ŷ (optional)")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Open offscreen, pin two demo points, assert cards, then exit",
    )
    return parser.parse_args(argv)


def load_data(args: argparse.Namespace):
    k = int(args.n_clusters)
    if args.demo:
        print(f"loading synthetic demo clusters  B={k}")
        n_demo = 300 if int(args.max_points) in (0, DEFAULT_POOL_SIZE) else int(args.max_points)
        return make_demo_data(n=max(n_demo, k), n_clusters=k, seed=int(args.seed))
    if args.table is not None:
        path = _resolve(args.table)
        print(f"loading table {path}")
        data = load_table(path)
        recluster(data, k, seed=int(args.seed))
        return data
    work_dir = _resolve(args.work_dir) if args.work_dir is not None else default_work_dir(PROJECT_ROOT)
    if work_dir is None:
        print("no MoE work-dir found; falling back to --demo")
        return make_demo_data(n=300, n_clusters=k, seed=int(args.seed))
    print(f"loading {work_dir}  method={args.method}  max_points={args.max_points}  B={k}")
    return load_work_dir(
        work_dir,
        max_points=int(args.max_points),
        method=str(args.method),
        seed=int(args.seed),
        n_components=int(args.n_components),
        n_clusters=k,
    )


def run_self_test(data) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    from tinymlinternship.nnue.cluster_explorer_ui import ClusterExplorerWindow

    app = QApplication.instance() or QApplication([])
    win = ClusterExplorerWindow(data, checkpoint=None)
    win.show()
    app.processEvents()
    assert win.isVisible(), "window did not become visible"
    win._toggle(0)
    win._toggle(min(1, len(data) - 1))
    app.processEvents()
    n_cards = len(win.cards)
    n_sel = len(win.selection)
    assert n_cards == n_sel == 2, f"expected 2 pins, got cards={n_cards} sel={n_sel}"
    win._toggle(0)
    app.processEvents()
    assert len(win.cards) == 1, "toggle-off failed"
    win.k_spin.setValue(6)
    app.processEvents()
    assert win.data.n_clusters == 6, f"expected B=6, got {win.data.n_clusters}"
    assert len(win.cluster_boxes) == 6
    win.proj_buttons["lle"].click()
    app.processEvents()
    assert win.data.method == "lle", f"expected lle, got {win.data.method}"
    labels_before = win.data.cluster_id.copy()
    win.display_slider.setValue(40)
    app.processEvents()
    assert np.array_equal(win.data.cluster_id, labels_before), "display % must not recluster"
    assert len(win._shown) <= len(win.data)
    print("self-test ok: window opened, pin/unpin, recluster, projection, display %")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_data(args)
    print(
        f"{len(data):,} points  clusters={data.n_clusters}  "
        f"source={data.source}  method={data.method}"
    )
    if args.self_test:
        return run_self_test(data)

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("No DISPLAY set. The Qt window cannot open on this session.", file=sys.stderr)
        return 2

    ckpt = _resolve(args.checkpoint) if args.checkpoint else None
    if ckpt is not None and not ckpt.is_file():
        ckpt = None

    from tinymlinternship.nnue.cluster_explorer_ui import run_explorer

    return run_explorer(data, checkpoint=ckpt)


if __name__ == "__main__":
    raise SystemExit(main())
