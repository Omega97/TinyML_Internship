#!/usr/bin/env python3
"""Open the clustering exploration window so you can click points and inspect boards.

This is the program you run. It starts the Qt app and keeps it open until you
close the window.

From this folder::

    python3.12 run_cluster_explorer.py

Synthetic boards (no MoE run required)::

    python3.12 run_cluster_explorer.py --demo

A specific cluster run::

    python3.12 run_cluster_explorer.py --work-dir data/processed/board_eval/moe/moe_b3_1m

In the window:
  * top bar: clustering algorithm, working-set size (300…30k), display %
    (the slider only hides points; it does not re-cluster)
  * **Projection** toggles PCA / t-SNE / UMAP / Isomap / LLE and recomputes
    the 2D layout (cluster colors stay; t-SNE/UMAP/Isomap/LLE can take a bit)
  * the **Clusters** spin box is B for k-Means / k-Medoids (default 4)
  * left-click a point to pin its chess board (no pin cap)
  * click the same point again, or the card ×, to unpin
  * drag to pan, scroll to zoom — the connector lines follow
  * cluster checkboxes filter the scatter
  * **Light mode** (off by default): pure white plot, black text, light-gray
    grid — cluster colors and chess pieces stay the same
  * Esc clears the pins
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "explore_clusters.py"

if not SCRIPT.is_file():
    raise SystemExit(f"missing {SCRIPT}")

sys.argv = [str(SCRIPT), *sys.argv[1:]]
runpy.run_path(str(SCRIPT), run_name="__main__")
