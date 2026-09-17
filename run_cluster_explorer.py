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
  * the **Clusters** spin box is k-means B (default 4). Changing it re-fits
    clusters on the gradient vectors; points stay put, colors update
  * left-click a point to pin its chess board (up to 5; oldest drops off)
  * click the same point again, or the card ×, to unpin
  * drag to pan, scroll to zoom — the connector lines follow
  * cluster checkboxes and the slice box filter the scatter
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
