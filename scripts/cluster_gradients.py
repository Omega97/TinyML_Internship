#!/usr/bin/env python3
"""Cluster projected sample gradients (mini-batch k-means or DBSCAN)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME
from tinymlinternship.nnue.moe_pipeline import DEFAULT_CLUSTER_BATCH, cluster_gradients


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument(
        "--algorithm",
        choices=("minibatch_kmeans", "dbscan"),
        default="minibatch_kmeans",
    )
    parser.add_argument("--dbscan-epsilon", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CLUSTER_BATCH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    work_dir = args.input_dir if args.input_dir.is_absolute() else (PROJECT_ROOT / args.input_dir)
    cluster_gradients(
        work_dir,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        seed=args.seed,
        algorithm=args.algorithm,
        dbscan_epsilon=args.dbscan_epsilon,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
