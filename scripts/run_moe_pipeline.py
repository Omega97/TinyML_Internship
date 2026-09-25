#!/usr/bin/env python3
"""Run sample-gradient → cluster → dispatcher → experts → plots.

Default first run: 1M train rows, B=3, 48-d float16 grads (~92 MiB / million).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.moe_data import slice_folders
from tinymlinternship.nnue.moe_pipeline import (
    DEFAULT_GRAD_BATCH,
    DEFAULT_REDUCE_DIM,
    cluster_gradients,
    compute_gradients,
    configure_torch,
    evaluate_moe,
    fine_tune_experts,
    link_gradient_cache,
    load_base_on_device,
    missing_gradient_cache,
    train_dispatcher,
    write_plots,
)
from tinymlinternship.nnue.sample_gradients import bytes_per_million, max_reduce_dim_for_budget

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_CKPT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
PLOTS_DIR = PROJECT_ROOT / "plots" / "MoE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--run-name", type=str, default="moe_b3_1m")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--max-test", type=int, default=50_000)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument(
        "--algorithm",
        choices=("minibatch_kmeans", "dbscan"),
        default="minibatch_kmeans",
    )
    parser.add_argument(
        "--dbscan-epsilon",
        type=float,
        default=None,
        help="DBSCAN ε quantile in (0, 1). Omit to search the 0.1 grid for --n-clusters.",
    )
    parser.add_argument(
        "--skip-gradients",
        action="store_true",
        help="Reuse gradients.npy and train_pack.pt already in --work-dir.",
    )
    parser.add_argument(
        "--gradient-cache",
        type=Path,
        default=None,
        help="Symlink an existing gradient pack into --work-dir and skip recomputing it.",
    )
    parser.add_argument("--reduce-dim", type=int, default=DEFAULT_REDUCE_DIM)
    parser.add_argument("--grad-batch-size", type=int, default=DEFAULT_GRAD_BATCH)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--expert-labels", choices=("dispatcher", "kmeans"), default="dispatcher")
    parser.add_argument("--dispatcher-epochs", type=int, default=8)
    parser.add_argument("--expert-epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        if args.max_rows == 1_000_000:
            args.max_rows = 4096
        if args.max_test == 50_000:
            args.max_test = 1024
        if args.dispatcher_epochs == 8:
            args.dispatcher_epochs = 2
        if args.expert_epochs == 2:
            args.expert_epochs = 1
        if args.run_name == "moe_b3_1m":
            args.run_name = "moe_smoke"

    cap = max_reduce_dim_for_budget()
    budget = bytes_per_million(int(args.reduce_dim))
    if int(args.reduce_dim) > cap:
        print(
            f"WARNING: --reduce-dim {args.reduce_dim} is "
            f"{budget / (1024 ** 2):.1f} MiB / million (budget 100 MiB, cap {cap})",
            file=sys.stderr,
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(device)
    slices_dir = args.slices_dir if args.slices_dir.is_absolute() else (PROJECT_ROOT / args.slices_dir)
    folders = slice_folders(slices_dir)
    if not folders:
        print(f"no slice folders in {slices_dir}", file=sys.stderr)
        return 1

    work_dir = args.work_dir or (PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe" / args.run_name)
    work_dir = work_dir if work_dir.is_absolute() else (PROJECT_ROOT / work_dir)
    plots_dir = args.plots_dir if args.plots_dir.is_absolute() else (PROJECT_ROOT / args.plots_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else (PROJECT_ROOT / args.checkpoint)
    print(f"device={device} | checkpoint={ckpt}")
    print(f"work_dir={work_dir}")
    print(f"plots_dir={plots_dir}")
    print(
        f"max_rows={args.max_rows:,} B={args.n_clusters} algorithm={args.algorithm} "
        f"reduce_dim={args.reduce_dim} ({budget / (1024 ** 2):.1f} MiB / million)"
    )

    base = load_base_on_device(ckpt, device)
    if args.gradient_cache is not None:
        cache = args.gradient_cache if args.gradient_cache.is_absolute() else (PROJECT_ROOT / args.gradient_cache)
        link_gradient_cache(cache, work_dir)
        args.skip_gradients = True
    if args.skip_gradients:
        missing = missing_gradient_cache(work_dir)
        if missing:
            print(
                f"--skip-gradients: missing {', '.join(missing)} in {work_dir}",
                file=sys.stderr,
            )
            return 1
        print(f"reusing gradient cache in {work_dir}")
    else:
        compute_gradients(
            base,
            folders,
            work_dir,
            device=device,
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
            max_rows=args.max_rows,
            batch_size=args.grad_batch_size,
            reduce_dim=args.reduce_dim,
            resume=args.resume,
        )
    diag = cluster_gradients(
        work_dir,
        n_clusters=args.n_clusters,
        algorithm=args.algorithm,
        dbscan_epsilon=args.dbscan_epsilon,
    )
    n_clusters = int(diag["n_clusters"])
    train_dispatcher(
        base,
        work_dir,
        folders,
        device=device,
        n_clusters=n_clusters,
        epochs=args.dispatcher_epochs,
    )
    moe = fine_tune_experts(
        base,
        work_dir,
        folders,
        device=device,
        n_clusters=n_clusters,
        expert_labels=args.expert_labels,
        epochs=args.expert_epochs,
    )
    metrics = evaluate_moe(
        base,
        moe,
        folders,
        work_dir,
        device=device,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        max_test=args.max_test,
    )
    plots = write_plots(work_dir, plots_dir)
    summary = {
        "run_name": args.run_name,
        "checkpoint": str(ckpt),
        "max_rows": args.max_rows,
        "n_clusters": n_clusters,
        "requested_clusters": args.n_clusters,
        "algorithm": args.algorithm,
        "dbscan_epsilon": diag.get("dbscan_epsilon"),
        "reduce_dim": args.reduce_dim,
        "bytes_per_million": budget,
        "expert_labels": args.expert_labels,
        "metrics": metrics,
        "plots": [str(p) for p in plots],
    }
    (plots_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
