#!/usr/bin/env python3
"""Train a medium dual-POV WDL net (one hidden layer) from slice ``features.npz``.

Concat ``[STM ‖ opp]`` sparse 844+844 → H CReLU → 3 logits + softmax.
Default H=20. Loss is soft cross-entropy vs Lc0 STM WDL.

Default split: a random ``--test-fraction`` (0.10) of each slice is test
(seed ``--split-seed``). Train batches and both CE metrics are row-weighted.

    py -3.12 -u scripts/train_medium_wdl.py --epochs 10 --smoke
    py -3.12 -u scripts/train_medium_wdl.py --epochs 50 --lr 0.01 --run-name medium_h20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

import torch
from torch.utils.data import DataLoader

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import (
    UniformRowBatchSampler,
    collate_sparse,
)
from tinymlinternship.nnue.model import MediumWDLNNUE

import train_nnue as tn

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
PLOTS_DIR = PROJECT_ROOT / "plots"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train medium dual-POV WDL (844×2 → H CReLU → 3 softmax, soft CE)"
    )
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    tn.add_test_fraction_arg(parser)
    tn.add_split_seed_arg(parser)
    parser.add_argument("--hidden-dim", type=int, default=20, help="Hidden width after STM concat")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Initial Adam learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lr-end",
        type=float,
        default=None,
        help="Final learning rate after linear decay over --epochs (default: same as --lr, constant)",
    )
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--test-subset-size",
        type=int,
        default=0,
        help="Per-epoch test rows (0 = full set). Same subset every epoch. "
        "If set, best.pt is scored on the full test set once at the end.",
    )
    parser.add_argument("--test-subset-seed", type=int, default=0)
    parser.add_argument(
        "--train-val-subset-size",
        type=int,
        default=0,
        help="Frozen train CE rows (0 = same as --test-subset-size, else 3200)",
    )
    parser.add_argument(
        "--train-val-subset-seed",
        type=int,
        default=None,
        help="RNG seed for frozen train CE (default: same as --test-subset-seed)",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=NNUE_CHECKPOINTS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="CE plot path (e.g. plots/foo.png). Default: plots/{run_name}_ce.png",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="do_compile")
    args = parser.parse_args(argv)

    if args.hidden_dim < 1:
        print("--hidden-dim must be >= 1", file=sys.stderr)
        return 1

    slices_dir = tn._resolve(args.slices_dir)
    if not slices_dir.is_dir():
        print(f"slices dir not found: {slices_dir}", file=sys.stderr)
        return 1
    if not 0.0 < float(args.test_fraction) < 1.0:
        print("--test-fraction must be in (0, 1), e.g. 0.10", file=sys.stderr)
        return 1

    batch_size = args.batch_size
    batches_per_epoch = args.batches_per_epoch
    if args.fast:
        if batch_size == 2048:
            batch_size = 256
        if batches_per_epoch == 0:
            batches_per_epoch = 40
    max_train = args.max_train
    if args.smoke and max_train == 0:
        max_train = 20_000

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    print(
        f"Split: random {args.test_fraction:.0%} of each slice → test "
        f"(seed={args.split_seed}); rest → train"
    )
    print(f"Slices: {slices_dir}")
    try:
        train_ds, test_ds_full = tn.prepare_random_split(
            slices_dir,
            args.test_fraction,
            seed=args.split_seed,
            rebuild=args.rebuild_cache,
            progress=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    n_test_full = len(test_ds_full)
    test_ds = tn.maybe_subset_dataset(
        test_ds_full, args.test_subset_size, args.test_subset_seed
    )
    n_test_eval = len(test_ds)
    pool = len(train_ds)
    if batches_per_epoch > 0:
        n_batches = batches_per_epoch
    elif max_train > 0:
        n_batches = max(1, max_train // batch_size)
    else:
        n_batches = max(1, pool // batch_size)
    print(
        f"  train pool {pool:,} in {len(train_ds.slices)} slices | "
        f"test {n_test_eval:,}/{n_test_full:,} | "
        f"batch {batch_size} row-weighted × {n_batches} steps/epoch"
    )
    if n_test_eval < n_test_full:
        print(
            f"  per-epoch test subset {n_test_eval:,} of {n_test_full:,} "
            f"(seed={args.test_subset_seed}); full eval of best.pt at the end"
        )
    if args.encode_only:
        print("encode-only: slice DBs ready")
        return 0

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_sampler=UniformRowBatchSampler(
            pool,
            batch_size=batch_size,
            batches=n_batches,
        ),
        num_workers=args.workers,
        collate_fn=collate_sparse,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_sparse,
        pin_memory=pin,
    )
    train_eval_n = tn.frozen_train_eval_n(args)
    train_eval_seed = tn.resolve_train_eval_seed(args)
    train_val_ds = tn.maybe_subset_dataset(
        train_ds, train_eval_n, train_eval_seed
    )
    train_val_loader = DataLoader(
        train_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_sparse,
        pin_memory=pin,
    )
    print(
        f"  frozen train_ce on {len(train_val_ds):,}/{pool:,} rows "
        f"(seed={train_eval_seed}, same row-weighted measure as test; "
        f"n={train_eval_n:,} is noisy, full compare is printed at the end)"
    )

    model = MediumWDLNNUE(hidden_dim=args.hidden_dim).to(device)
    if args.do_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model, mode="reduce-overhead")
    if args.lr <= 0.0:
        print("--lr must be > 0", file=sys.stderr)
        return 1
    if args.lr_end is not None and args.lr_end <= 0.0:
        print("--lr-end must be > 0", file=sys.stderr)
        return 1
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = tn.make_linear_lr_scheduler(
        optimizer, lr=args.lr, lr_end=args.lr_end, epochs=args.epochs
    )

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or (
        f"medium_h{args.hidden_dim}"
        f"{'_smoke' if args.smoke else ''}{'_fast' if args.fast else ''}_{stamp}"
    )
    run_dir = tn._resolve(args.output_dir) / run_name
    tn.warn_if_run_dir_busy(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "slices_dir": str(slices_dir),
        "test_fraction": float(args.test_fraction),
        "split": "random fraction of each slice → test",
        "split_seed": int(args.split_seed),
        "train_ce": "frozen row-weighted",
        "rows_train_pool": pool,
        "n_train_slices": len(train_ds.slices),
        "rows_test": n_test_eval,
        "rows_test_full": n_test_full,
        "test_subset_size": int(args.test_subset_size),
        "test_subset_seed": int(args.test_subset_seed),
        "train_val_subset_size": int(len(train_val_ds)),
        "train_val_subset_seed": int(train_eval_seed),
        "architecture": "medium_wdl",
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "batches_per_epoch": n_batches,
        "lr": args.lr,
        "lr_end": args.lr if args.lr_end is None else args.lr_end,
        "lr_schedule": "linear" if scheduler is not None else "constant",
        "max_train": max_train,
        "smoke": bool(args.smoke),
        "fast": bool(args.fast),
        "device": str(device),
        "parameters": model.count_parameters(),
        "loss": "unweighted soft cross-entropy, target = STM WDL",
        "output": "3 logits + softmax (W, D, L) from side to move",
        "compiled": bool(args.do_compile),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"medium 844×2 → {args.hidden_dim} CReLU → 3 WDL softmax | "
        f"{model.count_parameters():,} params | loss=soft CE | "
        f"train steps {n_batches}×{batch_size} | "
        f"test {n_test_eval:,}/{n_test_full:,} | {device}"
    )
    print(f"Output: {run_dir}")

    history: list[dict] = []
    best_test = float("inf")
    best_path = run_dir / "best.pt"
    best_state: dict | None = None
    history.append(
        tn.eval_untrained(
            model, train_loader, test_loader, device, train_val_loader=train_val_loader
        )
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tn.train_epoch(model, train_loader, optimizer, device)
        metrics = tn.evaluate(model, test_loader, device)
        train_ce = tn.evaluate(model, train_val_loader, device)["ce"]
        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_ce": train_ce,
            "test_ce": metrics["ce"],
            "seconds": elapsed,
        }
        history.append(row)
        tn.log_epoch(epoch, train_ce, metrics["ce"], elapsed)
        if scheduler is not None:
            scheduler.step()
        payload = {
            "model_state_dict": model.state_dict(),
            "architecture": "medium_wdl",
            "hidden_dim": args.hidden_dim,
            "n_outputs": 3,
            "test_ce": metrics["ce"],
            "epoch": epoch,
        }
        if metrics["ce"] < best_test:
            best_test = metrics["ce"]
            best_state = tn.clone_state(model)
            torch.save(payload, best_path)
        torch.save(payload, run_dir / "last.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_path = tn.resolve_plot_path(args.plot, run_name, PLOTS_DIR)
    tn.plot_ce(
        history,
        plot_path,
        title=f"Medium dual-POV WDL (844×2 → {args.hidden_dim} CReLU → 3 softmax)",
    )
    (run_dir / "ce.png").write_bytes(plot_path.read_bytes())
    print(f"Best test_ce={best_test:.6f} → {best_path}")
    print(f"CE plot → {plot_path}")
    if best_path.is_file() or best_state is not None:
        tn.report_matched_train_test_ce(
            model,
            train_ds,
            test_ds_full,
            pool=pool,
            n_test_full=n_test_full,
            batch_size=batch_size,
            device=device,
            workers=args.workers,
            pin=pin,
            best_state=best_state,
        )
    if n_test_eval < n_test_full and best_path.is_file():
        full_loader = DataLoader(
            test_ds_full,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_sparse,
            pin_memory=pin,
        )
        tn.report_full_test(model, best_path, full_loader, device, state_dict=best_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
