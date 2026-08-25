#!/usr/bin/env python3
"""Train a linear dual-POV WDL net (no hidden layers) from slice ``features.npz``.

Concat ``[STM ‖ opp]`` sparse 844+844 → 3 logits + softmax. Loss is soft
cross-entropy vs Lc0 STM WDL (same labels as DualHidden).

Default test slice: ``fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90``.
Train = every other folder under ``data/processed/board_eval/fen_value_visits/``.

    py -3.12 -u scripts/train_linear_wdl.py --epochs 10 --smoke
    py -3.12 -u scripts/train_linear_wdl.py --epochs 10 --run-name linear_wdl
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
    AcrossSliceBatchSampler,
    FenValueVisitsDataset,
    MixedSliceDataset,
    collate_sparse,
)
from tinymlinternship.nnue.model import LinearWDLNNUE

import train_nnue as tn

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_TEST_SLICE = "fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90"
PLOTS_DIR = PROJECT_ROOT / "plots"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train linear dual-POV WDL (844×2 → 3 softmax, soft CE, no hidden layers)"
    )
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--test-slice", type=str, default=DEFAULT_TEST_SLICE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=NNUE_CHECKPOINTS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="do_compile")
    args = parser.parse_args(argv)

    slices_dir = tn._resolve(args.slices_dir)
    test_name = args.test_slice
    test_folder = slices_dir / test_name
    if not slices_dir.is_dir():
        print(f"slices dir not found: {slices_dir}", file=sys.stderr)
        return 1
    if not test_folder.is_dir():
        print(f"test slice not found: {test_folder}", file=sys.stderr)
        names = [p.name for p in slices_dir.iterdir() if p.is_dir()] if slices_dir.is_dir() else []
        for name in names:
            print(f"  {name}", file=sys.stderr)
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

    print(f"Train: all slices in {slices_dir} except {test_name}")
    print(f"Test:  {test_folder}")
    slice_dss = FenValueVisitsDataset.load_slice_datasets(
        slices_dir,
        skip_names={test_name},
        rebuild=args.rebuild_cache,
        progress=True,
    )
    train_ds = MixedSliceDataset(slice_dss)
    test_ds = FenValueVisitsDataset(
        test_folder,
        rebuild_cache=args.rebuild_cache,
        progress=True,
    )
    pool = len(train_ds)
    if batches_per_epoch > 0:
        n_batches = batches_per_epoch
    elif max_train > 0:
        n_batches = max(1, max_train // batch_size)
    else:
        n_batches = max(1, pool // batch_size)
    print(
        f"  train pool {pool:,} in {len(slice_dss)} slices | test {len(test_ds):,} | "
        f"batch {batch_size} mixed across slices × {n_batches} steps/epoch"
    )
    if args.encode_only:
        print("encode-only: slice DBs ready")
        return 0

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_sampler=AcrossSliceBatchSampler(
            train_ds.sizes,
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

    model = LinearWDLNNUE().to(device)
    if args.do_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or (
        f"linear_wdl{'_smoke' if args.smoke else ''}{'_fast' if args.fast else ''}_{stamp}"
    )
    run_dir = tn._resolve(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "slices_dir": str(slices_dir),
        "test_slice": test_name,
        "rows_train_pool": pool,
        "n_train_slices": len(slice_dss),
        "rows_test": len(test_ds),
        "architecture": "linear_wdl",
        "hidden": "none",
        "epochs": args.epochs,
        "batch_size": batch_size,
        "batches_per_epoch": n_batches,
        "lr": args.lr,
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
        f"linear 844×2 → 3 WDL softmax | {model.count_parameters():,} params | "
        f"loss=soft CE | train steps {n_batches}×{batch_size} | "
        f"test {len(test_ds):,} | {device}"
    )
    print(f"Output: {run_dir}")

    history: list[dict] = []
    best_test = float("inf")
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_ce = tn.train_epoch(model, train_loader, optimizer, device)
        metrics = tn.evaluate(model, test_loader, device)
        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_ce": train_ce,
            "test_ce": metrics["ce"],
            "test_mae": metrics["mae"],
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} | train_ce={train_ce:.6f} | "
            f"test_ce={metrics['ce']:.6f} | test_mae={metrics['mae']:.6f} | "
            f"{elapsed:.1f}s"
        )
        payload = {
            "model_state_dict": model.state_dict(),
            "architecture": "linear_wdl",
            "n_outputs": 3,
            "test_ce": metrics["ce"],
            "epoch": epoch,
        }
        if metrics["ce"] < best_test:
            best_test = metrics["ce"]
            torch.save(payload, best_path)
        torch.save(payload, run_dir / "last.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_path = tn._resolve(args.plot) if args.plot else (PLOTS_DIR / f"{run_name}_ce.png")
    tn.plot_ce(history, plot_path, title="Linear dual-POV WDL (844×2 → 3 softmax)")
    (run_dir / "ce.png").write_bytes(plot_path.read_bytes())
    print(f"Best test_ce={best_test:.6f} → {best_path}")
    print(f"CE plot → {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
