#!/usr/bin/env python3
"""Train dual-POV DualHidden NNUE from per-slice ``features.npz`` DBs.

Default split:
  test  = fen_value_visits_lichess_db_standard_rated_2026-07_100000-105000_d80_draw5
  train = every other folder under data/processed/board_eval/fen_value_visits/

No chess encoding at train time. Loss is unweighted soft cross-entropy on
STM WDL (visits omitted). Re-encode slices after the WDL schema change
(``--rebuild-cache`` or ``encode_slice_features.py --rebuild``).

    py -3.12 -u scripts/train_nnue.py --epochs 5 --smoke
    py -3.12 -u scripts/train_nnue.py --epochs 5 --fast
    py -3.12 -u scripts/train_nnue.py --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import (
    AcrossSliceBatchSampler,
    FenValueVisitsDataset,
    MixedSliceDataset,
    collate_sparse,
)
from tinymlinternship.nnue.model import DualHiddenNNUE

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_TEST_SLICE = "fen_value_visits_lichess_db_standard_rated_2026-07_100000-105000_d80_draw5"
PLOTS_DIR = PROJECT_ROOT / "plots"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ce_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Soft cross-entropy: ``-sum_k t_k log softmax(z)_k``, mean over the batch."""
    log_p = F.log_softmax(logits, dim=-1)
    nll = -(target * log_p).sum(dim=-1)
    w = weight.clamp(min=0.0)
    denom = w.sum().clamp(min=1e-8)
    return (nll * w).sum() / denom


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ce_sum = 0.0
    mae_sum = 0.0
    w_sum = 0.0
    n = 0
    for batch in loader:
        logits = model.forward_sparse(
            batch["white_idx"].to(device),
            batch["white_mask"].to(device),
            batch["black_idx"].to(device),
            batch["black_mask"].to(device),
            batch["stm_white"].to(device),
        )
        target = batch["target"].to(device)
        weight = batch["weight"].to(device)
        w = weight.clamp(min=0.0)
        log_p = F.log_softmax(logits, dim=-1)
        nll = -(target * log_p).sum(dim=-1)
        probs = F.softmax(logits, dim=-1)
        pred_v = probs[:, 0] - probs[:, 2]
        tgt_v = target[:, 0] - target[:, 2]
        ce_sum += float((nll * w).sum().item())
        mae_sum += float(((pred_v - tgt_v).abs() * w).sum().item())
        w_sum += float(w.sum().item())
        n += int(target.shape[0])
    return {
        "ce": ce_sum / max(w_sum, 1e-8),
        "mae": mae_sum / max(w_sum, 1e-8),
        "n": n,
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    n_batches = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_sparse(
            batch["white_idx"].to(device),
            batch["white_mask"].to(device),
            batch["black_idx"].to(device),
            batch["black_mask"].to(device),
            batch["stm_white"].to(device),
        )
        loss = ce_loss(logits, batch["target"].to(device), batch["weight"].to(device))
        loss.backward()
        optimizer.step()
        running += float(loss.item())
        n_batches += 1
    return running / max(n_batches, 1)


def plot_ce(history: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_ce = [row["train_ce"] for row in history]
    test_ce = [row["test_ce"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, train_ce, marker="o", label="train CE")
    ax.plot(epochs, test_ce, marker="s", label="test CE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Dual-POV NNUE WDL (L1 64×2, L2 128)")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a single DualHidden NNUE (L1 64×2, L2 128, 3-way STM WDL, soft CE)"
    )
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument(
        "--test-slice",
        type=str,
        default=DEFAULT_TEST_SLICE,
        help="Folder name under --slices-dir used only as the test set",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="L1 width per POV")
    parser.add_argument("--hidden2-dim", type=int, default=128, help="L2 width")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=0,
        help="Optimizer steps per epoch (0 = one pass over the pooled train size)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster run: batch-size 256 and 40 mixed batches/epoch unless those flags are set",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=0, help="0 = all train rows")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Cap train at 20k mixed samples/epoch unless --max-train is set",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=NNUE_CHECKPOINTS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Re-encode features.npz even if valid",
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Ensure slice DBs exist and exit",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="do_compile",
        help="torch.compile (off by default)",
    )
    args = parser.parse_args(argv)

    slices_dir = _resolve(args.slices_dir)
    test_name = args.test_slice
    test_folder = slices_dir / test_name
    if not slices_dir.is_dir():
        print(f"slices dir not found: {slices_dir}", file=sys.stderr)
        return 1
    if not test_folder.is_dir():
        print(f"test slice not found: {test_folder}", file=sys.stderr)
        names = [p.name for p in slices_dir.iterdir() if p.is_dir()] if slices_dir.is_dir() else []
        if names:
            print("available slices:", file=sys.stderr)
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

    model = DualHiddenNNUE(
        hidden_dim=args.hidden_dim,
        hidden2_dim=args.hidden2_dim,
    ).to(device)
    if args.do_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or (
        f"dual_W{args.hidden_dim}_H{args.hidden2_dim}"
        f"{'_smoke' if args.smoke else ''}{'_fast' if args.fast else ''}_{stamp}"
    )
    run_dir = _resolve(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "slices_dir": str(slices_dir),
        "test_slice": test_name,
        "rows_train_pool": pool,
        "n_train_slices": len(slice_dss),
        "rows_test": len(test_ds),
        "hidden_dim": args.hidden_dim,
        "hidden2_dim": args.hidden2_dim,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "batches_per_epoch": n_batches,
        "sample": "each batch: batch_size draws, uniform over slices then rows",
        "lr": args.lr,
        "max_train": max_train,
        "smoke": bool(args.smoke),
        "fast": bool(args.fast),
        "device": str(device),
        "parameters": model.count_parameters(),
        "architecture": "single DualHiddenNNUE (no expert buckets)",
        "loss": "unweighted soft cross-entropy, target = STM WDL",
        "output": "3 logits + softmax (W, D, L) from side to move",
        "compiled": bool(args.do_compile),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"single net | L1 {args.hidden_dim}×2 + L2 {args.hidden2_dim} + WDL softmax | "
        f"{model.count_parameters():,} params | loss=soft CE | "
        f"train steps {n_batches}×{batch_size} | test {len(test_ds):,} | {device}"
    )
    print(f"Output: {run_dir}")

    history: list[dict[str, Any]] = []
    best_test = float("inf")
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_ce = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, test_loader, device)
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
            "architecture": "dual_hidden_wdl",
            "hidden_dim": args.hidden_dim,
            "hidden2_dim": args.hidden2_dim,
            "n_outputs": 3,
            "test_ce": metrics["ce"],
            "epoch": epoch,
        }
        if metrics["ce"] < best_test:
            best_test = metrics["ce"]
            torch.save(payload, best_path)
        torch.save(payload, run_dir / "last.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_path = _resolve(args.plot) if args.plot else (PLOTS_DIR / f"{run_name}_ce.png")
    plot_ce(history, plot_path)
    (run_dir / "ce.png").write_bytes(plot_path.read_bytes())
    print(f"Best test_ce={best_test:.6f} → {best_path}")
    print(f"CE plot → {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
