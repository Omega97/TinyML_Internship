#!/usr/bin/env python3
"""Train SARDINE NNUE on labeled parquet splits.

Default architecture is **F3 single-head** (``844 → W`` dual POV → ``2W → 1``).
Legacy multi-expert pilots use ``--architecture bucketed``.

Works with ChessBench (precomputed features) or production labeled sets
(FEN + expected_reward; encode_dual at load time). Full production scale
still targets nnue-pytorch + large Lichess/Lc0 — see blueprint §Training.
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
import torch.nn as nn
from torch.utils.data import DataLoader

from tinymlinternship.config.settings import (
    CHESSBENCH_PROCESSED_DIR,
    NNUE_CHECKPOINTS_DIR,
)
from tinymlinternship.features import NUM_BUCKETS
from tinymlinternship.nnue import Architecture, ChessbenchDataset, build_nnue

# Mini production merge (override with --train / --val; ChessBench is pilot-only).
_LABELED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "labeled"
_DEFAULT_TRAIN = (
    _LABELED_DIR / "train.parquet"
    if (_LABELED_DIR / "train.parquet").exists()
    else CHESSBENCH_PROCESSED_DIR / "splits" / "train.parquet"
)
_DEFAULT_VAL = (
    _LABELED_DIR / "val.parquet"
    if (_LABELED_DIR / "val.parquet").exists()
    else CHESSBENCH_PROCESSED_DIR / "splits" / "val.parquet"
)


def collate_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "white_features": torch.stack([item["white_features"] for item in batch]),
        "black_features": torch.stack([item["black_features"] for item in batch]),
        "bucket_id": torch.tensor([item["bucket_id"] for item in batch], dtype=torch.long),
        "stm_white": torch.tensor([item["stm_white"] for item in batch], dtype=torch.bool),
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_abs = 0.0
    total_n = 0
    bucket_sq = torch.zeros(NUM_BUCKETS, dtype=torch.float64)
    bucket_n = torch.zeros(NUM_BUCKETS, dtype=torch.int64)

    for batch in loader:
        white = batch["white_features"].to(device)
        black = batch["black_features"].to(device)
        bucket_ids = batch["bucket_id"].to(device)
        stm_white = batch["stm_white"].to(device)
        target = batch["target"].to(device)

        pred = model(white, black, bucket_ids, stm_white)
        total_loss += loss_fn(pred, target).item()
        total_abs += (pred - target).abs().sum().item()
        total_n += target.numel()

        err_sq = (pred - target).pow(2).detach().cpu()
        buckets = bucket_ids.detach().cpu()
        for b in range(NUM_BUCKETS):
            mask = buckets == b
            if mask.any():
                bucket_sq[b] += err_sq[mask].sum().item()
                bucket_n[b] += int(mask.sum())

    per_bucket_mse = {
        f"bucket_{b}_mse": (bucket_sq[b] / bucket_n[b]).item() if bucket_n[b] else 0.0
        for b in range(NUM_BUCKETS)
    }
    return {
        "mse": total_loss / total_n,
        "mae": total_abs / total_n,
        **per_bucket_mse,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    running = 0.0
    n_batches = 0

    for batch in loader:
        white = batch["white_features"].to(device)
        black = batch["black_features"].to(device)
        bucket_ids = batch["bucket_id"].to(device)
        stm_white = batch["stm_white"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(white, black, bucket_ids, stm_white)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        running += loss.item()
        n_batches += 1

    return running / max(n_batches, 1)


def _checkpoint_payload(
    model: nn.Module,
    *,
    architecture: Architecture,
    hidden_dim: int,
    hidden2_dim: int,
    val_mse: float,
    epoch: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "architecture": architecture,
        "hidden_dim": hidden_dim,
        "hidden2_dim": hidden2_dim,
        "val_mse": val_mse,
        "epoch": epoch,
    }
    if architecture == "bucketed" and hasattr(model, "num_buckets"):
        payload["num_buckets"] = int(model.num_buckets)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train SARDINE NNUE (dual_hidden = 2 hidden layers)"
    )
    parser.add_argument("--train", type=Path, default=_DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=_DEFAULT_VAL)
    parser.add_argument(
        "--architecture",
        choices=("single_head", "bucketed", "dual_hidden"),
        default="dual_hidden",
        help="dual_hidden = 2-layer FFNN (default today); single_head = F3; bucketed = legacy",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, choices=[128, 256])
    parser.add_argument(
        "--hidden2-dim",
        type=int,
        default=256,
        help="Second hidden width for dual_hidden (default 256)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=NNUE_CHECKPOINTS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args(argv)

    if not args.train.exists():
        print(f"Train split not found: {args.train}", file=sys.stderr)
        return 1
    if not args.val.exists():
        print(f"Val split not found: {args.val}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    train_ds = ChessbenchDataset(args.train)
    val_ds = ChessbenchDataset(args.val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
    )

    architecture: Architecture = args.architecture
    model = build_nnue(
        architecture,
        hidden_dim=args.hidden_dim,
        hidden2_dim=args.hidden2_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if args.run_name:
        run_name = args.run_name
    elif architecture == "dual_hidden":
        run_name = f"dual_W{args.hidden_dim}_H{args.hidden2_dim}_{stamp}"
    elif architecture == "single_head":
        run_name = f"single_W{args.hidden_dim}_{stamp}"
    else:
        run_name = f"bucketed_W{args.hidden_dim}_{stamp}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "train": str(args.train.resolve()),
        "val": str(args.val.resolve()),
        "architecture": architecture,
        "rows_train": len(train_ds),
        "rows_val": len(val_ds),
        "hidden_dim": args.hidden_dim,
        "hidden2_dim": args.hidden2_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "parameters": model.count_parameters(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Architecture: {architecture}")
    print(f"Train rows: {len(train_ds):,} | Val rows: {len(val_ds):,}")
    print(
        f"Parameters: {model.count_parameters():,} | "
        f"hidden_dim={args.hidden_dim} hidden2_dim={args.hidden2_dim}"
    )
    print(f"Output: {run_dir}")

    history: list[dict[str, float]] = []
    best_val_mse = float("inf")
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - t0

        row = {"epoch": epoch, "train_mse": train_loss, **metrics, "seconds": elapsed}
        history.append(row)
        print(
            f"epoch {epoch:02d} | train_mse={train_loss:.5f} | "
            f"val_mse={metrics['mse']:.5f} | val_mae={metrics['mae']:.5f} | {elapsed:.1f}s"
        )

        if metrics["mse"] < best_val_mse:
            best_val_mse = metrics["mse"]
            torch.save(
                _checkpoint_payload(
                    model,
                    architecture=architecture,
                    hidden_dim=args.hidden_dim,
                    hidden2_dim=args.hidden2_dim,
                    val_mse=metrics["mse"],
                    epoch=epoch,
                ),
                best_path,
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(
        _checkpoint_payload(
            model,
            architecture=architecture,
            hidden_dim=args.hidden_dim,
            hidden2_dim=args.hidden2_dim,
            val_mse=history[-1]["mse"] if history else float("nan"),
            epoch=args.epochs,
        ),
        run_dir / "last.pt",
    )
    print(f"Best val_mse={best_val_mse:.5f} → {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())