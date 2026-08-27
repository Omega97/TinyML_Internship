#!/usr/bin/env python3
"""Train a dense dual-POV FFNN (standard, not sparse) from slice ``features.npz``.

Converts the sparse indices to dense vectors on the fly, concatenates both POVs,
then passes through two hidden layers (ReLU) to 3 logits + softmax.

Default split matches train_nnue.py (test slice 0-5000_d90).

    py -3.12 -u scripts/train_ffnn.py --epochs 10 --smoke
    py -3.12 -u scripts/train_ffnn.py --epochs 50 --lr 0.001 --hidden1 256 --hidden2 256 --run-name ffnn_baseline
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import (
    AcrossSliceBatchSampler,
    FenValueVisitsDataset,
    MixedSliceDataset,
    collate_sparse,
    overlapping_dump_slices,
)

# Import common utilities from train_nnue
import train_nnue as tn

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_TEST_SLICE = "fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90"
PLOTS_DIR = PROJECT_ROOT / "plots"


class StandardFFNN(nn.Module):
    """Dense feed‑forward network: 844×2 → H1 → H2 → 3 logits (WDL)."""

    def __init__(self, hidden1=256, hidden2=256, num_features=844):
        super().__init__()
        self.num_features = num_features
        self.fc1 = nn.Linear(2 * num_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 3)          # 3 logits: W, D, L
        self.relu = nn.ReLU()

    def forward(self, white_idx, white_mask, black_idx, black_mask, stm_white):
        """
        white_idx: (B, K)  padded active feature indices for White POV
        white_mask: (B, K)  0/1 mask
        black_idx: (B, K)
        black_mask: (B, K)
        stm_white: (B,)  1 if White to move, 0 if Black to move
        """
        device = white_idx.device
        B = white_idx.size(0)

        # Build dense vectors (844 dims) for each POV
        white_dense = torch.zeros(B, self.num_features, device=device)
        black_dense = torch.zeros(B, self.num_features, device=device)

        # scatter_add: add 1 at indices where mask == 1
        white_dense.scatter_add_(1, white_idx, white_mask.float())
        black_dense.scatter_add_(1, black_idx, black_mask.float())

        # STM reorder: side-to-move first
        stm = stm_white.view(-1, 1)          # (B, 1)
        first = torch.where(stm == 1, white_dense, black_dense)
        second = torch.where(stm == 1, black_dense, white_dense)
        concat = torch.cat([first, second], dim=1)   # (B, 2*844)

        # Dense layers
        x = self.relu(self.fc1(concat))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)                 # (B, 3)
        return logits

    # Alias per compatibilità con train_nnue.evaluate()
    forward_sparse = forward

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Funzioni di logging locali (per evitare conflitti con train_nnue) ---

def log_epoch_local(
    epoch: int,
    train_ce: float,
    test_ce: float,
    test_mae: float,
    seconds: float,
    train_val_ce: float | None = None,
) -> None:
    msg = f"epoch {epoch:02d} | train_ce={train_ce:.6f}"
    if train_val_ce is not None:
        msg += f" | train_val_ce={train_val_ce:.6f}"
    msg += f" | test_ce={test_ce:.6f} | test_mae={test_mae:.6f} | {seconds:.1f}s"
    print(msg)


# --- Funzione di supporto per holdout ---

def holdout_skip_names(slices_dir: Path, test_name: str) -> set[str]:
    """Exact test folder plus dump slices whose game-index range overlaps."""
    names = [p.name for p in slices_dir.iterdir() if p.is_dir()]
    skip = overlapping_dump_slices(test_name, names)
    extra = sorted(skip - {test_name})
    if extra:
        print(f"  also hold out {len(extra)} overlapping dump-range slice(s):")
        for name in extra:
            print(f"    {name}")
    return skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train standard dense FFNN (2 hidden layers) on WDL labels."
    )
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument(
        "--test",
        "--test-slice",
        dest="test_slice",
        type=str,
        default=DEFAULT_TEST_SLICE,
        help="Test-slice folder name under --slices-dir",
    )
    parser.add_argument("--hidden1", type=int, default=256, help="First hidden layer size")
    parser.add_argument("--hidden2", type=int, default=256, help="Second hidden layer size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--test-subset-size",
        type=int,
        default=0,
        help="Per-epoch test rows (0 = full set). Same subset every epoch (seed 0).",
    )
    parser.add_argument("--test-subset-seed", type=int, default=0)
    parser.add_argument(
        "--train-val-subset-size",
        type=int,
        default=0,
        help="Frozen train-val rows after each epoch (0 = skip).",
    )
    parser.add_argument("--train-val-subset-seed", type=int, default=42)
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
    skip_names = holdout_skip_names(slices_dir, test_name)
    slice_dss = FenValueVisitsDataset.load_slice_datasets(
        slices_dir,
        skip_names=skip_names,
        rebuild=args.rebuild_cache,
        progress=True,
    )
    train_ds = MixedSliceDataset(slice_dss)
    test_ds_full = FenValueVisitsDataset(
        test_folder,
        rebuild_cache=args.rebuild_cache,
        progress=True,
    )
    n_test_full = len(test_ds_full)
    test_ds = tn.maybe_subset_dataset(test_ds_full, args.test_subset_size, args.test_subset_seed)
    n_test_eval = len(test_ds)
    pool = len(train_ds)
    if batches_per_epoch > 0:
        n_batches = batches_per_epoch
    elif max_train > 0:
        n_batches = max(1, max_train // batch_size)
    else:
        n_batches = max(1, pool // batch_size)
    print(
        f"  train pool {pool:,} in {len(slice_dss)} slices | "
        f"test {n_test_eval:,}/{n_test_full:,} | "
        f"batch {batch_size} mixed across slices × {n_batches} steps/epoch"
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
    train_val_ds = tn.maybe_subset_train_pool(
        train_ds, args.train_val_subset_size, args.train_val_subset_seed
    )
    train_val_loader = None
    if train_val_ds is not None:
        train_val_loader = DataLoader(
            train_val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_sparse,
            pin_memory=pin,
        )
        print(
            f"  train-val subset {len(train_val_ds):,} of {pool:,} "
            f"(seed={args.train_val_subset_seed}, frozen eval)"
        )

    model = StandardFFNN(hidden1=args.hidden1, hidden2=args.hidden2).to(device)
    if args.do_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or (
        f"ffnn_h{args.hidden1}_{args.hidden2}"
        f"{'_smoke' if args.smoke else ''}{'_fast' if args.fast else ''}_{stamp}"
    )
    run_dir = tn._resolve(args.output_dir) / run_name
    tn.warn_if_run_dir_busy(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "slices_dir": str(slices_dir),
        "test_slice": test_name,
        "rows_train_pool": pool,
        "n_train_slices": len(slice_dss),
        "rows_test": n_test_eval,
        "rows_test_full": n_test_full,
        "test_subset_size": int(args.test_subset_size),
        "test_subset_seed": int(args.test_subset_seed),
        "train_val_subset_size": int(args.train_val_subset_size),
        "train_val_subset_seed": int(args.train_val_subset_seed),
        "architecture": "standard FFNN (dense, 2 hidden layers)",
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
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
        f"FFNN | dense 844×2 → {args.hidden1} → {args.hidden2} → 3 WDL softmax | "
        f"{model.count_parameters():,} params | loss=soft CE | "
        f"train steps {n_batches}×{batch_size} | "
        f"test {n_test_eval:,}/{n_test_full:,} | {device}"
    )
    print(f"Output: {run_dir}")

    history: list[dict] = []
    best_test = float("inf")
    best_path = run_dir / "best.pt"
    best_state: dict[str, torch.Tensor] | None = None
    # Eval untrained: usa la funzione di train_nnue (richiede forward_sparse)
    history.append(
        tn.eval_untrained(
            model, train_loader, test_loader, device, train_val_loader=train_val_loader
        )
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_ce = tn.train_epoch(model, train_loader, optimizer, device)
        metrics = tn.evaluate(model, test_loader, device)
        train_val_ce = None
        if train_val_loader is not None:
            train_val_ce = tn.evaluate(model, train_val_loader, device)["ce"]
        elapsed = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_ce": train_ce,
            "train_val_ce": train_val_ce,
            "test_ce": metrics["ce"],
            "test_mae": metrics["mae"],
            "seconds": elapsed,
        }
        history.append(row)

        # Logging locale (per evitare conflitti di firma)
        log_epoch_local(
            epoch,
            train_ce,
            metrics["ce"],
            metrics["mae"],
            elapsed,
            train_val_ce=train_val_ce,
        )

        payload = {
            "model_state_dict": model.state_dict(),
            "architecture": "ffnn",
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
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
        title=f"Standard FFNN ({args.hidden1} → {args.hidden2} → 3 WDL softmax)",
    )
    (run_dir / "ce.png").write_bytes(plot_path.read_bytes())
    print(f"Best test_ce={best_test:.6f} → {best_path}")
    print(f"CE plot → {plot_path}")
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
