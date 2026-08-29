#!/usr/bin/env python3
"""Train dual-POV DualHidden NNUE from per-slice ``features.npz`` DBs.

Default split: a random ``--test-fraction`` (0.10) of each slice is test
(seed ``--split-seed``). Train batches and both CE metrics are row-weighted.

No chess encoding at train time. Loss is unweighted soft cross-entropy on
STM WDL (visits omitted). Re-encode slices after the WDL schema change
(``--rebuild-cache`` or ``encode_slice_features.py --rebuild``).

    py -3.12 -u scripts/train_nnue.py --epochs 5 --smoke
    py -3.12 -u scripts/train_nnue.py --epochs 5 --fast
    py -3.12 -u scripts/train_nnue.py --epochs 10
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import (
    ConcatSliceDataset,
    FenValueVisitsDataset,
    MixedSliceDataset,
    UniformRowBatchSampler,
    collate_sparse,
    split_last_fraction,
    split_random_fraction,
)
from tinymlinternship.nnue.model import DualHiddenNNUE

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_TEST_FRACTION = 0.10
PLOTS_DIR = PROJECT_ROOT / "plots"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def add_test_fraction_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        metavar="P",
        help="Random this fraction of each slice is test; the rest is train "
        f"(default: {DEFAULT_TEST_FRACTION})",
    )


def add_split_seed_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="RNG seed for the random train/test split (default: 0)",
    )


def frozen_train_eval_n(args: argparse.Namespace) -> int:
    """Row-weighted train CE subset; matches --test-subset-size when unset."""
    n = int(args.train_val_subset_size)
    if n <= 0:
        n = int(args.test_subset_size)
    if n <= 0:
        n = 3200
    return n


def resolve_train_eval_seed(args: argparse.Namespace) -> int:
    """Same seed as test eval unless ``--train-val-subset-seed`` is set."""
    seed = getattr(args, "train_val_subset_seed", None)
    if seed is None:
        return int(args.test_subset_seed)
    return int(seed)


def report_matched_train_test_ce(
    model: torch.nn.Module,
    train_ds,
    test_ds_full,
    *,
    pool: int,
    n_test_full: int,
    batch_size: int,
    device: torch.device,
    workers: int,
    pin: bool,
    best_state: dict[str, torch.Tensor] | None,
) -> None:
    """Print train vs test CE on the same n (up to 50k), seed 0, best weights."""
    compare_n = min(50_000, pool, n_test_full)
    train_cmp = maybe_subset_dataset(train_ds, compare_n, 0)
    test_cmp = maybe_subset_dataset(test_ds_full, compare_n, 0)
    kw = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_sparse,
        pin_memory=pin,
    )
    train_loader = DataLoader(train_cmp, **kw)
    test_loader = DataLoader(test_cmp, **kw)
    raw = getattr(model, "_orig_mod", model)
    if best_state is not None:
        raw.load_state_dict(
            {key.replace("_orig_mod.", ""): tensor for key, tensor in best_state.items()}
        )
    train_m = evaluate(model, train_loader, device)
    test_m = evaluate(model, test_loader, device)
    print(
        f"matched {compare_n:,}-row CE (best weights, seed=0) | "
        f"train_ce={train_m['ce']:.6f} | test_ce={test_m['ce']:.6f} | "
        f"gap={train_m['ce'] - test_m['ce']:+.6f}"
    )


def prepare_fraction_split(
    slices_dir: Path,
    test_fraction: float,
    *,
    rebuild: bool = False,
    progress: bool = True,
) -> tuple[MixedSliceDataset, ConcatSliceDataset]:
    """Load every slice, then last ``test_fraction`` of each → test."""
    frac = float(test_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"--test-fraction must be in (0, 1), got {test_fraction}")
    slices = FenValueVisitsDataset.load_slice_datasets(
        slices_dir,
        rebuild=rebuild,
        progress=progress,
    )
    if not slices:
        raise FileNotFoundError(f"no slice JSON folders in {slices_dir}")
    train_parts, test_parts = split_last_fraction(slices, frac)
    if not train_parts:
        raise ValueError("train split is empty; lower --test-fraction")
    if not test_parts:
        raise ValueError("test split is empty; raise --test-fraction or use larger slices")
    return MixedSliceDataset(train_parts), ConcatSliceDataset(test_parts)


def prepare_random_split(
    slices_dir: Path,
    test_fraction: float,
    *,
    seed: int = 0,
    rebuild: bool = False,
    progress: bool = True,
) -> tuple[ConcatSliceDataset, ConcatSliceDataset]:
    """Load every slice, then a random ``test_fraction`` of each → test."""
    frac = float(test_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"--test-fraction must be in (0, 1), got {test_fraction}")
    slices = FenValueVisitsDataset.load_slice_datasets(
        slices_dir,
        rebuild=rebuild,
        progress=progress,
    )
    if not slices:
        raise FileNotFoundError(f"no slice JSON folders in {slices_dir}")
    train_parts, test_parts = split_random_fraction(slices, frac, seed=int(seed))
    if not train_parts:
        raise ValueError("train split is empty; lower --test-fraction")
    if not test_parts:
        raise ValueError("test split is empty; raise --test-fraction or use larger slices")
    return ConcatSliceDataset(train_parts), ConcatSliceDataset(test_parts)


def resolve_plot_path(plot: Path | None, run_name: str, plots_dir: Path) -> Path:
    """``--plot path.png`` is used as-is. Omit it → ``plots/{run_name}_ce.png``."""
    if plot is None:
        return plots_dir / f"{run_name}_ce.png"
    path = _resolve(plot)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        return path
    return path / f"{run_name}_ce.png"


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


def log_epoch(epoch: int, train_ce: float, test_ce: float, seconds: float) -> None:
    print(
        f"epoch {epoch:02d} | train_ce={train_ce:.6f} | test_ce={test_ce:.6f} | {seconds:.1f}s"
    )


def make_linear_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    lr: float,
    lr_end: float | None,
    epochs: int,
) -> object | None:
    """Linear decay from ``lr`` to ``lr_end`` over ``epochs``. ``lr_end is None`` → constant."""
    if lr_end is None or epochs < 1 or lr <= 0.0 or lr_end <= 0.0:
        return None
    if abs(lr_end - lr) / lr < 1e-12:
        return None
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=float(lr_end) / float(lr),
        total_iters=int(epochs),
    )


def eval_untrained(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    train_val_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Epoch 00: CE on random weights (no optimizer step)."""
    t0 = time.perf_counter()
    train_m = evaluate(
        model,
        train_val_loader if train_val_loader is not None else train_loader,
        device,
    )
    test_m = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - t0
    row = {
        "epoch": 0,
        "train_ce": train_m["ce"],
        "test_ce": test_m["ce"],
        "seconds": elapsed,
    }
    log_epoch(0, row["train_ce"], row["test_ce"], elapsed)
    return row


def packed_subset_indices(sizes: list[int], take: int, seed: int) -> list[int]:
    """Linear 0..N-1 → MixedSliceDataset packed ``slice << 32 | row``."""
    n = int(sum(sizes))
    take = min(int(take), n)
    if take <= 0:
        return []
    g = torch.Generator()
    g.manual_seed(int(seed))
    linear = torch.randperm(n, generator=g)[:take].tolist()
    offsets: list[int] = []
    acc = 0
    for size in sizes:
        offsets.append(acc)
        acc += int(size)
    packed: list[int] = []
    for i in linear:
        sid = bisect.bisect_right(offsets, i) - 1
        row = i - offsets[sid]
        packed.append((sid << 32) | row)
    return packed


def maybe_subset_dataset(dataset, size: int, seed: int = 0):
    """Fixed random subset for per-epoch eval. ``size <= 0`` keeps the full set."""
    n = len(dataset)
    take = int(size)
    if take <= 0 or take >= n:
        return dataset
    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(n, generator=g)[:take].tolist()
    return Subset(dataset, idx)


def maybe_subset_train_pool(dataset, size: int, seed: int = 42):
    """Train-val subset. MixedSliceDataset needs packed indices, not 0..N-1.

    ``size <= 0`` → ``None`` (skip frozen train-val eval).
    """
    take = int(size)
    if take <= 0:
        return None
    if isinstance(dataset, MixedSliceDataset):
        n = len(dataset)
        idx = packed_subset_indices(dataset.sizes, min(take, n), seed)
        return Subset(dataset, idx)
    return maybe_subset_dataset(dataset, take, seed)


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def warn_if_run_dir_busy(run_dir: Path) -> None:
    if (run_dir / "best.pt").is_file() or (run_dir / "last.pt").is_file():
        print(
            f"warning: {run_dir} already has checkpoints; "
            "parallel jobs with the same --run-name will overwrite best.pt / last.pt / plots",
            file=sys.stderr,
        )


def report_full_test(
    model: torch.nn.Module,
    best_path: Path,
    full_loader: DataLoader,
    device: torch.device,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Score the full test set with this run's best weights (in-memory, else ``best.pt``)."""
    raw = getattr(model, "_orig_mod", model)
    if state_dict is not None:
        state = {key.replace("_orig_mod.", ""): tensor for key, tensor in state_dict.items()}
        label = "in-memory best"
    else:
        payload = torch.load(best_path, map_location=device, weights_only=False)
        state = payload["model_state_dict"]
        state = {key.replace("_orig_mod.", ""): tensor for key, tensor in state.items()}
        label = "best.pt"
    try:
        raw.load_state_dict(state)
    except RuntimeError as exc:
        print(
            "full test skipped: checkpoint width does not match this model "
            "(another job likely reused --run-name and overwrote best.pt).",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return {}
    metrics = evaluate(model, full_loader, device)
    print(f"full test ({label}) | test_ce={metrics['ce']:.6f} | n={metrics['n']:,}")
    return metrics


def plot_ce(
    history: list[dict[str, Any]],
    path: Path,
    *,
    title: str = "Dual-POV NNUE WDL (L1 64×2, L2 128)",
) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_ce = [row["train_ce"] for row in history]
    test_ce = [row["test_ce"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, train_ce, marker="o", label="train CE")
    ax.plot(epochs, test_ce, marker="s", label="test CE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title(title)
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
    add_test_fraction_arg(parser)
    add_split_seed_arg(parser)
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
    parser.add_argument("--max-train", type=int, default=0, help="0 = all train rows")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Cap train at 20k mixed samples/epoch unless --max-train is set",
    )
    parser.add_argument(
        "--test-subset-size",
        type=int,
        default=0,
        help="Per-epoch test rows (0 = full set). Same subset every epoch (seed 0). "
        "If set, best.pt is scored on the full test set once at the end.",
    )
    parser.add_argument(
        "--test-subset-seed",
        type=int,
        default=0,
        help="RNG seed for --test-subset-size",
    )
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
        train_ds, test_ds_full = prepare_random_split(
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
    test_ds = maybe_subset_dataset(
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
    train_eval_n = frozen_train_eval_n(args)
    train_eval_seed = resolve_train_eval_seed(args)
    train_val_ds = maybe_subset_dataset(
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

    model = DualHiddenNNUE(
        hidden_dim=args.hidden_dim,
        hidden2_dim=args.hidden2_dim,
    ).to(device)
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
    scheduler = make_linear_lr_scheduler(
        optimizer, lr=args.lr, lr_end=args.lr_end, epochs=args.epochs
    )

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or (
        f"dual_W{args.hidden_dim}_H{args.hidden2_dim}"
        f"{'_smoke' if args.smoke else ''}{'_fast' if args.fast else ''}_{stamp}"
    )
    run_dir = _resolve(args.output_dir) / run_name
    warn_if_run_dir_busy(run_dir)
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
        "hidden_dim": args.hidden_dim,
        "hidden2_dim": args.hidden2_dim,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "batches_per_epoch": n_batches,
        "sample": "each batch: batch_size i.i.d. draws uniform over training rows",
        "lr": args.lr,
        "lr_end": args.lr if args.lr_end is None else args.lr_end,
        "lr_schedule": "linear" if scheduler is not None else "constant",
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
        f"train steps {n_batches}×{batch_size} | "
        f"test {n_test_eval:,}/{n_test_full:,} | {device}"
    )
    print(f"Output: {run_dir}")

    history: list[dict[str, Any]] = []
    best_test = float("inf")
    best_path = run_dir / "best.pt"
    best_state: dict[str, torch.Tensor] | None = None
    history.append(
        eval_untrained(
            model, train_loader, test_loader, device, train_val_loader=train_val_loader
        )
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, test_loader, device)
        train_ce = evaluate(model, train_val_loader, device)["ce"]
        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_ce": train_ce,
            "test_ce": metrics["ce"],
            "seconds": elapsed,
        }
        history.append(row)
        log_epoch(epoch, train_ce, metrics["ce"], elapsed)
        if scheduler is not None:
            scheduler.step()
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
            best_state = clone_state(model)
            torch.save(payload, best_path)
        torch.save(payload, run_dir / "last.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_path = resolve_plot_path(args.plot, run_name, PLOTS_DIR)
    plot_ce(history, plot_path)
    (run_dir / "ce.png").write_bytes(plot_path.read_bytes())
    print(f"Best test_ce={best_test:.6f} → {best_path}")
    print(f"CE plot → {plot_path}")
    if best_path.is_file() or best_state is not None:
        report_matched_train_test_ce(
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
        report_full_test(model, best_path, full_loader, device, state_dict=best_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
