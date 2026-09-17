#!/usr/bin/env python3
"""Train dual-POV DualHidden NNUE from per-slice ``features.npz`` DBs.

Optimized for NVIDIA Tensor Core architectures (Spark / GH200 / RTX).

Default split: a random ``--test-fraction`` (0.10) of each slice is test
(seed ``--split-seed``). Train batches and both CE metrics are row-weighted.
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
    discover_slice_folders,
    slice_source_json,
    split_last_fraction,
    split_random_fraction,
)
from tinymlinternship.nnue.model import DualHiddenNNUE

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_TEST_FRACTION = 0.10
PLOTS_DIR = PROJECT_ROOT / "plots"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_slices_dir(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    default_slices: Path = DEFAULT_SLICES,
) -> Path:
    """Resolve ``--slices-dir``.

    Absolute paths are kept. Relative paths are tried under ``project_root``,
    then as a slice folder name under ``default_slices``
    (``data/processed/board_eval/fen_value_visits/``).
    """
    if path.is_absolute():
        return path
    at_root = (project_root / path).resolve()
    under_default = (default_slices / path).resolve()
    if at_root.is_dir():
        return at_root
    if under_default.is_dir():
        return under_default
    if len(path.parts) == 1:
        return under_default
    return at_root


def slice_folders(slices_dir: Path) -> list[Path]:
    """Child slice folders, or ``[slices_dir]`` if it is itself a slice."""
    folders = discover_slice_folders(slices_dir)
    if folders:
        return folders
    if slice_source_json(slices_dir) is not None:
        return [slices_dir]
    return []


def create_dataloader(
    dataset,
    *,
    batch_size: int = 2048,
    batch_sampler=None,
    shuffle: bool = False,
    workers: int = 0,
    pin: bool = True,
) -> DataLoader:
    kwargs: dict[str, Any] = dict(
        num_workers=workers,
        collate_fn=collate_sparse,
        pin_memory=pin,
    )
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs["batch_size"] = batch_size
        kwargs["shuffle"] = shuffle

    return DataLoader(dataset, **kwargs)


def ce_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=-1)
    nll = -(target * log_p).sum(dim=-1)
    w = weight.clamp(min=0.0)
    denom = w.sum().clamp(min=1e-8)
    return (nll * w).sum() / denom


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, float]:
    model.eval()
    ce_sum = 0.0
    mae_sum = 0.0
    w_sum = 0.0
    n = 0
    use_amp = (device.type == "cuda" and amp_dtype is not None)

    for batch in loader:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model.forward_sparse(
                batch["white_idx"].to(device, non_blocking=True),
                batch["white_mask"].to(device, non_blocking=True),
                batch["black_idx"].to(device, non_blocking=True),
                batch["black_mask"].to(device, non_blocking=True),
                batch["stm_white"].to(device, non_blocking=True),
            )
        target = batch["target"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        w = weight.clamp(min=0.0)

        logits_f32 = logits.float()
        log_p = F.log_softmax(logits_f32, dim=-1)
        nll = -(target * log_p).sum(dim=-1)
        probs = F.softmax(logits_f32, dim=-1)
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
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> float:
    model.train()
    running = 0.0
    n_batches = 0
    use_amp = (device.type == "cuda" and amp_dtype is not None)

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model.forward_sparse(
                batch["white_idx"].to(device, non_blocking=True),
                batch["white_mask"].to(device, non_blocking=True),
                batch["black_idx"].to(device, non_blocking=True),
                batch["black_mask"].to(device, non_blocking=True),
                batch["stm_white"].to(device, non_blocking=True),
            )
            loss = ce_loss(
                logits,
                batch["target"].to(device, non_blocking=True),
                batch["weight"].to(device, non_blocking=True),
            )

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += float(loss.item())
        n_batches += 1

    return running / max(n_batches, 1)


def eval_untrained(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    train_val_loader: DataLoader | None = None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    train_m = evaluate(
        model,
        train_val_loader if train_val_loader is not None else train_loader,
        device,
        amp_dtype=amp_dtype,
    )
    test_m = evaluate(model, test_loader, device, amp_dtype=amp_dtype)
    elapsed = time.perf_counter() - t0
    row = {
        "epoch": 0,
        "train_ce": train_m["ce"],
        "test_ce": test_m["ce"],
        "seconds": elapsed,
    }
    log_epoch(0, row["train_ce"], row["test_ce"], elapsed)
    return row


def prepare_random_split(
    slices_dir: Path,
    test_fraction: float,
    *,
    seed: int = 0,
    rebuild: bool = False,
    progress: bool = True,
) -> tuple[ConcatSliceDataset, ConcatSliceDataset]:
    frac = float(test_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"--test-fraction must be in (0, 1), got {test_fraction}")
    folders = slice_folders(slices_dir)
    if not folders:
        raise FileNotFoundError(f"no slice JSON folders in {slices_dir}")
    slices = [
        FenValueVisitsDataset(
            folder,
            rebuild_cache=rebuild,
            progress=progress,
        )
        for folder in folders
    ]
    train_parts, test_parts = split_random_fraction(slices, frac, seed=int(seed))
    if not train_parts:
        raise ValueError("train split is empty; lower --test-fraction")
    if not test_parts:
        raise ValueError("test split is empty; raise --test-fraction or use larger slices")
    return ConcatSliceDataset(train_parts), ConcatSliceDataset(test_parts)


def resolve_plot_path(plot: Path | None, run_name: str, plots_dir: Path) -> Path:
    if plot is None:
        return plots_dir / f"{run_name}_ce.png"
    path = _resolve(plot)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        return path
    return path / f"{run_name}_ce.png"


def log_epoch(epoch: int, train_ce: float, test_ce: float, seconds: float) -> None:
    print(f"epoch {epoch:02d} | train_ce={train_ce:.6f} | test_ce={test_ce:.6f} | {seconds:.1f}s")


def make_linear_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    lr: float,
    lr_end: float | None,
    epochs: int,
) -> object | None:
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


def maybe_subset_dataset(dataset, size: int, seed: int = 0):
    n = len(dataset)
    take = int(size)
    if take <= 0 or take >= n:
        return dataset
    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(n, generator=g)[:take].tolist()
    return Subset(dataset, idx)


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def plot_ce(
    history: list[dict[str, Any]],
    path: Path,
    *,
    title: str = "Dual-POV NNUE WDL",
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
        description="Optimized DualHidden NNUE training pipeline"
    )
    parser.add_argument(
        "--slices-dir",
        type=Path,
        default=DEFAULT_SLICES,
        help=(
            "Parent of per-slice folders, or one slice folder / name under "
            f"{DEFAULT_SLICES.name}/ (default: {DEFAULT_SLICES})"
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden2-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-end", type=float, default=None)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--test-subset-size", type=int, default=0)
    parser.add_argument("--test-subset-seed", type=int, default=0)
    parser.add_argument("--train-val-subset-size", type=int, default=0)
    parser.add_argument("--train-val-subset-seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=NNUE_CHECKPOINTS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="do_compile")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="AMP precision (default: bfloat16)",
    )
    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    slices_dir = resolve_slices_dir(args.slices_dir)
    batch_size = 256 if (args.fast and args.batch_size == 2048) else args.batch_size
    batches_per_epoch = 40 if (args.fast and args.batches_per_epoch == 0) else args.batches_per_epoch
    max_train = 20_000 if (args.smoke and args.max_train == 0) else args.max_train

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    amp_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": None,
    }
    amp_dtype = amp_dtype_map[args.amp_dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    print(f"Slices: {slices_dir}")
    try:
        train_ds, test_ds_full = prepare_random_split(
            slices_dir, args.test_fraction, seed=args.split_seed, rebuild=args.rebuild_cache
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.encode_only:
        print("encode-only: slice DBs ready")
        return 0

    n_test_full = len(test_ds_full)
    test_ds = maybe_subset_dataset(test_ds_full, args.test_subset_size, args.test_subset_seed)
    n_test_eval = len(test_ds)
    pool = len(train_ds)

    if batches_per_epoch > 0:
        n_batches = batches_per_epoch
    elif max_train > 0:
        n_batches = max(1, max_train // batch_size)
    else:
        n_batches = max(1, pool // batch_size)

    pin = device.type == "cuda"
    train_loader = create_dataloader(
        train_ds,
        batch_sampler=UniformRowBatchSampler(pool, batch_size=batch_size, batches=n_batches),
        workers=args.workers,
        pin=pin,
    )
    test_loader = create_dataloader(
        test_ds, batch_size=batch_size, shuffle=False, workers=args.workers, pin=pin
    )

    train_eval_n = args.train_val_subset_size or args.test_subset_size or 3200
    train_eval_seed = args.train_val_subset_seed if args.train_val_subset_seed is not None else args.test_subset_seed
    train_val_ds = maybe_subset_dataset(train_ds, train_eval_n, train_eval_seed)
    train_val_loader = create_dataloader(
        train_val_ds, batch_size=batch_size, shuffle=False, workers=args.workers, pin=pin
    )

    model = DualHiddenNNUE(hidden_dim=args.hidden_dim, hidden2_dim=args.hidden2_dim).to(device)

    if args.do_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model, mode="reduce-overhead")

    fused_available = device.type == "cuda" and hasattr(torch.optim.Adam, "supports_fused")
    optimizer_kwargs = {"lr": args.lr}
    if fused_available:
        optimizer_kwargs["fused"] = True

    try:
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = make_linear_lr_scheduler(optimizer, lr=args.lr, lr_end=args.lr_end, epochs=args.epochs)

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or f"dual_W{args.hidden_dim}_H{args.hidden2_dim}_{stamp}"
    run_dir = _resolve(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "slices_dir": str(slices_dir),
        "test_fraction": float(args.test_fraction),
        "hidden_dim": args.hidden_dim,
        "hidden2_dim": args.hidden2_dim,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "batches_per_epoch": n_batches,
        "lr": args.lr,
        "lr_end": args.lr if args.lr_end is None else args.lr_end,
        "amp_dtype": args.amp_dtype,
        "device": str(device),
        "parameters": model.count_parameters(),
        "compiled": bool(args.do_compile),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Training on {device} | AMP: {args.amp_dtype} | TF32: On | Fused Adam: {fused_available}")

    history: list[dict[str, Any]] = []
    best_test = float("inf")
    best_path = run_dir / "best.pt"

    history.append(
        eval_untrained(
            model, train_loader, test_loader, device, train_val_loader=train_val_loader, amp_dtype=amp_dtype
        )
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_epoch(model, train_loader, optimizer, device, scaler=scaler, amp_dtype=amp_dtype)

        metrics = evaluate(model, test_loader, device, amp_dtype=amp_dtype)
        train_ce = evaluate(model, train_val_loader, device, amp_dtype=amp_dtype)["ce"]
        elapsed = time.perf_counter() - t0

        row = {"epoch": epoch, "train_ce": train_ce, "test_ce": metrics["ce"], "seconds": elapsed}
        history.append(row)
        log_epoch(epoch, train_ce, metrics["ce"], elapsed)

        if scheduler is not None:
            scheduler.step()

        payload = {
            "model_state_dict": model.state_dict(),
            "architecture": "dual_hidden_wdl",
            "hidden_dim": args.hidden_dim,
            "hidden2_dim": args.hidden2_dim,
            "test_ce": metrics["ce"],
            "epoch": epoch,
        }
        if metrics["ce"] < best_test:
            best_test = metrics["ce"]
            torch.save(payload, best_path)
        torch.save(payload, run_dir / "last.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_path = resolve_plot_path(args.plot, run_name, PLOTS_DIR)
    plot_ce(history, plot_path)
    (run_dir / "ce.png").write_bytes(plot_path.read_bytes())
    print(f"Best test_ce={best_test:.6f} saved to {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())