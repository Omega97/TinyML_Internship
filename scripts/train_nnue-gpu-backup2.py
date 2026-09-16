#!/usr/bin/env python3
"""Train dual-POV DualHidden NNUE from per-slice ``features.npz`` DBs.

Optimized for NVIDIA Tensor Core architectures (Spark / GH200 / RTX).

The sparse tables are concatenated once and moved onto the device. Training
then samples with ``torch.randint`` + gather instead of a Python DataLoader
(``__getitem__`` + ``collate_sparse`` per row), which is what left the GPU
idle between batches.

Default split: a random ``--test-fraction`` (0.10) of each slice is test
(seed ``--split-seed``). Train batches and both CE metrics are row-weighted.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import (
    ConcatSliceDataset,
    FenValueVisitsDataset,
    PackedSparseTensors,
    discover_slice_folders,
    slice_source_json,
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


def configure_torch(device: torch.device) -> None:
    threads = os.cpu_count() or 1
    torch.set_num_threads(int(threads))
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def move_pack(pack: PackedSparseTensors, device: torch.device) -> PackedSparseTensors:
    if pack.device == device:
        return pack
    if device.type == "cuda" and pack.device.type == "cpu":
        pack = pack.pin_memory()
        pack = pack.to(device, non_blocking=True)
        torch.cuda.synchronize()
        return pack
    return pack.to(device)


def batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample = batch["target"]
    if sample.device == device:
        return batch
    return {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}


def sparse_logits(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Call compiled ``forward`` (sparse) rather than ``forward_sparse``."""
    return model(
        batch["white_idx"],
        batch["black_idx"],
        batch["stm_white"],
        batch["white_mask"],
        batch["black_mask"],
    )


def ce_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=-1)
    nll = -(target * log_p).sum(dim=-1)
    w = weight.clamp(min=0.0)
    denom = w.sum().clamp(min=1e-8)
    return (nll * w).sum() / denom


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    pack: PackedSparseTensors,
    device: torch.device,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    batch_size: int = 2048,
) -> dict[str, float]:
    model.eval()
    use_amp = device.type == "cuda" and amp_dtype is not None
    ce_sum = torch.zeros((), device=device, dtype=torch.float32)
    mae_sum = torch.zeros((), device=device, dtype=torch.float32)
    w_sum = torch.zeros((), device=device, dtype=torch.float32)

    for batch in pack.iter_batches(batch_size, pad=True):
        batch = batch_to_device(batch, device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = sparse_logits(model, batch)
        target = batch["target"]
        weight = batch["weight"]
        w = weight.clamp(min=0.0)

        logits_f32 = logits.float()
        log_p = F.log_softmax(logits_f32, dim=-1)
        nll = -(target * log_p).sum(dim=-1)
        probs = F.softmax(logits_f32, dim=-1)
        pred_v = probs[:, 0] - probs[:, 2]
        tgt_v = target[:, 0] - target[:, 2]

        ce_sum = ce_sum + (nll * w).sum()
        mae_sum = mae_sum + ((pred_v - tgt_v).abs() * w).sum()
        w_sum = w_sum + w.sum()

    denom = w_sum.clamp(min=1e-8)
    return {
        "ce": float((ce_sum / denom).item()),
        "mae": float((mae_sum / denom).item()),
        "n": len(pack),
    }


def train_epoch(
    model: torch.nn.Module,
    pack: PackedSparseTensors,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    batch_size: int,
    n_batches: int,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> float:
    model.train()
    use_amp = device.type == "cuda" and amp_dtype is not None
    n = len(pack)
    use_scaler = scaler is not None and scaler.is_enabled()
    loss_acc = torch.zeros((), device=device, dtype=torch.float32)

    for _ in range(int(n_batches)):
        idx = torch.randint(n, (batch_size,), device=pack.device, generator=generator)
        batch = batch_to_device(pack.gather(idx), device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = sparse_logits(model, batch)
            loss = ce_loss(logits, batch["target"], batch["weight"])

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_acc = loss_acc + loss.detach()

    return float(loss_acc.item()) / max(int(n_batches), 1)


def eval_untrained(
    model: torch.nn.Module,
    train_pack: PackedSparseTensors,
    test_pack: PackedSparseTensors,
    device: torch.device,
    train_val_pack: PackedSparseTensors | None = None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    batch_size: int = 2048,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    train_m = evaluate(
        model,
        train_val_pack if train_val_pack is not None else train_pack,
        device,
        amp_dtype=amp_dtype,
        batch_size=batch_size,
    )
    test_m = evaluate(
        model, test_pack, device, amp_dtype=amp_dtype, batch_size=batch_size
    )
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


def log_epoch(
    epoch: int,
    train_ce: float,
    test_ce: float,
    seconds: float,
    *,
    extras: str = "",
) -> None:
    extra = f" | {extras}" if extras else ""
    print(f"epoch {epoch:02d} | train_ce={train_ce:.6f} | test_ce={test_ce:.6f} | {seconds:.1f}s{extra}")


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


def make_optimizer(model: torch.nn.Module, lr: float, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": lr}
    if device.type == "cuda":
        kwargs["fused"] = True
        kwargs["capturable"] = True
    try:
        return torch.optim.Adam(model.parameters(), **kwargs)
    except TypeError:
        return torch.optim.Adam(model.parameters(), lr=lr)


def _silence_stderr():
    """Hide gcc/Triton chatter when inductor is missing headers."""

    class _Guard:
        def __enter__(self):
            self._null = open(os.devnull, "w")
            self._old = os.dup(2)
            os.dup2(self._null.fileno(), 2)
            return self

        def __exit__(self, *args: object) -> None:
            os.dup2(self._old, 2)
            os.close(self._old)
            self._null.close()

    return _Guard()


def try_compile(
    model: torch.nn.Module,
    pack: PackedSparseTensors,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None,
    batch_size: int,
    mode: str,
) -> tuple[torch.nn.Module, bool]:
    """Compile and run one train-shaped step. Fall back to eager on inductor/Triton errors."""
    dummy = batch_to_device(next(iter(pack.iter_batches(batch_size, pad=True))), device)
    use_amp = device.type == "cuda" and amp_dtype is not None
    compiled = torch.compile(model, mode=mode, dynamic=False)
    try:
        with _silence_stderr():
            compiled.train()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = sparse_logits(compiled, dummy)
                loss = ce_loss(logits, dummy["target"], dummy["weight"])
            loss.backward()
            compiled.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
        print("torch.compile ready")
        return compiled, True
    except Exception as exc:
        model.zero_grad(set_to_none=True)
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        print(
            f"torch.compile failed ({type(exc).__name__}); continuing eager. "
            "Install python3.12-dev if you want inductor/CUDA graphs."
        )
        return model, False


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
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Eval batch size (default: max(train batch, 32768) on CUDA)",
    )
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-end", type=float, default=None)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--test-subset-size", type=int, default=0)
    parser.add_argument("--test-subset-seed", type=int, default=0)
    parser.add_argument("--train-val-subset-size", type=int, default=0)
    parser.add_argument("--train-val-subset-seed", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Ignored: batches are gathered from packed device tensors",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--cpu-data",
        action="store_true",
        help="Keep packed tensors on CPU (still gathered, no DataLoader)",
    )
    parser.add_argument("--output-dir", type=Path, default=NNUE_CHECKPOINTS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="do_compile")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: reduce-overhead / CUDA graphs)",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="AMP precision (default: bfloat16)",
    )
    args = parser.parse_args(argv)

    slices_dir = resolve_slices_dir(args.slices_dir)
    batch_size = 256 if (args.fast and args.batch_size == 2048) else args.batch_size
    batches_per_epoch = 40 if (args.fast and args.batches_per_epoch == 0) else args.batches_per_epoch
    max_train = 20_000 if (args.smoke and args.max_train == 0) else args.max_train

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    configure_torch(device)

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

    train_eval_n = args.train_val_subset_size or args.test_subset_size or 3200
    train_eval_seed = args.train_val_subset_seed if args.train_val_subset_seed is not None else args.test_subset_seed
    train_val_ds = maybe_subset_dataset(train_ds, train_eval_n, train_eval_seed)

    print(
        f"Packing {pool:,} train + {n_test_eval:,} test rows "
        f"(full test {n_test_full:,}) ..."
    )
    train_pack = PackedSparseTensors.from_dataset(train_ds)
    test_pack = PackedSparseTensors.from_dataset(test_ds)
    train_val_pack = PackedSparseTensors.from_dataset(train_val_ds)
    del train_ds, test_ds, test_ds_full, train_val_ds
    gc.collect()

    data_device = device if (device.type == "cuda" and not args.cpu_data) else torch.device("cpu")
    train_pack = move_pack(train_pack, data_device)
    test_pack = move_pack(test_pack, data_device)
    train_val_pack = move_pack(train_val_pack, data_device)

    if args.eval_batch_size > 0:
        eval_batch_size = int(args.eval_batch_size)
    elif device.type == "cuda":
        eval_batch_size = max(int(batch_size), 32768)
    else:
        eval_batch_size = int(batch_size)

    model = DualHiddenNNUE(hidden_dim=args.hidden_dim, hidden2_dim=args.hidden2_dim).to(device)
    n_params = model.count_parameters()

    compiled_ok = False
    if args.do_compile and hasattr(torch, "compile"):
        print(f"Compiling model with torch.compile({args.compile_mode!r}) ...")
        model, compiled_ok = try_compile(
            model,
            train_pack,
            device,
            amp_dtype=amp_dtype,
            batch_size=batch_size,
            mode=args.compile_mode,
        )

    optimizer = make_optimizer(model, args.lr, device)
    fused_available = bool(optimizer.defaults.get("fused", False))
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
        "eval_batch_size": eval_batch_size,
        "lr": args.lr,
        "lr_end": args.lr if args.lr_end is None else args.lr_end,
        "amp_dtype": args.amp_dtype,
        "device": str(device),
        "data_device": str(data_device),
        "parameters": n_params,
        "compiled": bool(compiled_ok),
        "compile_mode": args.compile_mode if compiled_ok else None,
        "gpu_resident": data_device.type == "cuda",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.workers:
        print(f"Note: --workers {args.workers} is unused (packed device gathers, no DataLoader)")
    mem = ""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        mem = f" | GPU alloc {allocated:.2f} GB"
    print(
        f"Training on {device} | data {data_device} | AMP: {args.amp_dtype} | "
        f"TF32: On | Fused Adam: {fused_available} | "
        f"{n_batches} x {batch_size}{mem}"
    )

    history: list[dict[str, Any]] = []
    best_test = float("inf")
    best_path = run_dir / "best.pt"
    rng = torch.Generator(device=train_pack.device)
    rng.manual_seed(int(args.split_seed))

    history.append(
        eval_untrained(
            model,
            train_pack,
            test_pack,
            device,
            train_val_pack=train_val_pack,
            amp_dtype=amp_dtype,
            batch_size=eval_batch_size,
        )
    )

    positions_per_epoch = n_batches * batch_size
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_epoch(
            model,
            train_pack,
            optimizer,
            device,
            batch_size=batch_size,
            n_batches=n_batches,
            scaler=scaler,
            amp_dtype=amp_dtype,
            generator=rng,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0

        metrics = evaluate(
            model, test_pack, device, amp_dtype=amp_dtype, batch_size=eval_batch_size
        )
        train_ce = evaluate(
            model, train_val_pack, device, amp_dtype=amp_dtype, batch_size=eval_batch_size
        )["ce"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        row = {"epoch": epoch, "train_ce": train_ce, "test_ce": metrics["ce"], "seconds": elapsed}
        history.append(row)
        rate = positions_per_epoch / max(train_s, 1e-9)
        log_epoch(
            epoch,
            train_ce,
            metrics["ce"],
            elapsed,
            extras=f"train {train_s:.2f}s | {rate:,.0f} pos/s",
        )

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
