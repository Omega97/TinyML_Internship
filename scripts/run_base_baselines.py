#!/usr/bin/env python3
"""Train dense dual-POV baselines (Linear + single-hidden FFNN) on the GPU compact path.

Mirrors ``scripts/train_nnue-gpu.py`` so the numbers are directly comparable to the
DualHidden NNUE bases already staged in ``RESULTS/base/``: same random per-slice
test split (``--test-fraction``), same soft cross-entropy target (STM WDL), same
CE/MAE metrics, and the same train-vs-test CE plot. On top of that it writes MSE,
R2 and inference throughput (NPS) to ``metrics.json``, which the NNUE staging
currently omits.

Architectures (all import from ``tinymlinternship.nnue.model``):

  linear -> LinearWDLNNUE    : concat [STM || opp] 2x844 -> 3 logits
  medium -> MediumWDLNNUE    : concat [STM || opp] 2x844 -> H CReLU -> 3 logits (single hidden)
  dual   -> DualHiddenFFNN   : concat [STM || opp] 2x844 -> H1 CReLU -> H2 CReLU -> 3 logits (dual hidden)

The compact int16 tables are loaded once and reused across every run, so a full
battery (1 linear + 3 medium widths) pays the data-load cost a single time.

    python scripts/run_base_baselines.py --smoke
    python scripts/run_base_baselines.py
"""

from __future__ import annotations

import argparse
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

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import (
    CompactPackedTensors,
    discover_slice_folders,
    load_compact_split,
    slice_source_json,
)
from tinymlinternship.nnue.model import DualHiddenFFNN, LinearWDLNNUE, MediumWDLNNUE

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
PLOTS_DIR = PROJECT_ROOT / "plots"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def slice_folders(slices_dir: Path) -> list[Path]:
    folders = discover_slice_folders(slices_dir)
    if folders:
        return folders
    if slice_source_json(slices_dir) is not None:
        return [slices_dir]
    return []


def configure_torch(device: torch.device) -> None:
    torch.set_num_threads(os.cpu_count() or 1)
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def sparse_logits(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Scatter sparse indices into dense (B, 844) POV vectors, then call the dense
    ``forward``. Matmuls run on Tensor Cores (the embedding-gather ``forward_sparse``
    is bandwidth-bound and too slow for the full dataset)."""
    feat_dim = int(model.feature_dim)
    b = batch["white_idx"]
    device = b.device
    white = torch.zeros(b.shape[0], feat_dim, device=device, dtype=torch.float32)
    black = torch.zeros(b.shape[0], feat_dim, device=device, dtype=torch.float32)
    white.scatter_add_(1, b.long().clamp(min=0, max=feat_dim - 1), batch["white_mask"].to(white.dtype))
    black.scatter_add_(
        1, batch["black_idx"].long().clamp(min=0, max=feat_dim - 1), batch["black_mask"].to(black.dtype)
    )
    return model(white, black, batch["stm_white"])


def ce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=-1)
    nll = -(target * log_p).sum(dim=-1)
    w = weight.clamp(min=0.0)
    denom = w.sum().clamp(min=1e-8)
    return (nll * w).sum() / denom


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    if batch["target"].device == device:
        return batch
    return {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    pack: CompactPackedTensors,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    batch_size: int = 32768,
    with_regression: bool = False,
) -> dict[str, float]:
    model.eval()
    use_amp = device.type == "cuda" and amp_dtype is not None
    ce_sum = torch.zeros((), device=device, dtype=torch.float32)
    mae_sum = torch.zeros((), device=device, dtype=torch.float32)
    mse_sum = torch.zeros((), device=device, dtype=torch.float32)
    ss_res_sum = torch.zeros((), device=device, dtype=torch.float32)
    ss_tot_sum = torch.zeros((), device=device, dtype=torch.float32)
    w_sum = torch.zeros((), device=device, dtype=torch.float32)
    tgt_mean_sum = torch.zeros((), device=device, dtype=torch.float32)

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
        if with_regression:
            mse_sum = mse_sum + ((pred_v - tgt_v) ** 2 * w).sum()
        tgt_mean_sum = tgt_mean_sum + (tgt_v * w).sum()
        w_sum = w_sum + w.sum()

    denom = w_sum.clamp(min=1e-8)
    out: dict[str, float] = {
        "ce": float((ce_sum / denom).item()),
        "mae": float((mae_sum / denom).item()),
        "n": len(pack),
    }
    if with_regression:
        tgt_mean = tgt_mean_sum / denom
        # second pass over predictions is expensive; approximate ss_tot with the
        # variance of the target computed via a lightweight re-scan of target only.
        ss_tot = torch.zeros((), device=device, dtype=torch.float32)
        for batch in pack.iter_batches(batch_size, pad=True):
            batch = batch_to_device(batch, device)
            w2 = batch["weight"].clamp(min=0.0)
            tgt_v2 = batch["target"][:, 0] - batch["target"][:, 2]
            ss_tot = ss_tot + (((tgt_v2 - tgt_mean) ** 2) * w2).sum()
        out["mse"] = float((mse_sum / denom).item())
        out["r2"] = float((1.0 - mse_sum / ss_tot.clamp(min=1e-8)).item())
    return out


def train_epoch(
    model: torch.nn.Module,
    pack: CompactPackedTensors,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    batch_size: int,
    n_batches: int,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> float:
    model.train()
    use_amp = device.type == "cuda" and amp_dtype is not None
    n = len(pack)
    loss_acc = torch.zeros((), device=device, dtype=torch.float32)

    for _ in range(int(n_batches)):
        idx = torch.randint(n, (batch_size,), device=pack.device, generator=generator)
        batch = batch_to_device(pack.gather(idx), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = sparse_logits(model, batch)
            loss = ce_loss(logits, batch["target"], batch["weight"])
        loss.backward()
        optimizer.step()
        loss_acc = loss_acc + loss.detach()

    return float(loss_acc.item()) / max(int(n_batches), 1)


def make_optimizer(model: torch.nn.Module, lr: float, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": lr}
    if device.type == "cuda":
        kwargs["fused"] = True
        kwargs["capturable"] = True
    try:
        return torch.optim.Adam(model.parameters(), **kwargs)
    except TypeError:
        return torch.optim.Adam(model.parameters(), lr=lr)


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
        optimizer, start_factor=1.0, end_factor=float(lr_end) / float(lr), total_iters=int(epochs)
    )


def plot_ce(history: list[dict[str, Any]], path: Path, *, title: str) -> None:
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


def subset_pack(pack: CompactPackedTensors, size: int, seed: int) -> CompactPackedTensors:
    n = len(pack)
    take = int(size)
    if take <= 0 or take >= n:
        return pack
    g = torch.Generator(device=pack.device)
    g.manual_seed(int(seed))
    idx = torch.randperm(n, generator=g, device=pack.device)[:take]
    return pack.index_select(idx)


def run_one(
    *,
    model: torch.nn.Module,
    arch: str,
    run_name: str,
    output_dir: Path,
    train_pack: CompactPackedTensors,
    test_pack: CompactPackedTensors,
    train_val_pack: CompactPackedTensors,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    epochs: int,
    batch_size: int,
    n_batches: int,
    eval_batch_size: int,
    lr: float,
    lr_end: float | None,
    split_seed: int,
    slices_dir: Path,
    test_fraction: float,
) -> dict[str, Any]:
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    optimizer = make_optimizer(model, lr, device)
    scheduler = make_linear_lr_scheduler(optimizer, lr=lr, lr_end=lr_end, epochs=epochs)

    config = {
        "slices_dir": str(slices_dir),
        "test_fraction": float(test_fraction),
        "split": "random fraction of each slice -> test",
        "split_seed": int(split_seed),
        "architecture": arch,
        "hidden_dim": int(model.hidden_dim) if getattr(model, "hidden_dim", None) is not None else None,
        "hidden1_dim": int(model.hidden1_dim) if getattr(model, "hidden1_dim", None) is not None else None,
        "hidden2_dim": int(model.hidden2_dim) if getattr(model, "hidden2_dim", None) is not None else None,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "batches_per_epoch": int(n_batches),
        "eval_batch_size": int(eval_batch_size),
        "lr": float(lr),
        "lr_end": float(lr) if lr_end is None else float(lr_end),
        "amp_dtype": "bfloat16" if amp_dtype == torch.bfloat16 else ("float16" if amp_dtype == torch.float16 else "float32"),
        "device": str(device),
        "data_device": str(train_pack.device),
        "parameters": int(model.count_parameters()),
        "loss": "unweighted soft cross-entropy, target = STM WDL",
        "output": "3 logits + softmax (W, D, L) from side to move",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        f"[{run_name}] {arch} | {model.count_parameters():,} params | "
        f"{n_batches} x {batch_size} steps/epoch | epochs {epochs} | {device}",
        flush=True,
    )

    rng = torch.Generator(device=train_pack.device)
    rng.manual_seed(int(split_seed))

    history: list[dict[str, Any]] = []

    # epoch 0 (untrained)
    t0 = time.perf_counter()
    train_m = evaluate(model, train_val_pack, device, amp_dtype=amp_dtype, batch_size=eval_batch_size)
    test_m = evaluate(model, test_pack, device, amp_dtype=amp_dtype, batch_size=eval_batch_size)
    history.append({"epoch": 0, "train_ce": train_m["ce"], "test_ce": test_m["ce"], "seconds": time.perf_counter() - t0})

    best_test = float("inf")
    best_path = run_dir / "best.pt"
    positions_per_epoch = n_batches * batch_size
    last_train_s = 0.0

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_ce = train_epoch(
            model, train_pack, optimizer, device,
            batch_size=batch_size, n_batches=n_batches, amp_dtype=amp_dtype, generator=rng,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        last_train_s = train_s

        test_m = evaluate(model, test_pack, device, amp_dtype=amp_dtype, batch_size=eval_batch_size)
        train_m = evaluate(model, train_val_pack, device, amp_dtype=amp_dtype, batch_size=eval_batch_size)
        elapsed = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_ce": train_m["ce"],
            "test_ce": test_m["ce"],
            "test_mae": test_m["mae"],
            "seconds": elapsed,
        }
        history.append(row)
        rate = positions_per_epoch / max(train_s, 1e-9)
        print(
            f"  epoch {epoch:03d} | train_ce={row['train_ce']:.6f} | "
            f"test_ce={row['test_ce']:.6f} | mae={row['test_mae']:.6f} | {elapsed:.1f}s | {rate:,.0f} pos/s",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()

        payload = {
            "model_state_dict": model.state_dict(),
            "architecture": arch,
            "n_outputs": 3,
            "test_ce": row["test_ce"],
            "epoch": epoch,
        }
        if row["test_ce"] < best_test:
            best_test = row["test_ce"]
            torch.save(payload, best_path)
        torch.save(payload, run_dir / "last.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_path = run_dir / "ce.png"
    plot_ce(history, plot_path, title=f"{arch} ({run_name})")

    # Final full-test regression metrics + throughput.
    t0 = time.perf_counter()
    final = evaluate(
        model, test_pack, device, amp_dtype=amp_dtype,
        batch_size=eval_batch_size, with_regression=True,
    )
    eval_s = time.perf_counter() - t0
    n_test = int(final["n"])
    metrics = {
        "run_name": run_name,
        "architecture": arch,
        "parameters": int(model.count_parameters()),
        "n_test": n_test,
        "test_ce": float(final["ce"]),
        "test_mae": float(final["mae"]),
        "test_mse": float(final["mse"]),
        "test_r2": float(final["r2"]),
        "train_nps": float(positions_per_epoch / max(last_train_s, 1e-9)),
        "infer_nps": float(n_test / max(eval_s, 1e-9)),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[{run_name}] best_test_ce={best_test:.6f} | metrics={metrics}", flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train dense baselines (linear + single-hidden FFNN).")
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--batches-per-epoch", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-end", type=float, default=1e-3)
    parser.add_argument("--medium-hidden", type=str, default="64,128,256")
    parser.add_argument("--dual-hidden", type=str, default="64,128,256")
    parser.add_argument("--output-dir", type=Path, default=NNUE_CHECKPOINTS_DIR.parent / "baselines")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--only", type=str, default=None, help="comma list: linear,medium,dual")
    parser.add_argument("--max-slices", type=int, default=0, help="Use only the first N slice folders (0 = all)")
    parser.add_argument("--smoke", action="store_true", help="2 epochs, 4 batches/epoch for a quick sanity check")
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(device)

    amp_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": None}
    amp_dtype = amp_dtype_map[args.amp_dtype]

    slices_dir = _resolve(args.slices_dir)
    folders = slice_folders(slices_dir)
    if not folders:
        print(f"no slice JSON folders in {slices_dir}", file=sys.stderr)
        return 1

    if args.smoke:
        epochs = 2
        batches_per_epoch = 4
        batch_size = 2048
        max_slices = args.max_slices or 8
    else:
        epochs = args.epochs
        batches_per_epoch = args.batches_per_epoch
        batch_size = args.batch_size
        max_slices = args.max_slices

    if max_slices > 0:
        folders = folders[:max_slices]

    data_device = device if device.type == "cuda" else torch.device("cpu")
    print(f"Loading compact int16 tables onto {data_device} (test_fraction={args.test_fraction}) ...", flush=True)
    train_pack, test_pack_full = load_compact_split(
        folders, args.test_fraction, seed=args.split_seed,
        rebuild=False, progress=True, device=data_device,
    )
    n_train = len(train_pack)
    n_test = len(test_pack_full)
    print(f"  train {n_train:,} | test {n_test:,} | train {train_pack.nbytes/1e9:.2f} GB", flush=True)

    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else max(batch_size, 32768)
    train_val_pack = subset_pack(train_pack, 3200, args.split_seed)

    only = set((args.only or "").replace(",", " ").split())
    if not only:
        only = {"linear", "medium", "dual"}

    metrics: dict[str, dict[str, Any]] = {}

    if "linear" in only:
        model = LinearWDLNNUE().to(device)
        metrics["linear"] = run_one(
            model=model, arch="linear_wdl", run_name="linear_wdl",
            output_dir=_resolve(args.output_dir),
            train_pack=train_pack, test_pack=test_pack_full, train_val_pack=train_val_pack,
            device=device, amp_dtype=amp_dtype,
            epochs=epochs, batch_size=batch_size, n_batches=batches_per_epoch,
            eval_batch_size=eval_batch_size, lr=args.lr, lr_end=args.lr_end,
            split_seed=args.split_seed, slices_dir=slices_dir, test_fraction=args.test_fraction,
        )
        del model

    if "medium" in only:
        for h in [int(x) for x in args.medium_hidden.split(",") if x.strip()]:
            model = MediumWDLNNUE(hidden_dim=h).to(device)
            metrics[f"medium_h{h}"] = run_one(
                model=model, arch="medium_wdl", run_name=f"medium_h{h}",
                output_dir=_resolve(args.output_dir),
                train_pack=train_pack, test_pack=test_pack_full, train_val_pack=train_val_pack,
                device=device, amp_dtype=amp_dtype,
                epochs=epochs, batch_size=batch_size, n_batches=batches_per_epoch,
                eval_batch_size=eval_batch_size, lr=args.lr, lr_end=args.lr_end,
                split_seed=args.split_seed, slices_dir=slices_dir, test_fraction=args.test_fraction,
            )
            del model

    if "dual" in only:
        for h in [int(x) for x in args.dual_hidden.split(",") if x.strip()]:
            model = DualHiddenFFNN(hidden1_dim=h, hidden2_dim=h).to(device)
            metrics[f"ffnn2_h{h}"] = run_one(
                model=model, arch="ffnn_dual_hidden_wdl", run_name=f"ffnn2_h{h}",
                output_dir=_resolve(args.output_dir),
                train_pack=train_pack, test_pack=test_pack_full, train_val_pack=train_val_pack,
                device=device, amp_dtype=amp_dtype,
                epochs=epochs, batch_size=batch_size, n_batches=batches_per_epoch,
                eval_batch_size=eval_batch_size, lr=args.lr, lr_end=args.lr_end,
                split_seed=args.split_seed, slices_dir=slices_dir, test_fraction=args.test_fraction,
            )
            del model

    summary_path = _resolve(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
