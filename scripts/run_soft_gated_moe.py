#!/usr/bin/env python3
"""GOAL.md §4 — End-to-End Soft-Gated MoE.

Jointly trains a differentiable top-1 / top-2 softmax gating network and $K$
expert (L2, head) blocks with the shared L1 frozen, using an auxiliary
load-balancing loss. The gate + experts are trained end-to-end on the 2M-row
``moe_b4_2m`` pack (no pre-clustering, no pre-trained dispatcher).

Sweeps $K \\in \\{2, 4, 8, 16\\}$ for both top-1 and top-2 gating. Per run:

  1. build ``SoftGatedMoE`` (experts cloned from the base, gate random-init);
  2. train joint CE + $\\alpha$·load-balancing loss ($\\alpha=0.01$);
  3. evaluate base-vs-MoE CE/MAE on the 1% test split, plus gate entropy and
     per-expert load distribution.

Writes into ``RESULTS/moe/soft_gated_k<K>_top<tk>/``:
  model.pt, history.json, eval.json, metrics.json, summary.json, plots/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import FenValueVisitsDataset
from tinymlinternship.nnue.moe import SoftGatedMoE, load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import (
    batches_to_device,
    load_train_pack,
    plan_split_indices,
    slice_folders,
    subsample_parts,
)
from tinymlinternship.nnue.moe_pipeline import ce_and_mae, configure_torch
from tinymlinternship.nnue.moe_plots import plot_moe_vs_base

CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
PACK_DIR = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe" / "moe_b4_2m"
OUT_ROOT = PROJECT_ROOT / "RESULTS" / "moe"

KS = (2, 4, 8, 16)
TOP_KS = (1, 2)
EPOCHS = 5
BATCH_SIZE = 2048
LR = 1e-3
ALPHA = 0.01
EVAL_BATCH = 4096
TEST_FRACTION = 0.01
MAX_TEST = 50_000


def log(message: str) -> None:
    print(message, flush=True)


def ce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=-1)
    nll = -(target * log_p).sum(dim=-1)
    w = weight.clamp(min=0.0)
    return (nll * w).sum() / w.sum().clamp_min(1e-8)


def load_balance_loss(gate_probs: torch.Tensor, topk_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    b = int(gate_probs.shape[0])
    f = torch.zeros(n_experts, device=gate_probs.device, dtype=gate_probs.dtype)
    idx = topk_idx.reshape(-1)
    f = f.scatter_add(0, idx, torch.ones_like(idx, dtype=gate_probs.dtype)) / b
    p = gate_probs.mean(dim=0)
    return float(n_experts) * (f * p).sum()


def train_epoch(
    moe: SoftGatedMoE,
    pack,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    batch_size: int,
    alpha: float,
    rng: np.random.RandomState,
) -> tuple[float, float, float]:
    moe.train()
    n = len(pack)
    perm = rng.permutation(n).astype(np.int64)
    ce_acc = 0.0
    lb_acc = 0.0
    n_batches = 0
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        batch = batches_to_device(pack.gather(torch.from_numpy(idx)), device)
        optimizer.zero_grad(set_to_none=True)
        logits, gate_probs, topk_idx = moe(
            batch["white_idx"],
            batch["black_idx"],
            batch["stm_white"],
            batch["white_mask"],
            batch["black_mask"],
        )
        ce = ce_loss(logits, batch["target"], batch["weight"])
        lb = load_balance_loss(gate_probs, topk_idx, moe.n_experts)
        loss = ce + alpha * lb
        loss.backward()
        optimizer.step()
        ce_acc += float(ce.item())
        lb_acc += float(lb.item())
        n_batches += 1
    return ce_acc / max(n_batches, 1), lb_acc / max(n_batches, 1), float(n)


@torch.inference_mode()
def evaluate(
    base,
    moe: SoftGatedMoE,
    folders: list[Path],
    device: torch.device,
    *,
    max_test: int = MAX_TEST,
    batch_size: int = EVAL_BATCH,
) -> dict:
    planned = plan_split_indices(folders, TEST_FRACTION, seed=0)
    test_parts = subsample_parts([p[2] for p in planned], max_test, seed=1)
    base_ce = base_mae = moe_ce = moe_mae = 0.0
    entropy = 0.0
    load = np.zeros(moe.n_experts, dtype=np.float64)
    w_sum = 0.0
    n_rows = 0
    base.eval()
    moe.eval()
    for (folder, _tr, _te), rows in zip(planned, test_parts):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            continue
        ds = FenValueVisitsDataset(folder, progress=False)
        for start in range(0, int(rows.size), int(batch_size)):
            idx = rows[start : start + int(batch_size)]
            batch = batches_to_device(ds.gather(idx), device)
            base_logits = base(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            moe_logits, gate_probs, topk_idx = moe(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            bce, bmae, w = ce_and_mae(base_logits, batch["target"], batch["weight"])
            mce, mmae, _ = ce_and_mae(moe_logits, batch["target"], batch["weight"])
            base_ce += float(bce.item())
            base_mae += float(bmae.item())
            moe_ce += float(mce.item())
            moe_mae += float(mmae.item())
            w_sum += float(w.item())
            n_rows += int(idx.size)
            ent = -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=-1).mean()
            entropy += float(ent.item()) * int(idx.size)
            for e in topk_idx.reshape(-1).cpu().numpy():
                load[int(e)] += 1.0
        del ds
    denom = max(w_sum, 1e-8)
    load = load / max(float(load.sum()), 1.0)
    return {
        "n_test": n_rows,
        "base_ce": base_ce / denom,
        "base_mae": base_mae / denom,
        "moe_ce": moe_ce / denom,
        "moe_mae": moe_mae / denom,
        "gate_entropy": entropy / max(n_rows, 1),
        "load_distribution": [round(float(x), 5) for x in load],
        "load_variance": float(np.var(load)),
        "load_max_min_ratio": float(load.max() / max(load.min(), 1e-9)),
    }


def plot_history(history: list[dict], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_ce = [row["train_ce"] for row in history]
    test_ce = [row["test_ce"] for row in history]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--top-ks", type=int, nargs="+", default=list(TOP_KS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = full pack; else subsample (for smoke)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.epochs = 1
        if args.max_rows == 0:
            args.max_rows = 50_000

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(device)

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else (PROJECT_ROOT / args.checkpoint)
    base = load_dual_hidden_checkpoint(ckpt, device=device)
    log(f"device={device} | checkpoint={ckpt}")

    slices_dir = args.slices_dir if args.slices_dir.is_absolute() else (PROJECT_ROOT / args.slices_dir)
    folders = slice_folders(slices_dir)
    if not folders:
        log(f"no slice folders in {slices_dir}")
        return 1

    pack_dir = args.pack_dir if args.pack_dir.is_absolute() else (PROJECT_ROOT / args.pack_dir)
    pack, _slice_ids, _local_rows = load_train_pack(pack_dir)
    n = len(pack)
    log(f"train pack rows {n:,}")
    if args.max_rows > 0 and args.max_rows < n:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(n, size=args.max_rows, replace=False)).astype(np.int64)
        pack = pack.index_select(torch.from_numpy(keep))
        n = len(pack)
        log(f"subsampled pack to {n:,} rows")

    results: dict[str, dict] = {}
    for top_k in args.top_ks:
        for k in args.ks:
            run_dir = OUT_ROOT / f"soft_gated_k{k}_top{top_k}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log(f"=== top-{top_k} | K={k} ===")

            moe = SoftGatedMoE(base, n_experts=k, top_k=top_k).to(device)
            moe.freeze_l1()
            trainable = sum(p.numel() for p in moe.parameters() if p.requires_grad)
            log(f"  trainable params {trainable:,}")

            optimizer = torch.optim.Adam([p for p in moe.parameters() if p.requires_grad], lr=args.lr)
            rng = np.random.RandomState(0)

            history: list[dict] = []
            t0 = time.perf_counter()
            best_test = float("inf")
            best_state = None
            for epoch in range(1, args.epochs + 1):
                t_ep = time.perf_counter()
                train_ce, train_lb, _n = train_epoch(
                    moe, pack, optimizer, device, batch_size=BATCH_SIZE, alpha=args.alpha, rng=rng
                )
                ev = evaluate(base, moe, folders, device, max_test=MAX_TEST)
                history.append(
                    {
                        "epoch": epoch,
                        "train_ce": train_ce,
                        "train_lb": train_lb,
                        "test_ce": ev["moe_ce"],
                        "test_mae": ev["moe_mae"],
                        "gate_entropy": ev["gate_entropy"],
                        "seconds": time.perf_counter() - t_ep,
                    }
                )
                log(
                    f"  epoch {epoch:02d} | train_ce={train_ce:.5f} lb={train_lb:.4f} | "
                    f"test_ce={ev['moe_ce']:.5f} | ent={ev['gate_entropy']:.3f} | "
                    f"{history[-1]['seconds']:.1f}s"
                )
                if ev["moe_ce"] < best_test:
                    best_test = ev["moe_ce"]
                    best_state = {kk: v.detach().cpu().clone() for kk, v in moe.state_dict().items()}

            if best_state is not None:
                moe.load_state_dict(best_state)
            final = evaluate(base, moe, folders, device, max_test=MAX_TEST)
            elapsed = time.perf_counter() - t0

            moe.save(run_dir / "model.pt")
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            (run_dir / "eval.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
            plot_history(history, run_dir / "plots" / "ce.png", title=f"Soft-gated MoE top-{top_k} K={k}")
            plot_moe_vs_base(final, run_dir / "plots" / "moe_vs_base_ce.png")

            summary = {
                "run_name": run_dir.name,
                "checkpoint": str(ckpt),
                "pack_dir": str(pack_dir),
                "top_k": top_k,
                "n_experts": k,
                "alpha": args.alpha,
                "epochs": args.epochs,
                "lr": args.lr,
                "trainable_params": trainable,
                "train_rows": n,
                "metrics": final,
                "best_test_ce": best_test,
                "seconds": elapsed,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            results[run_dir.name] = final
            log(
                f"  K={k} top-{top_k}: base_ce={final['base_ce']:.5f} moe_ce={final['moe_ce']:.5f} "
                f"(mae {final['moe_mae']:.5f}) | load_var={final['load_variance']:.4f} | {elapsed:.0f}s"
            )

    (OUT_ROOT / "soft_gated_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log("SOFT_GATED_MOE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
