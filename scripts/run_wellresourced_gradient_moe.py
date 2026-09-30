#!/usr/bin/env python3
"""GOAL.md §4 — Well-Resourced Gradient-Clustered Hard MoE (K=8).

Targets the two bottlenecks isolated in Runs 6–11 — expert underfitting and the
weak L1→gradient dispatcher — by scaling the resources up to the §4 spec:

  1. mini-batch $k$-Means on the cached 48-d sample head gradients, $K=8$;
  2. single-hidden MLP dispatcher ``L1 -> h=64 -> Bucket ID`` trained for
     **200 epochs** (early-stopped on a 90/10 validation split);
  3. 8 expert NNUE (L2, OUT) blocks (frozen L1), cloned from base, fine-tuned
     for **100 epochs** with warm-start + cosine LR decay on **7M positions per
     head** (≈56M total, new ``moe_b8_56m`` pack);
  4. base-vs-MoE eval (CE + MAE), the nearest-gradient-centroid oracle, and the
     true argmin-loss oracle upper bound.

Writes into ``RESULTS/moe/wellresourced_k8/``: labels.npy, centroids.npy,
diagnostics.json, dispatcher.pt, dispatcher_history.json, labels_dispatcher.npy,
expert_metrics.json, eval.json, oracle_argmin.json, summary.json, plots/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.cluster import (
    cluster_diagnostics,
    fit_minibatch_kmeans,
    save_cluster_run,
)
from tinymlinternship.nnue.dataset import FenValueVisitsDataset
from tinymlinternship.nnue.moe import MLPDispatcher, load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import (
    batches_to_device,
    load_train_pack,
    plan_split_indices,
    slice_folders,
    subsample_parts,
)
from tinymlinternship.nnue.moe_pipeline import (
    ce_and_mae,
    compute_gradients,
    configure_torch,
    evaluate_moe,
    fine_tune_experts,
)
from tinymlinternship.nnue.moe_plots import plot_expert_ce_by_bucket, plot_moe_vs_base

CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
PACK_DIR = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe" / "moe_b8_56m"
OUT_ROOT = PROJECT_ROOT / "RESULTS" / "moe"
RUN_NAME = "wellresourced_k8"

K = 8
MLP_HIDDEN = 64
DISPATCHER_EPOCHS = 200
EXPERT_EPOCHS = 100
TOTAL_ROWS = 56_000_000
ROWS_PER_HEAD = 7_000_000
VAL_FRACTION = 0.10
SPLIT_SEED = 0
BATCH_SIZE = 4096
LR = 1e-2
LR_END = 1e-3
EXPERT_LR = 1e-3
EXPERT_LR_END = 1e-4
EXPERT_BATCH = 2048
L1_CHUNK = 16_384
CLUSTER_CHUNK = 1_000_000
PACK_FILES = ("train_pack.pt", "slice_ids.npy", "local_rows.npy")
GRAD_FILES = ("gradients.npy", "meta.json")


def log(message: str) -> None:
    print(message, flush=True)


def build_l1_gpu(pack, model, device: torch.device) -> torch.Tensor:
    """Compute the 256-d L1 concat for every row, resident on-device as float32."""
    n = len(pack)
    hidden = int(model.hidden_dim)
    out = torch.empty((n, 2 * hidden), dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, n, L1_CHUNK):
            end = min(start + L1_CHUNK, n)
            batch = pack.gather(torch.arange(start, end, dtype=torch.long))
            h = model.l1_concat(
                batch["white_idx"].to(device),
                batch["black_idx"].to(device),
                batch["stm_white"].to(device),
                batch["white_mask"].to(device),
                batch["black_mask"].to(device),
            )
            out[start:end] = h.detach()
            if end == n or (start // L1_CHUNK) % 25 == 0:
                log(f"  l1 {end:,}/{n:,}")
    return out


def train_mlp_dispatcher(
    l1: torch.Tensor,
    labels: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_clusters: int,
    epochs: int,
    device: torch.device,
) -> tuple[MLPDispatcher, dict]:
    in_dim = int(l1.shape[1])
    disp = MLPDispatcher(in_dim, n_clusters, hidden_dim=MLP_HIDDEN).to(device)
    opt = torch.optim.Adam(disp.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=LR_END / LR, total_iters=max(epochs, 1)
    )
    idx_train = torch.from_numpy(train_idx.astype(np.int64)).to(device)
    idx_val = torch.from_numpy(val_idx.astype(np.int64)).to(device)
    y_train = torch.from_numpy(labels[train_idx].astype(np.int64)).to(device)
    y_val = torch.from_numpy(labels[val_idx].astype(np.int64)).to(device)

    history: list[dict] = []
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in disp.state_dict().items()}
    n_train = int(train_idx.shape[0])
    n_val = int(val_idx.shape[0])

    for epoch in range(1, epochs + 1):
        disp.train()
        t0 = time.perf_counter()
        train_loss = 0.0
        train_correct = 0
        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            x = l1[idx_train[start:end]]
            y = y_train[start:end]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(disp(x), y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * int(end - start)
            train_correct += int((disp(x).argmax(dim=-1) == y).sum().item())
        sched.step()
        disp.eval()
        val_correct = 0
        with torch.no_grad():
            for start in range(0, n_val, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n_val)
                x = l1[idx_val[start:end]]
                y = y_val[start:end]
                val_correct += int((disp(x).argmax(dim=-1) == y).sum().item())
        train_acc = train_correct / max(n_train, 1)
        val_acc = val_correct / max(n_val, 1)
        history.append(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_loss": train_loss / max(n_train, 1),
                "seconds": time.perf_counter() - t0,
            }
        )
        if epoch % 10 == 0 or epoch == epochs:
            log(f"  dispatcher epoch {epoch:03d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in disp.state_dict().items()}

    disp.load_state_dict(best_state)
    disp.eval()
    return disp, {"best_val_acc": best_acc, "history": history, "n_train": n_train, "n_val": n_val}


def predict_full(disp: MLPDispatcher, l1: torch.Tensor, device: torch.device) -> np.ndarray:
    n = int(l1.shape[0])
    out = np.empty(n, dtype=np.int16)
    disp.eval()
    with torch.no_grad():
        for start in range(0, n, L1_CHUNK):
            end = min(start + L1_CHUNK, n)
            out[start:end] = disp(l1[start:end]).argmax(dim=-1).cpu().numpy().astype(np.int16)
    return out


@torch.inference_mode()
def evaluate_argmin_oracle(
    base,
    moe,
    folders: list[Path],
    device: torch.device,
    *,
    max_test: int = 50_000,
    batch_size: int = 4096,
) -> dict:
    """True routing upper bound: route each test position to argmin_k CE."""
    planned = plan_split_indices(folders, 0.01, seed=0)
    test_parts = subsample_parts([p[2] for p in planned], max_test, seed=1)
    base_ce = base_mae = oracle_ce = oracle_mae = 0.0
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
            target = batch["target"].float()
            weight = batch["weight"]
            base_logits = base(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            h = base.l1_concat(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            stacked = moe.all_expert_logits_from_h(h)  # (B, K, 3)
            log_p = F.log_softmax(stacked.float(), dim=-1)
            nll_e = -(target[:, None, :] * log_p).sum(dim=-1)  # (B, K)
            oracle_ids = nll_e.argmin(dim=-1)
            rows_t = torch.arange(int(idx.size), device=device)
            oracle_logits = stacked[rows_t, oracle_ids, :]
            bce, bmae, w = ce_and_mae(base_logits, target, weight)
            oce, omae, _ = ce_and_mae(oracle_logits, target, weight)
            base_ce += float(bce.item())
            base_mae += float(bmae.item())
            oracle_ce += float(oce.item())
            oracle_mae += float(omae.item())
            w_sum += float(w.item())
            n_rows += int(idx.size)
        del ds
    denom = max(w_sum, 1e-8)
    return {
        "n_test": n_rows,
        "base_ce": base_ce / denom,
        "base_mae": base_mae / denom,
        "oracle_ce": oracle_ce / denom,
        "oracle_mae": oracle_mae / denom,
    }


def link_files(src_dir: Path, work_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        src = src_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
        dest = work_dir / name
        if dest.is_symlink():
            if dest.resolve() == src.resolve():
                continue
            dest.unlink()
        elif dest.exists():
            dest.unlink()
        dest.symlink_to(src.resolve())


def write_plots(work_dir: Path) -> list[str]:
    plots = work_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    expert_path = work_dir / "expert_metrics.json"
    if expert_path.is_file():
        payload = json.loads(expert_path.read_text(encoding="utf-8"))
        buckets = [b for b in payload.get("buckets", []) if not b.get("skipped")]
        if buckets:
            written.append(
                str(
                    plot_expert_ce_by_bucket(
                        [float(b["base_hold_ce"]) for b in buckets],
                        [float(b["expert_hold_ce"]) for b in buckets],
                        plots / "expert_ce_by_bucket.png",
                        tick_labels=[str(b["expert"]) for b in buckets],
                    )
                )
            )
    eval_path = work_dir / "eval.json"
    if eval_path.is_file():
        metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        written.append(str(plot_moe_vs_base(metrics, plots / "moe_vs_base_ce.png")))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--max-rows", type=int, default=TOTAL_ROWS)
    parser.add_argument("--dispatcher-epochs", type=int, default=DISPATCHER_EPOCHS)
    parser.add_argument("--expert-epochs", type=int, default=EXPERT_EPOCHS)
    parser.add_argument("--expert-lr", type=float, default=EXPERT_LR)
    parser.add_argument("--expert-lr-end", type=float, default=EXPERT_LR_END)
    parser.add_argument("--max-test", type=int, default=50_000)
    parser.add_argument("--resume-grads", action="store_true")
    parser.add_argument("--skip-grads", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.max_rows = 200_000
        args.dispatcher_epochs = 2
        args.expert_epochs = 1
        args.max_test = 1024
        args.pack_dir = args.pack_dir.with_name(args.pack_dir.name + "_smoke")

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
    pack_dir.mkdir(parents=True, exist_ok=True)
    grad_path = pack_dir / "gradients.npy"
    if args.skip_grads and grad_path.is_file():
        log(f"skipping gradient computation (reusing {grad_path})")
    elif not grad_path.is_file() or not (pack_dir / "meta.json").is_file():
        log(f"building {args.max_rows:,}-row pack + 48-d gradients in {pack_dir}")
        compute_gradients(
            base,
            folders,
            pack_dir,
            device=device,
            max_rows=args.max_rows,
            batch_size=2048,
            resume=args.resume_grads,
            log=log,
        )

    work_dir = OUT_ROOT / RUN_NAME
    work_dir.mkdir(parents=True, exist_ok=True)
    link_files(pack_dir, work_dir, PACK_FILES)
    link_files(pack_dir, work_dir, GRAD_FILES)

    log(f"=== K={K} | mini-batch k-means on 48-d gradients ===")
    t0 = time.perf_counter()
    gradients = np.load(pack_dir / "gradients.npy", mmap_mode="r")
    n = int(gradients.shape[0])
    km = fit_minibatch_kmeans(gradients, K, seed=0)
    labels = np.empty(n, dtype=np.int16)
    for start in range(0, n, CLUSTER_CHUNK):
        end = min(start + CLUSTER_CHUNK, n)
        x = np.ascontiguousarray(gradients[start:end], dtype=np.float32)
        labels[start:end] = km.predict(x).astype(np.int16)
    centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
    diag = cluster_diagnostics(labels, centroids, inertia=float(km.inertia_))
    diag["algorithm"] = "minibatch_kmeans"
    diag["representation"] = "head_gradient_48d"
    diag["requested_clusters"] = K
    save_cluster_run(work_dir, labels=labels, centroids=centroids, diagnostics=diag)
    log(f"  sizes {diag['sizes']} | empty={diag['empty']} ({time.perf_counter() - t0:.1f}s)")
    del gradients

    pack, _slice_ids, _local_rows = load_train_pack(pack_dir)
    log(f"train pack rows {len(pack):,}")

    log(f"=== building L1 activations (GPU float32) for dispatcher ===")
    t1 = time.perf_counter()
    l1 = build_l1_gpu(pack, base, device)
    log(
        f"  L1 tensor {tuple(l1.shape)} {l1.dtype} ({l1.numel() * 4 / 1e9:.2f} GB) "
        f"in {time.perf_counter() - t1:.1f}s"
    )
    del pack
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n)
    n_val = max(1, int(n * VAL_FRACTION))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    log(f"=== K={K} | MLP dispatcher (h={MLP_HIDDEN}, {args.dispatcher_epochs} epochs) ===")
    t2 = time.perf_counter()
    disp, disp_info = train_mlp_dispatcher(
        l1, labels, train_idx=train_idx, val_idx=val_idx, n_clusters=K,
        epochs=args.dispatcher_epochs, device=device,
    )
    log(f"  best val acc {disp_info['best_val_acc']:.4f} ({time.perf_counter() - t2:.1f}s)")
    torch.save(
        {
            "state_dict": disp.state_dict(),
            "in_dim": disp.in_dim,
            "n_clusters": K,
            "hidden_dim": disp.hidden_dim,
            "val_acc": disp_info["best_val_acc"],
            "dispatcher_kind": "mlp",
        },
        work_dir / "dispatcher.pt",
    )
    (work_dir / "dispatcher_history.json").write_text(
        json.dumps({"best_val_acc": disp_info["best_val_acc"], "history": disp_info["history"]}, indent=2),
        encoding="utf-8",
    )
    disp_labels = predict_full(disp, l1, device)
    np.save(work_dir / "labels_dispatcher.npy", disp_labels)
    log(f"  dispatcher label counts: {np.bincount(disp_labels.astype(np.int64), minlength=K).tolist()}")

    del l1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    log(f"=== K={K} | expert fine-tune (frozen L1, {args.expert_epochs} epochs, cosine LR) ===")
    moe = fine_tune_experts(
        base,
        work_dir,
        folders,
        device=device,
        n_clusters=K,
        expert_labels="dispatcher",
        epochs=args.expert_epochs,
        batch_size=EXPERT_BATCH,
        lr=args.expert_lr,
        lr_end=args.expert_lr_end,
        dispatcher_kind="mlp",
        dispatcher_hidden=MLP_HIDDEN,
    )

    log(f"=== K={K} | evaluation (centroid oracle) ===")
    metrics = evaluate_moe(
        base, moe, folders, work_dir, device=device, test_fraction=0.01,
        split_seed=0, max_test=args.max_test, oracle=True,
    )
    log(
        f"  eval n={metrics['n_test']:,} | base_ce={metrics['base_ce']:.5f} | "
        f"moe_ce={metrics['moe_ce']:.5f}"
        + (f" | centroid_oracle_ce={metrics['oracle_ce']:.5f}" if "oracle_ce" in metrics else "")
    )

    log(f"=== K={K} | argmin-loss oracle upper bound ===")
    argmin = evaluate_argmin_oracle(base, moe, folders, device, max_test=args.max_test)
    (work_dir / "oracle_argmin.json").write_text(json.dumps(argmin, indent=2), encoding="utf-8")
    log(f"  argmin oracle_ce={argmin['oracle_ce']:.5f} | oracle_mae={argmin['oracle_mae']:.5f}")

    plots = write_plots(work_dir)
    summary = {
        "run_name": RUN_NAME,
        "checkpoint": str(ckpt),
        "pack_dir": str(pack_dir),
        "n_buckets": K,
        "router": "mlp_l1_to_gradient",
        "dispatcher_hidden": MLP_HIDDEN,
        "dispatcher_epochs": args.dispatcher_epochs,
        "dispatcher_val_acc": disp_info["best_val_acc"],
        "expert_labels": "dispatcher",
        "expert_epochs": args.expert_epochs,
        "expert_lr": args.expert_lr,
        "expert_lr_end": args.expert_lr_end,
        "train_rows": n,
        "rows_per_head_target": ROWS_PER_HEAD,
        "cluster_sizes": diag["sizes"],
        "dispatcher_label_sizes": np.bincount(disp_labels.astype(np.int64), minlength=K).tolist(),
        "metrics": metrics,
        "oracle_argmin": argmin,
        "plots": plots,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {work_dir / 'summary.json'}")

    log("WELLRESOURCED_GRADIENT_MOE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
