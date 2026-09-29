#!/usr/bin/env python3
"""GOAL.md §4 — Gradient-Clustered Hard MoE.

Per $K \\in \\{2, 4, 8, 16\\}$:

  1. mini-batch $k$-Means on the 48-d sample head-gradient projection
     ($\\nabla_w \\mathcal{L}$, Gaussian-projected from 66,563-d) — the
     pseudo-ground-truth buckets;
  2. single-hidden MLP dispatcher ``L1 -> h=64 -> Bucket ID`` (L1 activations
     predict the gradient-space clusters);
  3. per-bucket NNUE (L2, head) expert fine-tune with the shared $L1$ frozen;
  4. base-vs-MoE evaluation on the 1% test split (CE + MAE), plus the oracle
     upper bound (route each test position to its nearest gradient centroid).

Reuses the 2M-row ``moe_b4_2m`` training pack and its cached ``gradients.npy``
+ ``meta.json`` (no gradient recomputation). This is the full battery: MLP
dispatcher (not the preliminary linear one) and $K \\in \\{2, 4, 8, 16\\}$.

Writes (per $K$) into ``RESULTS/moe/grad_clustered_k<K>/``:
  labels.npy, centroids.npy, diagnostics.json, dispatcher.pt,
  dispatcher_history.json, labels_dispatcher.npy, expert_metrics.json,
  eval.json, summary.json, plots/
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
from tinymlinternship.nnue.moe import MLPDispatcher, load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import (
    load_train_pack,
    save_train_pack,
    slice_folders,
)
from tinymlinternship.nnue.moe_pipeline import configure_torch, evaluate_moe, fine_tune_experts
from tinymlinternship.nnue.moe_plots import plot_expert_ce_by_bucket, plot_moe_vs_base

CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
PACK_DIR = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe" / "moe_b4_2m"
OUT_ROOT = PROJECT_ROOT / "RESULTS" / "moe"

KS = (2, 4, 8, 16)
MLP_HIDDEN = 64
DISPATCHER_EPOCHS = 8
EXPERT_EPOCHS = 2
VAL_FRACTION = 0.10
SPLIT_SEED = 0
BATCH_SIZE = 1024
LR = 1e-2
LR_END = 1e-3
L1_CHUNK = 16_384
PACK_FILES = ("train_pack.pt", "slice_ids.npy", "local_rows.npy")
GRAD_FILES = ("gradients.npy", "meta.json")


def log(message: str) -> None:
    print(message, flush=True)


def build_l1(pack, model, device: torch.device) -> np.ndarray:
    n = len(pack)
    hidden = int(model.hidden_dim)
    out = np.empty((n, 2 * hidden), dtype=np.float32)
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
            out[start:end] = h.detach().float().cpu().numpy()
            if end == n or (start // L1_CHUNK) % 10 == 0:
                log(f"  l1 {end:,}/{n:,}")
    return out


def train_mlp_dispatcher(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_clusters: int,
    device: torch.device,
) -> tuple[MLPDispatcher, dict]:
    in_dim = int(x.shape[1])
    disp = MLPDispatcher(in_dim, n_clusters, hidden_dim=MLP_HIDDEN).to(device)
    opt = torch.optim.Adam(disp.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=LR_END / LR, total_iters=max(DISPATCHER_EPOCHS, 1)
    )
    x_train = torch.from_numpy(np.ascontiguousarray(x[train_idx])).to(device)
    y_train = torch.from_numpy(labels[train_idx].astype(np.int64)).to(device)
    x_val = torch.from_numpy(np.ascontiguousarray(x[val_idx])).to(device)
    y_val = torch.from_numpy(labels[val_idx].astype(np.int64)).to(device)

    history: list[dict] = []
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in disp.state_dict().items()}
    n_train = x_train.shape[0]
    for epoch in range(1, DISPATCHER_EPOCHS + 1):
        disp.train()
        t0 = time.perf_counter()
        train_loss = 0.0
        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(disp(x_train[start:end]), y_train[start:end])
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * int(end - start)
        sched.step()
        disp.eval()
        with torch.no_grad():
            val_acc = float((disp(x_val).argmax(dim=-1) == y_val).float().mean().item())
            train_acc = float((disp(x_train).argmax(dim=-1) == y_train).float().mean().item())
        history.append(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_loss": train_loss / max(n_train, 1),
                "seconds": time.perf_counter() - t0,
            }
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in disp.state_dict().items()}

    disp.load_state_dict(best_state)
    disp.eval()
    return disp, {"best_val_acc": best_acc, "history": history, "n_train": int(n_train), "n_val": int(x_val.shape[0])}


def predict_full(disp: MLPDispatcher, x: np.ndarray, device: torch.device) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.int16)
    disp.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], L1_CHUNK):
            end = min(start + L1_CHUNK, x.shape[0])
            xt = torch.from_numpy(np.ascontiguousarray(x[start:end])).to(device)
            out[start:end] = disp(xt).argmax(dim=-1).cpu().numpy().astype(np.int16)
    return out


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
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--expert-epochs", type=int, default=EXPERT_EPOCHS)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = full pack; else subsample (for smoke)")
    parser.add_argument("--max-test", type=int, default=50_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.expert_epochs = 1
        args.max_test = 1024
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
    pack, slice_ids, local_rows = load_train_pack(pack_dir)
    n = len(pack)
    log(f"train pack rows {n:,}")
    subsampled = args.max_rows > 0 and args.max_rows < n
    if subsampled:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(n, size=args.max_rows, replace=False)).astype(np.int64)
        pack = pack.index_select(torch.from_numpy(keep))
        slice_ids = slice_ids[keep]
        local_rows = local_rows[keep]
        n = len(pack)
        log(f"subsampled pack to {n:,} rows")

    log("loading cached sample head gradients")
    gradients = np.load(pack_dir / "gradients.npy", mmap_mode="r")
    if subsampled:
        gradients = np.ascontiguousarray(gradients[keep].astype(np.float32))
    else:
        gradients = np.ascontiguousarray(gradients.astype(np.float32))
    log(f"gradients matrix {gradients.shape} ({gradients.nbytes / 1e9:.2f} GB)")

    log("building L1 activations (dispatcher input)")
    l1 = build_l1(pack, base, device)
    log(f"L1 matrix {l1.shape} ({l1.nbytes / 1e9:.2f} GB)")

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n)
    n_val = max(1, int(n * VAL_FRACTION))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    for k in args.ks:
        work_dir = OUT_ROOT / f"grad_clustered_k{k}"
        work_dir.mkdir(parents=True, exist_ok=True)
        if subsampled:
            save_train_pack(work_dir, pack, slice_ids, local_rows)
            np.save(work_dir / "gradients.npy", gradients.astype(np.float16))
            meta = json.loads((pack_dir / "meta.json").read_text(encoding="utf-8"))
            meta["n_rows"] = n
            meta["max_rows"] = n
            (work_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            link_files(pack_dir, work_dir, PACK_FILES)
            link_files(pack_dir, work_dir, GRAD_FILES)

        log(f"=== K={k} | k-means on 48-d gradients ===")
        t0 = time.perf_counter()
        km = fit_minibatch_kmeans(gradients, k, seed=0)
        labels = np.asarray(km.labels_, dtype=np.int16)
        centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
        diag = cluster_diagnostics(labels, centroids, inertia=float(km.inertia_))
        diag["algorithm"] = "minibatch_kmeans"
        diag["representation"] = "head_gradient_48d"
        diag["requested_clusters"] = k
        save_cluster_run(work_dir, labels=labels, centroids=centroids, diagnostics=diag)
        log(f"  sizes {diag['sizes']} | empty={diag['empty']} ({time.perf_counter() - t0:.1f}s)")

        log(f"=== K={k} | MLP dispatcher (h={MLP_HIDDEN}) ===")
        t1 = time.perf_counter()
        disp, disp_info = train_mlp_dispatcher(
            l1, labels, train_idx=train_idx, val_idx=val_idx, n_clusters=k, device=device
        )
        log(f"  best val acc {disp_info['best_val_acc']:.4f} ({time.perf_counter() - t1:.1f}s)")
        torch.save(
            {
                "state_dict": disp.state_dict(),
                "in_dim": disp.in_dim,
                "n_clusters": k,
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
        log(f"  dispatcher labels: {np.bincount(disp_labels.astype(np.int64), minlength=k).tolist()}")

        log(f"=== K={k} | expert fine-tune (frozen L1, {args.expert_epochs} epochs) ===")
        moe = fine_tune_experts(
            base,
            work_dir,
            folders,
            device=device,
            n_clusters=k,
            expert_labels="dispatcher",
            epochs=args.expert_epochs,
            dispatcher_kind="mlp",
            dispatcher_hidden=MLP_HIDDEN,
        )

        log(f"=== K={k} | evaluation (with oracle) ===")
        metrics = evaluate_moe(
            base,
            moe,
            folders,
            work_dir,
            device=device,
            test_fraction=0.01,
            split_seed=0,
            max_test=args.max_test,
            oracle=True,
        )
        log(
            f"  K={k} eval n={metrics['n_test']:,} | base_ce={metrics['base_ce']:.5f} | "
            f"moe_ce={metrics['moe_ce']:.5f}"
            + (f" | oracle_ce={metrics['oracle_ce']:.5f}" if "oracle_ce" in metrics else "")
        )

        plots = write_plots(work_dir)
        summary = {
            "run_name": work_dir.name,
            "checkpoint": str(ckpt),
            "pack_dir": str(pack_dir),
            "n_buckets": k,
            "router": "mlp_l1_to_gradient",
            "dispatcher_hidden": MLP_HIDDEN,
            "dispatcher_val_acc": disp_info["best_val_acc"],
            "expert_labels": "dispatcher",
            "expert_epochs": args.expert_epochs,
            "train_rows": n,
            "cluster_sizes": diag["sizes"],
            "metrics": metrics,
            "plots": plots,
        }
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"wrote {work_dir / 'summary.json'}")

    log("GRADIENT_CLUSTERED_MOE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
