#!/usr/bin/env python3
"""GOAL.md §2 single-hidden MLP dispatchers, swept over hidden width and target.

Two pseudo-ground-truth bucket sources on the 1M-row ``moe_b3_1m`` subsample:

  l1        mini-batch k-means on the 256-d accumulator (L1) activations
  gradient  mini-batch k-means on the 48-d sample head-gradient projection

For each target, a single-hidden MLP router ``L1 → h → Bucket ID`` is trained
with ``h ∈ {32, 64, 128}`` and evaluated at ``B ∈ {2, 4, 8, 16}`` (90/10 split),
scored by Top-1/Top-2 accuracy, macro/weighted F1, ARI and NMI vs. the k-means
labels, plus majority-dummy and chance.

Writes:
  RESULTS/dispatcher/mlp/clusters/<target>/b<B>/{labels,centroids}.npy
  RESULTS/dispatcher/mlp/<target>/h<h>/b<B>/{dispatcher.pt,history.json,labels_pred.npy,diagnostics.json}
  RESULTS/dispatcher/mlp/stats.csv
  RESULTS/dispatcher/mlp/summary.json
  RESULTS/dispatcher/mlp/plots/
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import f1_score

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROJECT_ROOT
from tinymlinternship.nnue.cluster import fit_minibatch_kmeans
from tinymlinternship.nnue.clustering_eval import overlap_scores
from tinymlinternship.nnue.moe import MLPDispatcher, load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import load_train_pack

PACK_DIR = PROJECT_ROOT / "data" / "processed" / "board_eval" / "moe" / "moe_b3_1m"
CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
OUT_DIR = PROJECT_ROOT / "RESULTS" / "dispatcher" / "mlp"
L1_CHUNK = 16_384
TARGETS = ("l1", "gradient")
HIDDEN_SIZES = (32, 64, 128)
KS = (2, 4, 8, 16)
VAL_FRACTION = 0.10
SPLIT_SEED = 0
EPOCHS = 8
BATCH_SIZE = 1024
LR = 1e-2
LR_END = 1e-3


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


def load_gradients(pack_dir: Path) -> np.ndarray:
    g = np.load(pack_dir / "gradients.npy", mmap_mode="r")
    return np.ascontiguousarray(g.astype(np.float32))


def topk_accuracy(probs: torch.Tensor, y: torch.Tensor, k: int) -> float:
    top = probs.topk(min(int(k), probs.shape[1]), dim=-1).indices
    hit = (top == y.unsqueeze(1)).any(dim=1)
    return float(hit.float().mean().item())


def train_mlp(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_clusters: int,
    hidden_dim: int,
    device: torch.device,
) -> dict:
    in_dim = int(x.shape[1])
    disp = MLPDispatcher(in_dim, n_clusters, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(disp.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=LR_END / LR, total_iters=max(EPOCHS, 1)
    )
    x_train = torch.from_numpy(np.ascontiguousarray(x[train_idx])).to(device)
    y_train = torch.from_numpy(labels[train_idx].astype(np.int64)).to(device)
    x_val = torch.from_numpy(np.ascontiguousarray(x[val_idx])).to(device)
    y_val = torch.from_numpy(labels[val_idx].astype(np.int64)).to(device)

    history: list[dict] = []
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in disp.state_dict().items()}
    n_train = x_train.shape[0]
    for epoch in range(1, EPOCHS + 1):
        disp.train()
        t0 = time.perf_counter()
        train_loss = 0.0
        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            xb = x_train[start:end]
            yb = y_train[start:end]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(disp(xb), yb)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * int(yb.shape[0])
        sched.step()
        disp.eval()
        with torch.no_grad():
            probs = F.softmax(disp(x_val), dim=-1)
            val_acc = float((probs.argmax(dim=-1) == y_val).float().mean().item())
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
    with torch.no_grad():
        probs = F.softmax(disp(x_val), dim=-1)
        pred = probs.argmax(dim=-1)
        top1 = topk_accuracy(probs, y_val, 1)
        top2 = topk_accuracy(probs, y_val, 2)
        pred_np = pred.cpu().numpy().astype(np.int16)
        y_np = y_val.cpu().numpy().astype(np.int16)
    label_range = np.arange(n_clusters)
    f1_macro = float(f1_score(y_np, pred_np, labels=label_range, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_np, pred_np, labels=label_range, average="weighted", zero_division=0))
    ov = overlap_scores(pred_np, y_np)
    return {
        "dispatcher": disp,
        "best_val_acc": best_acc,
        "top1": top1,
        "top2": top2,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "ari": ov["ari"],
        "nmi": ov["nmi"],
        "history": history,
        "n_train": int(train_idx.shape[0]),
        "n_val": int(val_idx.shape[0]),
    }


def write_stats_csv(rows: list[dict]) -> None:
    path = OUT_DIR / "stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target",
        "h",
        "b",
        "n_train",
        "n_val",
        "top1",
        "top2",
        "macro_f1",
        "weighted_f1",
        "ari",
        "nmi",
        "chance_acc",
        "dummy_val_acc",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_accuracy(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    colors = {32: "#4c72b0", 64: "#dd8452", 128: "#55a868"}
    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        x = np.arange(len(KS))
        for h in HIDDEN_SIZES:
            vals = [
                next(r["top1"] for r in rows if r["target"] == target and r["h"] == h and r["b"] == b)
                for b in KS
            ]
            ax.plot(x, vals, marker="o", color=colors[h], label=f"h={h}")
        ax.plot(x, [1.0 / b for b in KS], marker="s", color="#8a8f98", linestyle="--", label="chance 1/B")
        ax.set_xticks(x)
        ax.set_xticklabels([f"B={b}" for b in KS])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("B")
        ax.set_ylabel("Top-1 accuracy (validation)")
        ax.set_title(f"MLP dispatcher → {target} k-means buckets")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _save(fig, plots / f"accuracy_{target}.png")


def plot_confusion(rows: list[dict], confusions: dict[str, dict[int, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    for target in TARGETS:
        fig, axes = plt.subplots(1, len(KS), figsize=(4.2 * len(KS), 4.0), squeeze=False)
        image = None
        for ax, b in zip(axes[0], KS):
            cm = confusions[target][b].astype(np.float64)
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
            image = ax.imshow(cm, vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_title(f"B={b}")
            ax.set_xlabel("predicted")
            ax.set_ylabel("k-means label")
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        fig.suptitle(f"{target} · MLP h=64 dispatcher — normalized confusion matrix")
        fig.tight_layout()
        _save(fig, plots / f"confusion_{target}.png")


def plot_pr(rows: list[dict], pr_data: dict[str, dict[int, dict]]) -> None:
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    for target in TARGETS:
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), squeeze=False)
        cmap = plt.get_cmap("tab20")
        for ax, b in zip(axes.flat, KS):
            for cls, (prec, rec) in pr_data[target][b].items():
                ax.plot(rec, prec, color=cmap(cls % 20), lw=1.2, label=f"cls {cls}")
            ax.set_title(f"B={b}")
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{target} · MLP h=64 dispatcher — per-class precision-recall")
        fig.tight_layout()
        _save(fig, plots / f"pr_{target}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    max_rows = args.max_rows
    if args.smoke and max_rows is None:
        max_rows = 20_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} | pack={PACK_DIR}")
    pack, _slice_ids, _local_rows = load_train_pack(PACK_DIR)
    n = len(pack)
    log(f"rows {n:,}")
    if max_rows is not None and max_rows < n:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(n, size=max_rows, replace=False)).astype(np.int64)
        pack = pack.index_select(torch.from_numpy(keep))

    model = load_dual_hidden_checkpoint(CHECKPOINT, device=device)
    log("building L1 activations")
    l1 = build_l1(pack, model, device)
    del pack, model
    n = int(l1.shape[0])

    log("loading sample gradients")
    gradients = load_gradients(PACK_DIR)
    if gradients.shape[0] != n:
        if max_rows is not None and gradients.shape[0] >= n:
            gradients = gradients[:n]
        else:
            raise ValueError(f"gradient rows {gradients.shape[0]} != sample rows {n}")

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n)
    n_val = max(1, int(n * VAL_FRACTION))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    features = {"l1": l1, "gradient": gradients}
    all_rows: list[dict] = []
    confusions: dict[str, dict[int, np.ndarray]] = {t: {} for t in TARGETS}
    pr_data: dict[str, dict[int, dict]] = {t: {} for t in TARGETS}

    for target in TARGETS:
        x_target = features[target]
        for b in KS:
            log(f"{target} k-means B={b}")
            km = fit_minibatch_kmeans(x_target, b, seed=0)
            labels = np.asarray(km.labels_, dtype=np.int16)
            centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
            cluster_dir = OUT_DIR / "clusters" / target / f"b{b}"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            np.save(cluster_dir / "labels.npy", labels)
            np.save(cluster_dir / "centroids.npy", centroids)

            train_labels = labels[train_idx].astype(np.int64)
            majority = int(np.bincount(train_labels).argmax())
            dummy_val_acc = float((labels[val_idx] == majority).mean())

            for h in HIDDEN_SIZES:
                log(f"training MLP h={h} → {target} B={b}")
                t1 = time.perf_counter()
                result = train_mlp(
                    l1, labels, train_idx=train_idx, val_idx=val_idx, n_clusters=b, hidden_dim=h, device=device
                )
                seconds = time.perf_counter() - t1
                disp = result["dispatcher"]
                run_dir = OUT_DIR / target / f"h{h}" / f"b{b}"
                run_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": disp.state_dict(),
                        "in_dim": disp.in_dim,
                        "n_clusters": b,
                        "hidden_dim": disp.hidden_dim,
                        "val_acc": result["best_val_acc"],
                    },
                    run_dir / "dispatcher.pt",
                )
                (run_dir / "history.json").write_text(
                    json.dumps({"best_val_acc": result["best_val_acc"], "history": result["history"]}, indent=2),
                    encoding="utf-8",
                )
                with torch.no_grad():
                    xt = torch.from_numpy(np.ascontiguousarray(l1[val_idx])).to(device)
                    probs = F.softmax(disp(xt), dim=-1)
                    pred = probs.argmax(dim=-1).cpu().numpy().astype(np.int16)
                    probs_np = probs.cpu().numpy()
                np.save(run_dir / "labels_pred.npy", pred)
                if h == 64:
                    from sklearn.metrics import confusion_matrix

                    confusions[target][b] = confusion_matrix(labels[val_idx], pred, labels=np.arange(b))
                    pr_b: dict[int, tuple] = {}
                    for cls in range(b):
                        from sklearn.metrics import precision_recall_curve

                        yb = (labels[val_idx] == cls).astype(np.int32)
                        prec, rec, _ = precision_recall_curve(yb, probs_np[:, cls])
                        pr_b[cls] = (prec, rec)
                    pr_data[target][b] = pr_b
                diag = {
                    "target": target,
                    "h": h,
                    "b": b,
                    "n_train": result["n_train"],
                    "n_val": result["n_val"],
                    "top1": result["top1"],
                    "top2": result["top2"],
                    "macro_f1": result["macro_f1"],
                    "weighted_f1": result["weighted_f1"],
                    "ari": result["ari"],
                    "nmi": result["nmi"],
                    "chance_acc": 1.0 / b,
                    "dummy_val_acc": dummy_val_acc,
                    "dummy_majority_label": majority,
                    "seconds": seconds,
                }
                (run_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
                all_rows.append(diag)
                log(
                    f"  {target} h={h} B={b}: top1={result['top1']:.4f} ari={result['ari']:.4f} ({seconds:.1f}s)"
                )

    write_stats_csv(all_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "source": str(PACK_DIR),
                "checkpoint": str(CHECKPOINT),
                "n_rows": n,
                "n_val": int(n_val),
                "val_fraction": VAL_FRACTION,
                "split_seed": SPLIT_SEED,
                "epochs": EPOCHS,
                "targets": list(TARGETS),
                "hidden_sizes": list(HIDDEN_SIZES),
                "k_values": list(KS),
                "pseudo_ground_truth": {
                    "l1": "mini-batch k-means on 256-d L1 activations",
                    "gradient": "mini-batch k-means on 48-d sample head-gradient projection",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_accuracy(all_rows)
    plot_confusion(all_rows, confusions)
    plot_pr(all_rows, pr_data)
    log(f"wrote {OUT_DIR / 'stats.csv'}")
    log("MLP_DISPATCHERS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
