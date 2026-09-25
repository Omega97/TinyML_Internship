#!/usr/bin/env python3
"""GOAL.md §2 non-neural L1 dispatchers — decision tree and XGBoost.

Pseudo-ground-truth buckets are mini-batch k-means on the 256-d accumulator
(L1) activations of the reference NNUE (W=128). Two non-neural routers map the
same L1 activations to those bucket IDs:

  tree    sklearn DecisionTreeClassifier (max_depth=12)
  xgb     xgboost.XGBClassifier (multi:softprob, 50 trees, depth 6)

For each B in {2, 4, 8, 16}, each router is trained on a 90/10 split and scored
on the held-out 10% (Top-1/Top-2 accuracy, macro and weighted F1, ARI, NMI vs.
the k-means labels, plus majority-dummy and chance).

Writes:
  RESULTS/dispatcher/tree/clusters/b<B>/{labels,centroids}.npy
  RESULTS/dispatcher/tree/<kind>/b<B>/{model,labels_pred.npy,diagnostics.json}
  RESULTS/dispatcher/tree/stats.csv
  RESULTS/dispatcher/tree/summary.json
  RESULTS/dispatcher/tree/plots/
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

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROJECT_ROOT
from tinymlinternship.nnue.cluster import fit_minibatch_kmeans
from tinymlinternship.nnue.clustering_eval import overlap_scores
from tinymlinternship.nnue.moe import load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import load_train_pack

PACK_DIR = PROJECT_ROOT / "data" / "processed" / "board_eval" / "moe" / "moe_b3_1m"
CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
OUT_DIR = PROJECT_ROOT / "RESULTS" / "dispatcher" / "tree"
L1_CHUNK = 16_384
KINDS = ("tree", "xgb")
KS = (2, 4, 8, 16)
VAL_FRACTION = 0.10
SPLIT_SEED = 0


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


def make_model(kind: str, n_clusters: int):
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=12, random_state=SPLIT_SEED)
    if kind == "xgb":
        return XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=SPLIT_SEED,
            objective="multi:softprob",
            num_class=n_clusters,
            eval_metric="mlogloss",
        )
    raise ValueError(f"unknown router kind {kind!r}")


def topk_accuracy(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-proba, axis=1)[:, : min(int(k), proba.shape[1])]
    hit = (top == y[:, None]).any(axis=1)
    return float(hit.mean())


def fit_model(kind: str, model, x_train, y_train, x_val, y_val, n_clusters: int) -> dict:
    t0 = time.perf_counter()
    if kind == "xgb":
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        model.fit(x_train, y_train)
    seconds = time.perf_counter() - t0
    proba = model.predict_proba(x_val)
    pred = np.asarray(proba.argmax(axis=1), dtype=np.int16)
    top1 = topk_accuracy(proba, y_val, 1)
    top2 = topk_accuracy(proba, y_val, 2)
    label_range = np.arange(n_clusters)
    f1_macro = float(f1_score(y_val, pred, labels=label_range, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_val, pred, labels=label_range, average="weighted", zero_division=0))
    ov = overlap_scores(pred, y_val)
    return {
        "model": model,
        "top1": top1,
        "top2": top2,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "ari": ov["ari"],
        "nmi": ov["nmi"],
        "seconds": seconds,
    }


def write_stats_csv(rows: list[dict]) -> None:
    path = OUT_DIR / "stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind",
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
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(KS))
    width = 0.28
    colors = {"tree": "#4c72b0", "xgb": "#dd8452"}
    for i, kind in enumerate(KINDS):
        vals = [next(r["top1"] for r in rows if r["kind"] == kind and r["b"] == b) for b in KS]
        ax.bar(x + (i - 0.5) * width, vals, width, label=kind, color=colors[kind])
    ax.plot(x, [1.0 / b for b in KS], marker="o", color="#8a8f98", label="chance 1/B")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in KS])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top-1 accuracy (validation)")
    ax.set_title("Non-neural L1 dispatchers vs k-means labels")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, plots / "accuracy.png")


def plot_confusion(rows: list[dict], confusions: dict[str, dict[int, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    for kind in KINDS:
        fig, axes = plt.subplots(1, len(KS), figsize=(4.2 * len(KS), 4.0), squeeze=False)
        image = None
        for ax, b in zip(axes[0], KS):
            cm = confusions[kind][b].astype(np.float64)
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
            image = ax.imshow(cm, vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_title(f"B={b}")
            ax.set_xlabel("predicted")
            ax.set_ylabel("k-means label")
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        fig.suptitle(f"{kind} L1 dispatcher — normalized confusion matrix")
        fig.tight_layout()
        _save(fig, plots / f"confusion_{kind}.png")


def plot_pr(rows: list[dict], pr_data: dict[str, dict[int, dict]]) -> None:
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    for kind in KINDS:
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), squeeze=False)
        cmap = plt.get_cmap("tab20")
        for ax, b in zip(axes.flat, KS):
            for cls, (prec, rec) in pr_data[kind][b].items():
                ax.plot(rec, prec, color=cmap(cls % 20), lw=1.2, label=f"cls {cls}")
            ax.set_title(f"B={b}")
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{kind} L1 dispatcher — per-class precision-recall")
        fig.tight_layout()
        _save(fig, plots / f"pr_{kind}.png")


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

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n)
    n_val = max(1, int(n * VAL_FRACTION))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    all_rows: list[dict] = []
    confusions: dict[str, dict[int, np.ndarray]] = {k: {} for k in KINDS}
    pr_data: dict[str, dict[int, dict]] = {k: {} for k in KINDS}

    for b in KS:
        log(f"k-means B={b} on L1")
        km = fit_minibatch_kmeans(l1, b, seed=0)
        labels = np.asarray(km.labels_, dtype=np.int16)
        centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
        cluster_dir = OUT_DIR / "clusters" / f"b{b}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        np.save(cluster_dir / "labels.npy", labels)
        np.save(cluster_dir / "centroids.npy", centroids)

        x_train = np.ascontiguousarray(l1[train_idx])
        y_train = labels[train_idx].astype(np.int64)
        x_val = np.ascontiguousarray(l1[val_idx])
        y_val = labels[val_idx].astype(np.int64)
        majority = int(np.bincount(y_train).argmax())
        dummy_val_acc = float((y_val == majority).mean())

        for kind in KINDS:
            log(f"training {kind} router B={b}")
            result = fit_model(kind, make_model(kind, b), x_train, y_train, x_val, y_val, b)
            kind_dir = OUT_DIR / kind / f"b{b}"
            kind_dir.mkdir(parents=True, exist_ok=True)
            if kind == "tree":
                joblib.dump(result["model"], kind_dir / "model.joblib")
            else:
                result["model"].save_model(str(kind_dir / "model.json"))
            proba = result["model"].predict_proba(x_val)
            pred = np.asarray(proba.argmax(axis=1), dtype=np.int16)
            np.save(kind_dir / "labels_pred.npy", pred)
            from sklearn.metrics import confusion_matrix

            confusions[kind][b] = confusion_matrix(y_val, pred, labels=np.arange(b))
            pr_b: dict[int, tuple] = {}
            for cls in range(b):
                from sklearn.metrics import precision_recall_curve

                yb = (y_val == cls).astype(np.int32)
                prec, rec, _ = precision_recall_curve(yb, proba[:, cls])
                pr_b[cls] = (prec, rec)
            pr_data[kind][b] = pr_b
            diag = {
                "kind": kind,
                "b": b,
                "n_train": int(train_idx.shape[0]),
                "n_val": int(val_idx.shape[0]),
                "top1": result["top1"],
                "top2": result["top2"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
                "ari": result["ari"],
                "nmi": result["nmi"],
                "chance_acc": 1.0 / b,
                "dummy_val_acc": dummy_val_acc,
                "dummy_majority_label": majority,
                "seconds": result["seconds"],
            }
            (kind_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
            all_rows.append(diag)
            log(
                f"  {kind} B={b}: top1={result['top1']:.4f} ari={result['ari']:.4f} ({result['seconds']:.1f}s)"
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
                "kinds": list(KINDS),
                "k_values": list(KS),
                "pseudo_ground_truth": "mini-batch k-means on 256-d L1 activations",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_accuracy(all_rows)
    plot_confusion(all_rows, confusions)
    plot_pr(all_rows, pr_data)
    log(f"wrote {OUT_DIR / 'stats.csv'}")
    log("TREE_DISPATCHERS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
