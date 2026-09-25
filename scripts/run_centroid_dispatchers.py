#!/usr/bin/env python3
"""GOAL.md §2 centroid dispatchers — route by argmin cosine similarity.

Two parameter-free dispatchers:
  board   nearest centroid (cosine) of the 1688-d board-state features
  l1      nearest centroid (cosine) of the 256-d accumulator (L1) activations

For each representation and B in {2, 4, 8, 16}, fit mini-batch k-means on the
1,000,000-row ``moe_b3_1m`` training subsample, then assign every position to
the nearest centroid by cosine similarity and score the routing against the
k-means (Euclidean) labels as pseudo-ground-truth (Top-1 accuracy, ARI, NMI,
macro/weighted F1).

Writes:
  RESULTS/dispatcher/centroid/stats.csv
  RESULTS/dispatcher/centroid/summary.json
  RESULTS/dispatcher/centroid/<rep>/b<B>/{centroids,labels_kmeans,labels_cosine}.npy
  RESULTS/dispatcher/centroid/<rep>/b<B>/diagnostics.json
  RESULTS/dispatcher/centroid/plots/
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
from sklearn.metrics import f1_score

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROJECT_ROOT
from tinymlinternship.features import FEATURE_DIM
from tinymlinternship.nnue.cluster import (
    assign_to_centroids_cosine,
    fit_minibatch_kmeans,
)
from tinymlinternship.nnue.clustering_eval import KMEANS_KS, fill_stm_board, overlap_scores
from tinymlinternship.nnue.moe import load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import load_train_pack

PACK_DIR = PROJECT_ROOT / "data" / "processed" / "board_eval" / "moe" / "moe_b3_1m"
CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
OUT_DIR = PROJECT_ROOT / "RESULTS" / "dispatcher" / "centroid"
BOARD_CHUNK = 4_096
L1_CHUNK = 16_384
REPS = ("board", "l1")


def log(message: str) -> None:
    print(message, flush=True)


def build_board(pack) -> np.ndarray:
    n = len(pack)
    board = np.zeros((n, 2 * FEATURE_DIM), dtype=np.float32)
    for start in range(0, n, BOARD_CHUNK):
        end = min(start + BOARD_CHUNK, n)
        batch = pack.gather(torch.arange(start, end, dtype=torch.long))
        fill_stm_board(
            board[start:end],
            batch["white_idx"].numpy(),
            batch["white_mask"].numpy(),
            batch["black_idx"].numpy(),
            batch["black_mask"].numpy(),
            batch["stm_white"].numpy(),
        )
        if end == n or (start // BOARD_CHUNK) % 25 == 0:
            log(f"  board {end:,}/{n:,}")
    return board


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


def _subsample(x: np.ndarray, n_rows: int, seed: int = 0) -> np.ndarray:
    if n_rows >= x.shape[0]:
        return x
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(x.shape[0], size=n_rows, replace=False))
    return np.ascontiguousarray(x[idx], dtype=np.float32)


def run_rep(rep: str, matrix: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    labels_store: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for b in KMEANS_KS:
        t0 = time.perf_counter()
        model = fit_minibatch_kmeans(matrix, b, seed=0)
        centroids = np.ascontiguousarray(model.cluster_centers_, dtype=np.float32)
        kmeans_labels = np.asarray(model.labels_, dtype=np.int16)
        cosine_labels = assign_to_centroids_cosine(matrix, centroids)
        seconds = time.perf_counter() - t0
        acc = float((cosine_labels == kmeans_labels).mean())
        ov = overlap_scores(cosine_labels, kmeans_labels)
        label_range = np.arange(b)
        f1_macro = float(f1_score(kmeans_labels, cosine_labels, labels=label_range, average="macro", zero_division=0))
        f1_weighted = float(f1_score(kmeans_labels, cosine_labels, labels=label_range, average="weighted", zero_division=0))
        k_sizes = np.bincount(kmeans_labels.astype(np.int64), minlength=b).astype(np.int64).tolist()
        c_sizes = np.bincount(cosine_labels.astype(np.int64), minlength=b).astype(np.int64).tolist()
        work = OUT_DIR / rep / f"b{b}"
        work.mkdir(parents=True, exist_ok=True)
        np.save(work / "centroids.npy", centroids)
        np.save(work / "labels_kmeans.npy", kmeans_labels)
        np.save(work / "labels_cosine.npy", cosine_labels)
        diag = {
            "representation": rep,
            "b": b,
            "n_rows": int(matrix.shape[0]),
            "feature_dim": int(matrix.shape[1]),
            "kmeans_sizes": k_sizes,
            "cosine_sizes": c_sizes,
            "top1_acc": acc,
            "ari": ov["ari"],
            "nmi": ov["nmi"],
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "seconds": seconds,
        }
        (work / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        labels_store[b] = (kmeans_labels, cosine_labels)
        rows.append(diag)
        log(
            f"  {rep} B={b}: acc={acc:.4f} ari={ov['ari']:.4f} nmi={ov['nmi']:.4f} "
            f"f1m={f1_macro:.4f} ({seconds:.1f}s)"
        )
    return rows


def write_stats_csv(rows: list[dict]) -> None:
    path = OUT_DIR / "stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "representation",
        "b",
        "n_rows",
        "feature_dim",
        "top1_acc",
        "ari",
        "nmi",
        "macro_f1",
        "weighted_f1",
        "kmeans_sizes",
        "cosine_sizes",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["kmeans_sizes"] = "|".join(str(v) for v in row["kmeans_sizes"])
            out["cosine_sizes"] = "|".join(str(v) for v in row["cosine_sizes"])
            writer.writerow(out)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_accuracy(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(KMEANS_KS))
    width = 0.35
    for i, rep in enumerate(REPS):
        vals = [next(r["top1_acc"] for r in rows if r["representation"] == rep and r["b"] == b) for b in KMEANS_KS]
        ax.bar(x + (i - 0.5) * width, vals, width, label=rep, color="#4c72b0" if rep == "board" else "#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in KMEANS_KS])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top-1 accuracy vs k-means")
    ax.set_title("Centroid dispatcher (argmin cosine) reproduces k-means labels")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, plots / "accuracy.png")


def plot_confusion(rows: list[dict], rep: str, confusions: dict[int, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    fig, axes = plt.subplots(1, len(KMEANS_KS), figsize=(4.2 * len(KMEANS_KS), 4.0), squeeze=False)
    image = None
    for ax, b in zip(axes[0], KMEANS_KS):
        cm = confusions[b]
        cm = cm.astype(np.float64)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        image = ax.imshow(cm, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"B={b}")
        ax.set_xlabel("k-means label")
        ax.set_ylabel("cosine label")
    axes[0, 0].set_ylabel("cosine label")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    fig.suptitle(f"{rep} · centroid dispatcher (argmin cosine) confusion")
    fig.tight_layout()
    _save(fig, plots / f"confusion_{rep}.png")


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
        log(f"subsampling to {max_rows:,} rows")
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(n, size=max_rows, replace=False)).astype(np.int64)
        pack = pack.index_select(torch.from_numpy(keep))

    model = load_dual_hidden_checkpoint(CHECKPOINT, device=device)
    log("building board features")
    board = build_board(pack)
    log("building L1 activations")
    l1 = build_l1(pack, model, device)
    del pack, model

    all_rows: list[dict] = []
    confusions: dict[str, dict[int, np.ndarray]] = {"board": {}, "l1": {}}
    for rep, matrix in (("board", board), ("l1", l1)):
        log(f"clustering + cosine routing: {rep} ({matrix.shape})")
        all_rows.extend(run_rep(rep, matrix))
        for b in KMEANS_KS:
            kmeans_labels = np.load(OUT_DIR / rep / f"b{b}" / "labels_kmeans.npy")
            cosine_labels = np.load(OUT_DIR / rep / f"b{b}" / "labels_cosine.npy")
            from sklearn.metrics import confusion_matrix

            confusions[rep][b] = confusion_matrix(kmeans_labels, cosine_labels, labels=np.arange(b))

    write_stats_csv(all_rows)
    summary = {
        "source": str(PACK_DIR),
        "checkpoint": str(CHECKPOINT),
        "n_rows": int(all_rows[0]["n_rows"]) if all_rows else 0,
        "representations": list(REPS),
        "k_values": list(KMEANS_KS),
        "routing": "argmin cosine similarity",
        "pseudo_ground_truth": "mini-batch k-means (Euclidean) labels",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_accuracy(all_rows)
    for rep in REPS:
        plot_confusion(all_rows, rep, confusions[rep])
    log(f"wrote {OUT_DIR / 'stats.csv'}")
    log("CENTROID_DISPATCHERS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
