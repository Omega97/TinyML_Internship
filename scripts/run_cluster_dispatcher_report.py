#!/usr/bin/env python3
"""Cluster the 2M reference gradients at B=2,4,8 and train one linear dispatcher each.

Writes:
  RESULTS/clustering/b{B}/   labels, centroids, diagnostics
  RESULTS/clustering/stats.csv and plots/
  RESULTS/dispatcher/b{B}/   dispatcher checkpoint, history, predicted labels
  RESULTS/dispatcher/stats.csv and plots/
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROJECT_ROOT
from tinymlinternship.nnue.moe_pipeline import (
    cluster_gradients,
    configure_torch,
    link_gradient_cache,
    load_base_on_device,
    train_dispatcher,
)

REF_CKPT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
REF_GRAD_DIR = PROJECT_ROOT / "data" / "processed" / "board_eval" / "moe" / "moe_b4_2m"
CLUSTER_ROOT = PROJECT_ROOT / "RESULTS" / "clustering"
DISPATCH_ROOT = PROJECT_ROOT / "RESULTS" / "dispatcher"
BUCKETS = (2, 4, 8)
SILHOUETTE_SAMPLES = 10_000
VAL_FRACTION = 0.10
SPLIT_SEED = 0
_COLORS = {2: "#4c72b0", 4: "#dd8452", 8: "#55a868"}


def _offdiag(mat: np.ndarray) -> np.ndarray:
    copied = np.array(mat, dtype=np.float64, copy=True)
    np.fill_diagonal(copied, np.nan)
    return copied


def _silhouette(grads: np.ndarray, labels: np.ndarray, n_samples: int, seed: int) -> float:
    from sklearn.metrics import silhouette_score

    n = int(labels.shape[0])
    take = min(int(n_samples), n)
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(n, size=take, replace=False)
    sample = np.ascontiguousarray(grads[idx], dtype=np.float32)
    sample_labels = np.asarray(labels[idx], dtype=np.int64)
    if np.unique(sample_labels).size < 2:
        raise RuntimeError("silhouette subsample collapsed to one cluster")
    return float(silhouette_score(sample, sample_labels, metric="euclidean"))


def _dummy_majority(labels: np.ndarray, val_fraction: float, seed: int) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    n = int(labels.shape[0])
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n)
    n_val = max(1, int(n * float(val_fraction)))
    val = labels[perm[:n_val]]
    train = labels[perm[n_val:]]
    majority = int(np.bincount(train).argmax())
    return {
        "dummy_majority_label": majority,
        "dummy_train_acc": float(np.mean(train == majority)),
        "dummy_val_acc": float(np.mean(val == majority)),
    }


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _copy_named(src: Path, dest: Path, names: tuple[str, ...]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = src / name
        if target.is_file():
            shutil.copy2(target, dest / name)


def run_one(b: int, base, device: torch.device) -> dict:
    work = Path(f"/tmp/tinyml_cluster_disp_b{b}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    link_gradient_cache(REF_GRAD_DIR, work)
    print(f"=== B={b} clustering ===", flush=True)
    t0 = time.perf_counter()
    diag = cluster_gradients(work, n_clusters=b, algorithm="minibatch_kmeans", seed=0)
    cluster_seconds = time.perf_counter() - t0
    grads = np.load(work / "gradients.npy", mmap_mode="r")
    labels = np.load(work / "labels.npy")
    silhouette = _silhouette(grads, labels, SILHOUETTE_SAMPLES, seed=0)
    diag["silhouette"] = silhouette
    diag["silhouette_sample_size"] = SILHOUETTE_SAMPLES
    (work / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"B={b} silhouette={silhouette:.4f}", flush=True)

    print(f"=== B={b} dispatcher ===", flush=True)
    t1 = time.perf_counter()
    trained = train_dispatcher(
        base,
        work,
        [],
        device=device,
        n_clusters=int(diag["n_clusters"]),
        epochs=8,
        val_fraction=VAL_FRACTION,
        seed=SPLIT_SEED,
    )
    dispatcher_seconds = time.perf_counter() - t1
    dummy = _dummy_majority(labels, VAL_FRACTION, SPLIT_SEED)
    cosine = _offdiag(diag["centroid_cosine"])
    distance = 1.0 - cosine
    history = trained["history"]
    last = history[-1]
    record = {
        "b": b,
        "n_rows": int(diag["n_rows"]),
        "sizes": [int(v) for v in diag["sizes"]],
        "inertia": float(diag["inertia"]),
        "empty_clusters": int(diag["empty"]),
        "min_size": int(diag["min_size"]),
        "max_size": int(diag["max_size"]),
        "silhouette": silhouette,
        "centroid_cosine": diag["centroid_cosine"],
        "centroid_cosine_distance_mean": float(np.nanmean(distance)),
        "centroid_cosine_distance_min": float(np.nanmin(distance)),
        "centroid_cosine_mean_offdiag": float(np.nanmean(cosine)),
        "centroid_cosine_min_offdiag": float(np.nanmin(cosine)),
        "cluster_seconds": cluster_seconds,
        "train_acc": float(last["train_acc"]),
        "val_acc_last": float(last["val_acc"]),
        "val_acc": float(trained["best_val_acc"]),
        "chance_acc": 1.0 / float(diag["n_clusters"]),
        "n_train": int(trained["n_train"]),
        "n_val": int(trained["n_val"]),
        "dispatcher_seconds": dispatcher_seconds,
        "history": history,
        **dummy,
    }
    record["val_acc_minus_dummy"] = record["val_acc"] - record["dummy_val_acc"]
    _copy_named(
        work,
        CLUSTER_ROOT / f"b{b}",
        ("labels.npy", "centroids.npy", "diagnostics.json"),
    )
    _copy_named(
        work,
        DISPATCH_ROOT / f"b{b}",
        ("dispatcher.pt", "dispatcher_history.json", "labels_dispatcher.npy"),
    )
    shutil.rmtree(work)
    print(
        f"B={b} val_acc={record['val_acc']:.4f} dummy={record['dummy_val_acc']:.4f}",
        flush=True,
    )
    return record


def _write_clustering_csv(records: list[dict]) -> None:
    max_b = max(len(row["sizes"]) for row in records)
    fields = [
        "b",
        "n_rows",
        "inertia",
        "silhouette",
        "silhouette_sample_size",
        "min_size",
        "max_size",
        "empty_clusters",
        "centroid_cosine_distance_mean",
        "centroid_cosine_distance_min",
        "centroid_cosine_mean_offdiag",
        "centroid_cosine_min_offdiag",
        "seconds",
    ] + [f"size_{i}" for i in range(max_b)]
    path = CLUSTER_ROOT / "stats.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            out = {
                "b": row["b"],
                "n_rows": row["n_rows"],
                "inertia": row["inertia"],
                "silhouette": row["silhouette"],
                "silhouette_sample_size": SILHOUETTE_SAMPLES,
                "min_size": row["min_size"],
                "max_size": row["max_size"],
                "empty_clusters": row["empty_clusters"],
                "centroid_cosine_distance_mean": row["centroid_cosine_distance_mean"],
                "centroid_cosine_distance_min": row["centroid_cosine_distance_min"],
                "centroid_cosine_mean_offdiag": row["centroid_cosine_mean_offdiag"],
                "centroid_cosine_min_offdiag": row["centroid_cosine_min_offdiag"],
                "seconds": row["cluster_seconds"],
            }
            for i, size in enumerate(row["sizes"]):
                out[f"size_{i}"] = size
            writer.writerow(out)


def _write_dispatcher_csv(records: list[dict]) -> None:
    fields = [
        "b",
        "epochs",
        "train_acc",
        "val_acc",
        "val_acc_last",
        "chance_acc",
        "dummy_majority_label",
        "dummy_train_acc",
        "dummy_val_acc",
        "val_acc_minus_dummy",
        "n_train",
        "n_val",
        "seconds",
    ]
    path = DISPATCH_ROOT / "stats.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "b": row["b"],
                    "epochs": len(row["history"]),
                    "train_acc": row["train_acc"],
                    "val_acc": row["val_acc"],
                    "val_acc_last": row["val_acc_last"],
                    "chance_acc": row["chance_acc"],
                    "dummy_majority_label": row["dummy_majority_label"],
                    "dummy_train_acc": row["dummy_train_acc"],
                    "dummy_val_acc": row["dummy_val_acc"],
                    "val_acc_minus_dummy": row["val_acc_minus_dummy"],
                    "n_train": row["n_train"],
                    "n_val": row["n_val"],
                    "seconds": row["dispatcher_seconds"],
                }
            )


def _plot_clustering(records: list[dict]) -> None:
    import matplotlib.pyplot as plt

    plots = CLUSTER_ROOT / "plots"
    fig, axes = plt.subplots(1, len(records), figsize=(11.6, 4.2), sharey=True)
    for ax, row in zip(axes, records):
        sizes = row["sizes"]
        ax.bar(np.arange(len(sizes)), sizes, color=_COLORS[row["b"]])
        ax.set_title(f"B = {row['b']}")
        ax.set_xlabel("bucket")
        ax.set_xticks(np.arange(len(sizes)))
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("positions")
    fig.suptitle("Mini-batch k-means cluster sizes (2,000,000 positions)")
    fig.tight_layout()
    _save(fig, plots / "cluster_sizes.png")

    bs = [row["b"] for row in records]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(bs, [row["inertia"] for row in records], marker="o", color="#4c72b0")
    ax.set_xticks(bs)
    ax.set_xlabel("B")
    ax.set_ylabel("inertia")
    ax.set_title("Within-cluster sum of squares")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, plots / "inertia.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(bs, [row["silhouette"] for row in records], marker="o", color="#dd8452")
    ax.set_xticks(bs)
    ax.set_xlabel("B")
    ax.set_ylabel("silhouette")
    ax.set_title(f"Silhouette ({SILHOUETTE_SAMPLES:,} row sample)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, plots / "silhouette.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        bs,
        [row["centroid_cosine_distance_mean"] for row in records],
        marker="o",
        color="#4c72b0",
        label="mean",
    )
    ax.plot(
        bs,
        [row["centroid_cosine_distance_min"] for row in records],
        marker="s",
        color="#dd8452",
        label="minimum",
    )
    ax.set_xticks(bs)
    ax.set_xlabel("B")
    ax.set_ylabel("cosine distance")
    ax.set_title("Pairwise centroid cosine distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, plots / "centroid_distance.png")

    fig = plt.figure(figsize=(13.6, 4.3))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.42)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    image = None
    for ax, row in zip(axes, records):
        mat = np.asarray(row["centroid_cosine"], dtype=np.float64)
        image = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ticks = list(range(mat.shape[0]))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(i) for i in range(1, mat.shape[0] + 1)])
        ax.set_yticklabels([str(i) for i in range(1, mat.shape[0] + 1)])
        ax.set_title(f"B = {row['b']}")
        ax.set_xlabel("cluster")
    axes[0].set_ylabel("cluster")
    fig.colorbar(image, cax=fig.add_subplot(grid[0, 3]))
    fig.suptitle("Centroid cosine")
    fig.subplots_adjust(top=0.84)
    _save(fig, plots / "centroid_cosine.png")


def _plot_dispatcher(records: list[dict]) -> None:
    import matplotlib.pyplot as plt

    plots = DISPATCH_ROOT / "plots"
    bs = [str(row["b"]) for row in records]
    x = np.arange(len(records))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    series = (
        ("val_acc", "dispatcher", "#4c72b0"),
        ("dummy_val_acc", "majority dummy", "#dd8452"),
        ("chance_acc", "chance 1/B", "#8a8f98"),
    )
    for i, (key, label, color) in enumerate(series):
        offset = (i - 1) * width
        vals = [row[key] for row in records]
        ax.bar(x + offset, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B = {b}" for b in bs])
    ax.set_ylabel("validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Linear dispatcher vs majority-class dummy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, plots / "accuracy.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for row in records:
        epochs = [step["epoch"] for step in row["history"]]
        color = _COLORS[row["b"]]
        ax.plot(
            epochs,
            [step["val_acc"] for step in row["history"]],
            marker="o",
            color=color,
            label=f"B = {row['b']} val",
        )
        ax.plot(
            epochs,
            [step["train_acc"] for step in row["history"]],
            marker=None,
            linestyle="--",
            color=color,
            alpha=0.75,
            label=f"B = {row['b']} train",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Dispatcher accuracy during training")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, plots / "accuracy_curves.png")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch(device)
    print(f"device={device}", flush=True)
    base = load_base_on_device(REF_CKPT, device)
    records = [run_one(b, base, device) for b in BUCKETS]
    _write_clustering_csv(records)
    _write_dispatcher_csv(records)
    _plot_clustering(records)
    _plot_dispatcher(records)
    print(f"clustering stats: {CLUSTER_ROOT / 'stats.csv'}", flush=True)
    print(f"dispatcher stats: {DISPATCH_ROOT / 'stats.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
