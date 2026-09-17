"""Thesis-style plots for the sample-gradient MoE pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

EPS = 1e-8


def _require_mpl():
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt = _require_mpl()
    plt.close(fig)
    return path


# 90% transparent points. Discrete 4-color palette (amber = #FFBF00).
_CLUSTER_POINT_ALPHA = 0.10
_CLUSTER_COLORS_4 = ("red", "#FFBF00", "green", "blue")


def _discrete_cluster_cmap(n_clusters: int):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    n = max(int(n_clusters), 1)
    if n == 4:
        colors = list(_CLUSTER_COLORS_4)
    else:
        import matplotlib.pyplot as plt

        tab = plt.get_cmap("tab10")
        colors = [tab(i % 10) for i in range(n)]
    cmap = ListedColormap(colors, name="cluster_discrete")
    bounds = np.arange(n + 1, dtype=np.float64) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def _scatter_clusters(ax, xs, ys, labels: np.ndarray):
    from matplotlib.cm import ScalarMappable

    n_clusters = int(np.max(labels) + 1) if labels.size else 1
    cmap, norm, bounds = _discrete_cluster_cmap(n_clusters)
    ax.scatter(
        xs,
        ys,
        c=labels,
        s=8,
        cmap=cmap,
        norm=norm,
        alpha=_CLUSTER_POINT_ALPHA,
        linewidths=0.4,
        edgecolors="face",
    )
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = ax.figure.colorbar(
        mappable,
        ax=ax,
        label="cluster",
        ticks=list(range(n_clusters)),
        boundaries=bounds,
        spacing="uniform",
    )
    cbar.set_ticklabels([str(i) for i in range(n_clusters)])
    return cbar


def plot_cluster_pca(
    gradients: np.ndarray,
    labels: np.ndarray,
    path: Path,
    *,
    max_points: int = 20_000,
    seed: int = 0,
    title: str = "Sample-gradient PCA",
) -> Path:
    from sklearn.decomposition import PCA

    plt = _require_mpl()
    x, y = _subsample_xy(gradients, labels, max_points, seed)
    coords = PCA(n_components=2, random_state=int(seed)).fit_transform(x)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _scatter_clusters(ax, coords[:, 0], coords[:, 1], y)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_cluster_tsne(
    gradients: np.ndarray,
    labels: np.ndarray,
    path: Path,
    *,
    max_points: int = 5_000,
    seed: int = 0,
    title: str = "Sample-gradient t-SNE",
) -> Path:
    from sklearn.manifold import TSNE

    plt = _require_mpl()
    x, y = _subsample_xy(gradients, labels, max_points, seed)
    coords = TSNE(
        n_components=2,
        perplexity=min(30, max(5, x.shape[0] // 10)),
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
    ).fit_transform(x)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _scatter_clusters(ax, coords[:, 0], coords[:, 1], y)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_cluster_sizes(
    kmeans_sizes: Sequence[int],
    dispatcher_sizes: Sequence[int] | None,
    path: Path,
    *,
    title: str = "Bucket sizes",
) -> Path:
    plt = _require_mpl()
    kmeans_sizes = list(kmeans_sizes)
    b = len(kmeans_sizes)
    x = np.arange(b)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.38 if dispatcher_sizes is not None else 0.6
    ax.bar(x - (width / 2 if dispatcher_sizes is not None else 0), kmeans_sizes, width, label="k-means")
    if dispatcher_sizes is not None:
        ax.bar(x + width / 2, list(dispatcher_sizes), width, label="dispatcher")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(b)])
    ax.set_xlabel("bucket")
    ax.set_ylabel("positions")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_centroid_cosine(cosine: np.ndarray, path: Path, *, title: str = "Centroid cosine") -> Path:
    plt = _require_mpl()
    mat = np.asarray(cosine, dtype=np.float64)
    b = int(mat.shape[0])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ticks = list(range(b))
    labels = [str(i) for i in range(1, b + 1)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("cluster")
    ax.set_ylabel("cluster")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return _save(fig, path)


def plot_dispatcher_acc(
    history: list[dict[str, Any]],
    path: Path,
    *,
    title: str = "Dispatcher accuracy",
) -> Path:
    plt = _require_mpl()
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if "train_acc" in history[0]:
        ax.plot(epochs, [row["train_acc"] for row in history], marker="o", label="train acc")
    if "val_acc" in history[0]:
        ax.plot(epochs, [row["val_acc"] for row in history], marker="s", label="val acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(
    kmeans_labels: np.ndarray,
    dispatcher_labels: np.ndarray,
    n_clusters: int,
    path: Path,
    *,
    title: str = "Dispatcher vs k-means",
) -> Path:
    plt = _require_mpl()
    km = np.asarray(kmeans_labels, dtype=np.int64)
    ds = np.asarray(dispatcher_labels, dtype=np.int64)
    mat = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    for a, b in zip(km, ds):
        if 0 <= a < n_clusters and 0 <= b < n_clusters:
            mat[a, b] += 1.0
    row = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, np.maximum(row, EPS))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xlabel("dispatcher")
    ax.set_ylabel("k-means")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="row fraction")
    fig.tight_layout()
    return _save(fig, path)


def plot_expert_ce_by_bucket(
    base_ce: Sequence[float],
    expert_ce: Sequence[float],
    path: Path,
    *,
    title: str = "Per-bucket CE (holdout)",
) -> Path:
    plt = _require_mpl()
    b = len(base_ce)
    x = np.arange(b)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - 0.2, list(base_ce), 0.4, label="base")
    ax.bar(x + 0.2, list(expert_ce), 0.4, label="expert")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(b)])
    ax.set_xlabel("bucket")
    ax.set_ylabel("cross-entropy")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_moe_vs_base(
    metrics: dict[str, float],
    path: Path,
    *,
    title: str = "Test CE / MAE",
) -> Path:
    plt = _require_mpl()
    names = [k for k in ("base_ce", "moe_ce", "oracle_ce") if k in metrics]
    maes = [k for k in ("base_mae", "moe_mae", "oracle_mae") if k in metrics]
    labels = [k.replace("_ce", "").replace("_", " ") for k in names]
    fig, axes = plt.subplots(1, 2 if maes else 1, figsize=(8.8 if maes else 5.2, 4.4))
    if not maes:
        axes = [axes]
    axes[0].bar(labels, [metrics[k] for k in names], color=["#4c72b0", "#dd8452", "#55a868"][: len(names)])
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("CE")
    axes[0].grid(True, axis="y", alpha=0.3)
    if maes:
        mae_labels = [k.replace("_mae", "").replace("_", " ") for k in maes]
        axes[1].bar(
            mae_labels,
            [metrics[k] for k in maes],
            color=["#4c72b0", "#dd8452", "#55a868"][: len(maes)],
        )
        axes[1].set_ylabel("MAE (EV)")
        axes[1].set_title("MAE")
        axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def _subsample_xy(
    gradients: np.ndarray,
    labels: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(gradients, dtype=np.float32)
    y = np.asarray(labels)
    n = int(x.shape[0])
    take = n if int(max_points) <= 0 or n <= int(max_points) else int(max_points)
    if take < n:
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(n, size=take, replace=False)
        x = x[idx]
        y = y[idx]
    return x, y
