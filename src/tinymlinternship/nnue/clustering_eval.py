"""Summary metrics and projection plots for the GOAL.md clustering battery.

Full per-row assignments stay in memory only long enough to score overlap.
What gets written is the metric table, the cosine-distance matrices, and the
low-dimensional coordinates of a plot subsample.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tinymlinternship.features.bucket import (
    PIECE_COUNT_BUCKETS,
    piece_count_bucket_name,
)
from tinymlinternship.nnue.cluster import pairwise_centroid_cosine

KMEANS_KS: tuple[int, ...] = (2, 4, 8, 16)
DBSCAN_TARGET_K = 8
DBSCAN_K_RANGE = (2, 16)
# Density estimation above this width is reduced to this many PCs first.
# The ε grid is a quantile of 8-NN distances, so it stays meaningful after PCA.
DBSCAN_NATIVE_DIM_MAX = 64
DBSCAN_PCA_DIM = 48
SILHOUETTE_SAMPLES = 10_000
VIZ_SAMPLES = 10_000
EMBED_DIM = 50


def piece_counts_from_indices(
    indices: np.ndarray,
    n_active: np.ndarray,
    *,
    limit: int,
) -> np.ndarray:
    """Count piece-square features in each row.

    ``indices`` is ``(N, K)`` and ``n_active`` is ``(N,)``. Features at or
    above ``limit`` are castling, en passant, or tactical bits, not pieces.
    One perspective encodes every piece once, kings included.
    """
    idx = np.asarray(indices)
    active = np.asarray(n_active).reshape(-1)
    if idx.ndim != 2 or active.shape[0] != idx.shape[0]:
        raise ValueError(f"indices {idx.shape} and n_active {active.shape} do not match")
    cols = np.arange(idx.shape[1], dtype=np.int64)
    used = cols[None, :] < active.astype(np.int64, copy=False)[:, None]
    piece = (idx >= 0) & (idx < int(limit))
    return (used & piece).sum(axis=1).astype(np.int16, copy=False)


def fill_stm_board(
    dest: np.ndarray,
    white_idx: np.ndarray,
    white_mask: np.ndarray,
    black_idx: np.ndarray,
    black_mask: np.ndarray,
    stm_white: np.ndarray,
) -> None:
    """Write STM-ordered binary features ``[x_stm ‖ x_opp]`` into ``dest`` (N, 2D)."""
    if dest.ndim != 2 or int(dest.shape[1]) % 2 != 0:
        raise ValueError(f"dest must be (N, 2D), got {dest.shape}")
    width = int(dest.shape[1] // 2)
    stm = np.asarray(stm_white, dtype=bool).reshape(-1)
    if stm.shape[0] != dest.shape[0]:
        raise ValueError("stm_white length does not match dest")
    _scatter_offset(dest, 0, white_idx, white_mask, stm, width)
    _scatter_offset(dest, width, black_idx, black_mask, stm, width)
    opp = ~stm
    _scatter_offset(dest, 0, black_idx, black_mask, opp, width)
    _scatter_offset(dest, width, white_idx, white_mask, opp, width)


def _scatter_offset(
    dest: np.ndarray,
    col_offset: int,
    indices: np.ndarray,
    mask: np.ndarray,
    row_sel: np.ndarray,
    width: int,
) -> None:
    if not np.any(row_sel):
        return
    sub_idx = np.asarray(indices)[row_sel]
    sub_mask = np.asarray(mask, dtype=bool)[row_sel]
    local_rows, slot = np.nonzero(sub_mask)
    if local_rows.size == 0:
        return
    global_rows = np.flatnonzero(row_sel)[local_rows]
    feat = sub_idx[local_rows, slot].astype(np.int64, copy=False)
    valid = (feat >= 0) & (feat < int(width))
    dest[global_rows[valid], int(col_offset) + feat[valid]] = 1.0


def piece_count_labels(counts: np.ndarray) -> np.ndarray:
    """Map piece counts onto ``PIECE_COUNT_BUCKETS`` (index 0 is 32 pieces)."""
    values = np.asarray(counts, dtype=np.int16).reshape(-1)
    labels = np.full(values.shape, -1, dtype=np.int16)
    for index, (lo, hi, _name) in enumerate(PIECE_COUNT_BUCKETS):
        labels[(values >= lo) & (values <= hi)] = index
    bad = values[labels < 0]
    if bad.size:
        sample = np.unique(bad)[:8].tolist()
        raise ValueError(f"piece counts outside 2..32, for example {sample}")
    return labels


def cosine_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance ``1 - cosine``. Empty or non-finite rows stay NaN."""
    centers = np.asarray(centroids, dtype=np.float64)
    if centers.ndim != 2:
        raise ValueError(f"centroids must be 2-d, got {centers.shape}")
    n_clusters = int(centers.shape[0])
    dist = np.full((n_clusters, n_clusters), np.nan, dtype=np.float64)
    finite = np.isfinite(centers).all(axis=1)
    finite &= np.linalg.norm(centers, axis=1) > 0.0
    if int(finite.sum()) == 0:
        return dist
    similarity = pairwise_centroid_cosine(centers[finite])
    block = 1.0 - similarity
    np.fill_diagonal(block, 0.0)
    keep = np.flatnonzero(finite)
    dist[np.ix_(keep, keep)] = block
    return dist


def _offdiag_stats(dist: np.ndarray) -> tuple[float | None, float | None]:
    if dist.size == 0 or dist.shape[0] < 2:
        return None, None
    off = dist.copy()
    np.fill_diagonal(off, np.nan)
    if not np.isfinite(off).any():
        return None, None
    return float(np.nanmean(off)), float(np.nanmin(off))


def centroids_from_labels(x: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    data = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels)
    centers = np.full((int(n_clusters), data.shape[1]), np.nan, dtype=np.float64)
    for index in range(int(n_clusters)):
        member = lab == index
        if np.any(member):
            centers[index] = data[member].mean(axis=0)
    return centers


def inertia_from_labels(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    data = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels)
    total = 0.0
    for index in range(int(centroids.shape[0])):
        center = centroids[index]
        if not np.isfinite(center).all():
            continue
        member = lab == index
        if not np.any(member):
            continue
        diff = data[member] - center
        total += float(np.einsum("nd,nd->", diff, diff))
    return total


def subsample_silhouette(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    n_samples: int,
    seed: int,
) -> float | None:
    """Euclidean silhouette on a subsample. Clusters with one draw are dropped."""
    from sklearn.metrics import silhouette_score

    lab_all = np.asarray(labels)
    n = int(lab_all.shape[0])
    take = min(int(n_samples), n)
    if take < 3:
        return None
    rng = np.random.RandomState(int(seed))
    chosen = rng.choice(n, size=take, replace=False)
    lab = lab_all[chosen]
    values, counts = np.unique(lab, return_counts=True)
    keep = values[counts >= 2]
    if keep.size < 2:
        return None
    mask = np.isin(lab, keep)
    sample = np.ascontiguousarray(np.asarray(x)[chosen][mask], dtype=np.float32)
    try:
        return float(silhouette_score(sample, lab[mask], metric="euclidean"))
    except ValueError:
        return None


def summarize_partition(
    x: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    inertia: float | None = None,
    silhouette_samples: int = SILHOUETTE_SAMPLES,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Sizes, proportions, WCSS, silhouette, and centroid cosine distance."""
    lab = np.asarray(labels).reshape(-1)
    centers = np.asarray(centroids)
    n_clusters = int(centers.shape[0])
    if lab.shape[0] != np.asarray(x).shape[0]:
        raise ValueError("labels and rows disagree")
    sizes = np.bincount(lab.astype(np.int64), minlength=n_clusters).astype(np.int64)
    n_rows = int(lab.shape[0])
    proportions = (sizes / max(n_rows, 1)).astype(np.float64)
    if inertia is None:
        inertia = inertia_from_labels(x, lab, centers)
    dist = cosine_distance_matrix(centers)
    mean_d, min_d = _offdiag_stats(dist)
    present = sizes > 0
    return {
        "n_clusters": n_clusters,
        "n_rows": n_rows,
        "sizes": sizes.tolist(),
        "proportions": [float(v) for v in proportions],
        "empty_clusters": int(np.sum(~present)),
        "inertia": float(inertia),
        "silhouette": subsample_silhouette(
            x, lab, n_samples=silhouette_samples, seed=seed
        ),
        "silhouette_samples": int(min(silhouette_samples, n_rows)),
        "cosine_distance_mean": mean_d,
        "cosine_distance_min": min_d,
        "cosine_distance": dist.tolist(),
        "names": list(names) if names is not None else [str(i) for i in range(n_clusters)],
    }


def overlap_scores(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    a = np.asarray(labels_a).reshape(-1)
    b = np.asarray(labels_b).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"overlap length {a.shape[0]} != {b.shape[0]}")
    return {
        "ari": float(adjusted_rand_score(a, b)),
        "nmi": float(normalized_mutual_info_score(a, b)),
    }


def reduce_for_dbscan(
    x: np.ndarray,
    *,
    seed: int,
    fit_cap: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the matrix DBSCAN should see, plus a note of any PCA reduction.

    Wide inputs (raw 1688-d boards, 256-d L1) make brute-force 8-NN on the
    100k fit cap impractical, and Euclidean neighborhoods there are not
    meaningful. A 48-d PCA fit on that same cap is the density space.
    """
    data = np.asarray(x)
    if int(data.shape[1]) <= DBSCAN_NATIVE_DIM_MAX:
        return np.ascontiguousarray(data, dtype=np.float32), {
            "dbscan_reduced": False,
            "cluster_feature_dim": int(data.shape[1]),
        }
    from sklearn.decomposition import PCA

    from tinymlinternship.nnue.cluster import _fit_index

    n_components = min(DBSCAN_PCA_DIM, int(data.shape[0]) - 1, int(data.shape[1]))
    idx = _fit_index(int(data.shape[0]), seed, fit_cap)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    pca.fit(np.ascontiguousarray(data[idx], dtype=np.float32))
    reduced = np.empty((data.shape[0], n_components), dtype=np.float32)
    batch = 20_000
    for start in range(0, int(data.shape[0]), batch):
        end = min(start + batch, int(data.shape[0]))
        reduced[start:end] = pca.transform(np.ascontiguousarray(data[start:end], dtype=np.float32))
    return reduced, {
        "dbscan_reduced": True,
        "cluster_feature_dim": int(n_components),
        "dbscan_pca_var": float(np.sum(pca.explained_variance_ratio_)),
        "dbscan_pca_fit_rows": int(idx.shape[0]),
    }


def embed_for_plots(
    x: np.ndarray,
    viz_idx: np.ndarray,
    *,
    seed: int,
    n_components: int = EMBED_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Leading PCs of ``x``, transformed on ``viz_idx``. Returns ``(z, variance)``."""
    data = np.asarray(x)
    k = min(int(n_components), int(data.shape[1]), int(data.shape[0]) - 1)
    if k < 2:
        raise ValueError(f"need at least 2 components, got dim {data.shape[1]}")
    if int(data.shape[1]) > DBSCAN_NATIVE_DIM_MAX:
        from sklearn.decomposition import IncrementalPCA

        model = IncrementalPCA(n_components=k, batch_size=max(10_000, k))
        batch = int(model.batch_size)
        for start in range(0, int(data.shape[0]), batch):
            block = np.ascontiguousarray(data[start : start + batch], dtype=np.float32)
            if int(block.shape[0]) < k:
                break
            model.partial_fit(block)
        z = model.transform(np.ascontiguousarray(data[viz_idx], dtype=np.float32))
        variance = np.asarray(model.explained_variance_ratio_, dtype=np.float64)
    else:
        from sklearn.decomposition import PCA

        model = PCA(n_components=k, random_state=int(seed))
        model.fit(np.ascontiguousarray(data, dtype=np.float32))
        z = model.transform(np.ascontiguousarray(data[viz_idx], dtype=np.float32))
        variance = np.asarray(model.explained_variance_ratio_, dtype=np.float64)
    return np.ascontiguousarray(z, dtype=np.float32), variance


def manifold_embed(
    z: np.ndarray,
    *,
    n_components: int,
    seed: int,
    method: str,
) -> np.ndarray:
    """t-SNE or UMAP of an already-reduced plot sample."""
    data = np.ascontiguousarray(z, dtype=np.float32)
    if method == "tsne":
        from sklearn.manifold import TSNE

        model = TSNE(
            n_components=int(n_components),
            perplexity=30.0,
            init="pca",
            learning_rate="auto",
            max_iter=750,
            random_state=int(seed),
            method="barnes_hut",
        )
        return np.ascontiguousarray(model.fit_transform(data), dtype=np.float32)
    if method == "umap":
        try:
            from umap import UMAP
        except ImportError:
            from umap.umap_ import UMAP

        model = UMAP(
            n_components=int(n_components),
            n_neighbors=30,
            min_dist=0.1,
            random_state=int(seed),
            verbose=False,
        )
        return np.ascontiguousarray(model.fit_transform(data), dtype=np.float32)
    raise ValueError(f"unknown manifold {method!r}")


def _cluster_cmap(n_clusters: int):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    import matplotlib.pyplot as plt

    n = max(int(n_clusters), 1)
    if n == 4:
        colors = ["#d62728", "#FFBF00", "#2ca02c", "#1f77b4"]
    else:
        tab = plt.get_cmap("tab20" if n > 10 else "tab10")
        colors = [tab(i % tab.N) for i in range(n)]
    cmap = ListedColormap(colors[:n], name="clusters")
    norm = BoundaryNorm(np.arange(n + 1, dtype=np.float64) - 0.5, cmap.N)
    return cmap, norm


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_embedding(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    axis_names: Sequence[str],
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    pts = np.asarray(coords)
    lab = np.asarray(labels).astype(np.int64, copy=False)
    n_clusters = int(max(int(lab.max()) + 1, 1)) if lab.size else 1
    cmap, norm = _cluster_cmap(n_clusters)
    if pts.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        ax.scatter(pts[:, 0], pts[:, 1], c=lab, s=7, cmap=cmap, norm=norm, alpha=0.45, linewidths=0)
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
    elif pts.shape[1] == 3:
        fig = plt.figure(figsize=(7.0, 5.6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2], c=lab, s=4, cmap=cmap, norm=norm, alpha=0.4, linewidths=0
        )
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        ax.set_zlabel(axis_names[2])
    else:
        raise ValueError(f"embedding must be 2-d or 3-d, got {pts.shape}")
    ax.set_title(title)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(lab)
    fig.colorbar(mappable, ax=ax, ticks=np.arange(n_clusters), fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, path)


def plot_size_panels(
    path: Path,
    panels: Sequence[tuple[str, Sequence[int], Sequence[str] | None]],
) -> Path:
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.8, 3.8), squeeze=False)
    for ax, (title, sizes, names) in zip(axes[0], panels):
        vals = np.asarray(sizes, dtype=np.int64)
        xpos = np.arange(vals.size)
        ax.bar(xpos, vals, color="#4c72b0")
        ax.set_title(title)
        ax.set_ylabel("positions")
        ax.set_xticks(xpos)
        if names is None:
            ax.set_xlabel("cluster")
            if vals.size > 8:
                ax.tick_params(axis="x", labelsize=7)
        else:
            ax.set_xticklabels(list(names), rotation=40, ha="right")
    fig.tight_layout()
    return _save(fig, path)


def plot_cosine_panels(
    path: Path,
    panels: Sequence[tuple[str, np.ndarray]],
) -> Path:
    import matplotlib.pyplot as plt

    n = len(panels)
    fig = plt.figure(figsize=(3.5 * n + 1.2, 3.9))
    grid = fig.add_gridspec(1, n + 1, width_ratios=[1.0] * n + [0.06], wspace=0.45)
    image = None
    for index, (title, matrix) in enumerate(panels):
        ax = fig.add_subplot(grid[0, index])
        data = np.asarray(matrix, dtype=np.float64)
        image = ax.imshow(data, vmin=0.0, vmax=2.0, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("cluster")
        ax.set_ylabel("cluster")
    if image is not None:
        fig.colorbar(image, cax=fig.add_subplot(grid[0, n]), label="cosine distance")
    fig.suptitle("Centroid cosine distance")
    fig.subplots_adjust(top=0.78, left=0.06, right=0.96)
    return _save(fig, path)


def write_stats_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "representation",
        "algorithm",
        "requested_k",
        "n_clusters",
        "feature_dim",
        "cluster_feature_dim",
        "n_rows",
        "sizes",
        "proportions",
        "inertia",
        "silhouette",
        "silhouette_samples",
        "cosine_distance_mean",
        "cosine_distance_min",
        "empty_clusters",
        "n_noise",
        "dbscan_epsilon",
        "dbscan_eps",
        "dbscan_min_samples",
        "dbscan_fit_rows",
        "dbscan_pca_var",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["sizes"] = _join(row.get("sizes"))
            out["proportions"] = _join(row.get("proportions"), fmt="{:.6f}")
            writer.writerow(out)


def write_overlap_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "representation_a",
        "algorithm_a",
        "k_a",
        "representation_b",
        "algorithm_b",
        "k_b",
        "ari",
        "nmi",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _join(values: Any, fmt: str = "{}") -> str:
    if values is None:
        return ""
    return "|".join(fmt.format(v) for v in values)


def piece_bucket_names() -> list[str]:
    return [piece_count_bucket_name(i) for i in range(len(PIECE_COUNT_BUCKETS))]
