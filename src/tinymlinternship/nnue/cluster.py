"""Mini-batch k-means on projected sample gradients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

EPS = 1e-8


def _as_float32_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"gradients must be 2-d, got {arr.shape}")
    if arr.dtype == np.float32 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float32)


def fit_minibatch_kmeans(
    gradients: np.ndarray,
    n_clusters: int,
    *,
    batch_size: int = 10_000,
    seed: int = 0,
    n_init: int = 3,
    max_iter: int = 100,
    reassignment_ratio: float = 0.01,
) -> MiniBatchKMeans:
    n_clusters = int(n_clusters)
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
    x = _as_float32_matrix(gradients)
    if x.shape[0] < n_clusters:
        raise ValueError(f"need at least {n_clusters} rows, got {x.shape[0]}")
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(int(batch_size), int(x.shape[0])),
        random_state=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
        reassignment_ratio=float(reassignment_ratio),
        init="k-means++",
    )
    model.fit(x)
    return model


def predict_labels(model: MiniBatchKMeans, gradients: np.ndarray) -> np.ndarray:
    x = _as_float32_matrix(gradients)
    return model.predict(x).astype(np.int16, copy=False)


def assign_to_centroids(gradients: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Euclidean nearest-centroid (same objective as k-means predict)."""
    x = _as_float32_matrix(gradients)
    c = np.ascontiguousarray(centroids, dtype=np.float32)
    x2 = np.einsum("nd,nd->n", x, x)[:, None]
    c2 = np.einsum("kd,kd->k", c, c)[None, :]
    d2 = x2 + c2 - 2.0 * (x @ c.T)
    return np.argmin(d2, axis=1).astype(np.int16, copy=False)


def cluster_size_histogram(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    labels = np.asarray(labels)
    return np.bincount(labels.astype(np.int64), minlength=int(n_clusters)).astype(np.int64)


def pairwise_centroid_cosine(centroids: np.ndarray) -> np.ndarray:
    c = np.ascontiguousarray(centroids, dtype=np.float64)
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    unit = c / norms
    return unit @ unit.T


def cluster_diagnostics(
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    inertia: float | None = None,
) -> dict[str, Any]:
    n_clusters = int(centroids.shape[0])
    sizes = cluster_size_histogram(labels, n_clusters)
    cosine = pairwise_centroid_cosine(centroids)
    off = cosine.copy()
    np.fill_diagonal(off, np.nan)
    return {
        "n_clusters": n_clusters,
        "n_rows": int(np.asarray(labels).shape[0]),
        "sizes": sizes.tolist(),
        "empty": int(np.sum(sizes == 0)),
        "min_size": int(sizes.min()) if sizes.size else 0,
        "max_size": int(sizes.max()) if sizes.size else 0,
        "inertia": None if inertia is None else float(inertia),
        "centroid_cosine_mean_offdiag": None
        if n_clusters < 2
        else float(np.nanmean(off)),
        "centroid_cosine_min_offdiag": None
        if n_clusters < 2
        else float(np.nanmin(off)),
        "centroid_cosine": cosine.tolist(),
    }


def save_cluster_run(
    out_dir: Path,
    *,
    labels: np.ndarray,
    centroids: np.ndarray,
    diagnostics: dict[str, Any],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy", np.asarray(labels, dtype=np.int16))
    np.save(out_dir / "centroids.npy", np.ascontiguousarray(centroids, dtype=np.float32))
    (out_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
