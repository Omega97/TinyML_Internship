"""Cluster projected sample gradients for the MoE pipeline.

Mini-batch k-means takes a fixed bucket count B. DBSCAN takes an ε
quantile of 8-NN distances (the same 0.1 grid as the explorer) and the
pipeline then asks for a target B by searching that grid. Density is fit
on at most ``DBSCAN_FIT_CAP`` rows; every row is assigned to the nearest
core centroid so the expert heads still see the full training pack.
"""

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


def assign_to_centroids_cosine(gradients: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest centroid by cosine similarity (argmin of cosine distance).

    Assigns each row to the centroid with the smallest angle, regardless of
    magnitude: ``argmax_k (x_i · c_k) / (‖x_i‖ ‖c_k‖)``.
    """
    x = _as_float32_matrix(gradients)
    c = np.ascontiguousarray(centroids, dtype=np.float32)
    num = x @ c.T
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    cn = np.linalg.norm(c, axis=1, keepdims=True)
    denom = np.maximum(xn * cn.T, EPS)
    sim = num / denom
    return np.argmax(sim, axis=1).astype(np.int16, copy=False)


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


DBSCAN_EPSILONS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DBSCAN_FIT_CAP = 100_000
DEFAULT_DBSCAN_EPSILON = 0.3


def quantize_dbscan_epsilon(value: float | None = None) -> float:
    """Snap ε to ``{0.1, 0.2, …, 0.9}``, matching the explorer control."""
    if value is None:
        return DEFAULT_DBSCAN_EPSILON
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return DEFAULT_DBSCAN_EPSILON
    if not np.isfinite(raw):
        return DEFAULT_DBSCAN_EPSILON
    stepped = round(raw * 10.0) / 10.0
    if stepped < 0.1:
        return 0.1
    if stepped > 0.9:
        return 0.9
    return float(stepped)


def normalize_train_algorithm(name: str) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key in {"kmeans", "minibatchkmeans", "minibatch"}:
        return "minibatch_kmeans"
    if key == "dbscan":
        return "dbscan"
    raise ValueError(
        f"unknown training clustering algorithm {name!r} "
        "(expected minibatch_kmeans or dbscan)"
    )


def dbscan_min_samples(n: int) -> int:
    """Same min_samples schedule as the explorer density fit."""
    n = int(n)
    min_samples = max(25, min(80, n // 120))
    return int(min(min_samples, max(5, n // 8)))


def _core_cluster_count(labels: np.ndarray) -> int:
    vals = np.asarray(labels)
    vals = vals[vals >= 0]
    return int(vals.max()) + 1 if vals.size else 0


def _knn_kth(x: np.ndarray, n_neighbors: int = 8) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    n = int(x.shape[0])
    k = min(int(n_neighbors), max(2, n - 1))
    dists, _ = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)
    return np.asarray(dists[:, -1], dtype=np.float64)


def _dbscan_raw(
    x: np.ndarray,
    kth: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.cluster import DBSCAN

    epsilon = quantize_dbscan_epsilon(epsilon)
    percentile = 100.0 * float(epsilon)
    eps = max(float(np.percentile(kth, percentile)), 1e-5)
    min_samples = dbscan_min_samples(int(x.shape[0]))
    raw = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(x)
    meta = {
        "algorithm": "dbscan",
        "dbscan_eps": float(eps),
        "dbscan_epsilon": float(epsilon),
        "dbscan_min_samples": int(min_samples),
        "dbscan_percentile": float(percentile),
    }
    return np.asarray(raw, dtype=np.int64), meta


def _centroids_from_core(
    x: np.ndarray,
    raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mean of each dense DBSCAN component. Noise stays out of the means."""
    keep = [int(v) for v in np.unique(raw) if int(v) >= 0]
    if len(keep) < 2:
        raise ValueError(
            f"DBSCAN found {len(keep)} dense cluster(s); need at least 2"
        )
    centroids = np.zeros((len(keep), x.shape[1]), dtype=np.float32)
    core_sizes: list[int] = []
    n_noise = int(np.sum(np.asarray(raw) < 0))
    for new, old in enumerate(keep):
        member = np.asarray(raw) == old
        core_sizes.append(int(member.sum()))
        centroids[new] = x[member].mean(axis=0)
    return centroids, {"n_noise": n_noise, "core_sizes": core_sizes}


def _fit_index(n: int, seed: int, fit_cap: int) -> np.ndarray:
    cap = int(fit_cap)
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    return np.sort(rng.choice(n, size=cap, replace=False)).astype(np.int64)


def _finish_dbscan(
    x: np.ndarray,
    subset: np.ndarray,
    raw: np.ndarray,
    meta: dict[str, Any],
    *,
    requested_clusters: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    centroids, extra = _centroids_from_core(subset, raw)
    labels = assign_to_centroids(x, centroids)
    diff = x - centroids[labels.astype(np.int64)]
    inertia = float(np.einsum("nd,nd->", diff, diff))
    diag = cluster_diagnostics(labels, centroids, inertia=inertia)
    diag.update(meta)
    diag.update(extra)
    diag["algorithm"] = "dbscan"
    diag["dbscan_fit_rows"] = int(subset.shape[0])
    diag["n_rows_full"] = int(x.shape[0])
    if requested_clusters is not None:
        diag["requested_clusters"] = int(requested_clusters)
        diag["dbscan_b_match"] = bool(int(diag["n_clusters"]) == int(requested_clusters))
    return labels, centroids, diag


def fit_dbscan(
    gradients: np.ndarray,
    epsilon: float = DEFAULT_DBSCAN_EPSILON,
    *,
    seed: int = 0,
    fit_cap: int = DBSCAN_FIT_CAP,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """DBSCAN at one ε quantile, then nearest-centroid labels for every row."""
    x = _as_float32_matrix(gradients)
    idx = _fit_index(int(x.shape[0]), seed, fit_cap)
    subset = x[idx]
    raw, meta = _dbscan_raw(subset, _knn_kth(subset), epsilon)
    return _finish_dbscan(x, subset, raw, meta, requested_clusters=None)


def fit_dbscan_for_b(
    gradients: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 0,
    fit_cap: int = DBSCAN_FIT_CAP,
    epsilons: tuple[float, ...] = DBSCAN_EPSILONS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pick the ε quantile whose dense-cluster count is closest to ``n_clusters``."""
    target = int(n_clusters)
    if target < 2:
        raise ValueError(f"n_clusters must be >= 2, got {target}")
    x = _as_float32_matrix(gradients)
    idx = _fit_index(int(x.shape[0]), seed, fit_cap)
    subset = x[idx]
    kth = _knn_kth(subset)
    trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float], np.ndarray, dict[str, Any]] | None = None
    for epsilon in epsilons:
        raw, meta = _dbscan_raw(subset, kth, epsilon)
        n_core = _core_cluster_count(raw)
        n_noise = int(np.sum(raw < 0))
        snapped = float(meta["dbscan_epsilon"])
        trials.append(
            {
                "dbscan_epsilon": snapped,
                "n_clusters": n_core,
                "n_noise": n_noise,
                "dbscan_eps": meta["dbscan_eps"],
            }
        )
        if n_core < 2:
            continue
        rank = (abs(n_core - target), float(n_noise), abs(snapped - DEFAULT_DBSCAN_EPSILON))
        if best is None or rank < best[0]:
            best = (rank, raw, meta)
    if best is None:
        raise ValueError(
            f"DBSCAN never found 2 or more clusters while targeting B={target}. "
            f"Trials: {trials}"
        )
    _rank, raw, meta = best
    labels, centroids, diag = _finish_dbscan(
        x, subset, raw, meta, requested_clusters=target
    )
    diag["dbscan_trials"] = trials
    return labels, centroids, diag


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
