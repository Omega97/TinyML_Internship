"""Load per-sample gradient clusters for the interactive explorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tinymlinternship.nnue.dataset import (
    epd_key,
    load_fen_value_visits,
    slice_source_json,
)

CLUSTER_COLORS: tuple[str, ...] = (
    "#e74c3c",  # red
    "#daa520",  # goldenrod
    "#2ecc71",  # green
    "#3498db",  # blue
    "#9b59b6",
    "#e67e22",
    "#1abc9c",
    "#f1c40f",
    "#e84393",
    "#00cec9",
    "#6c5ce7",
    "#fd79a8",
)

REQUIRED_TABLE_COLUMNS = ("coord_x", "coord_y", "cluster_id")
OPTIONAL_TABLE_COLUMNS = (
    "sample_id",
    "coord_z",
    "fen",
    "eval_target",
    "eval_pred",
    "grad_norm",
    "slice_id",
    "local_row",
)


NOISE_COLOR = "#9aa6b8"
POOL_SIZES: tuple[int, ...] = (300, 1_000, 3_000, 10_000, 30_000)
DEFAULT_POOL_SIZE = 10_000
CLUSTER_ALGORITHMS: tuple[str, ...] = ("kmeans", "kmedoids", "dbscan")
CLUSTER_ALGO_LABELS: dict[str, str] = {
    "kmeans": "Mini-batch k-Means",
    "kmedoids": "k-Medoids",
    "dbscan": "DBSCAN",
}


def cluster_color(cluster_id: int, n_clusters: int | None = None) -> str:
    if int(cluster_id) < 0:
        return NOISE_COLOR
    n = max(int(n_clusters or 0), int(cluster_id) + 1, 1)
    if 0 <= int(cluster_id) < len(CLUSTER_COLORS) and n <= len(CLUSTER_COLORS):
        return CLUSTER_COLORS[int(cluster_id)]
    import matplotlib.pyplot as plt

    tab = plt.get_cmap("tab10")
    rgba = tab(int(cluster_id) % 10)
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


def cluster_palette(n_clusters: int) -> list[str]:
    n = max(int(n_clusters), 1)
    return [cluster_color(i, n) for i in range(n)]


def stm_value_from_wdl(wdl: Any) -> float | None:
    if wdl is None:
        return None
    arr = np.asarray(wdl, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return float(arr[0] - arr[2])


@dataclass
class PositionInfo:
    index: int
    sample_id: int
    cluster_id: int
    coord_x: float
    coord_y: float
    coord_z: float | None = None
    fen: str | None = None
    eval_target: float | None = None
    eval_pred: float | None = None
    grad_norm: float | None = None
    slice_id: int | None = None
    local_row: int | None = None
    slice_name: str | None = None


@dataclass
class ExplorerData:
    sample_id: np.ndarray
    coord_x: np.ndarray
    coord_y: np.ndarray
    cluster_id: np.ndarray
    coord_z: np.ndarray | None = None
    fen: np.ndarray | None = None
    eval_target: np.ndarray | None = None
    eval_pred: np.ndarray | None = None
    grad_norm: np.ndarray | None = None
    slice_id: np.ndarray | None = None
    local_row: np.ndarray | None = None
    folders: list[Path] = field(default_factory=list)
    features: np.ndarray | None = None
    cluster_seed: int = 0
    requested_clusters: int = 0
    method: str = "pca"
    algorithm: str = "kmeans"
    source: str = ""
    n_source_rows: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    projection_cache: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = int(self.sample_id.shape[0])
        self.sample_id = np.asarray(self.sample_id, dtype=np.int64)
        self.coord_x = np.asarray(self.coord_x, dtype=np.float32)
        self.coord_y = np.asarray(self.coord_y, dtype=np.float32)
        self.cluster_id = np.asarray(self.cluster_id, dtype=np.int16)
        if self.coord_x.shape[0] != n or self.coord_y.shape[0] != n or self.cluster_id.shape[0] != n:
            raise ValueError("explorer arrays must share the same length")
        if self.features is not None:
            feat = np.ascontiguousarray(self.features, dtype=np.float32)
            if feat.ndim != 2 or int(feat.shape[0]) != n:
                raise ValueError(f"features must be (n, d), got {getattr(feat, 'shape', None)}")
            self.features = feat
        if self.n_source_rows <= 0:
            self.n_source_rows = n
        if self.requested_clusters <= 0 and self.cluster_id.size:
            self.requested_clusters = int(self.cluster_id.max()) + 1

    def __len__(self) -> int:
        return int(self.sample_id.shape[0])

    @property
    def n_clusters(self) -> int:
        if self.cluster_id.size == 0:
            return 0
        pos = self.cluster_id[self.cluster_id >= 0]
        inferred = int(pos.max()) + 1 if pos.size else 0
        if str(self.algorithm) == "dbscan":
            return inferred
        if self.requested_clusters > 0:
            return int(self.requested_clusters)
        return inferred

    @property
    def has_noise(self) -> bool:
        return bool(self.cluster_id.size and int(self.cluster_id.min()) < 0)

    def visible_mask(self, *, clusters: Iterable[int] | None = None) -> np.ndarray:
        mask = np.ones(len(self), dtype=np.bool_)
        if clusters is not None:
            allowed = np.fromiter((int(c) for c in clusters), dtype=np.int16)
            mask &= np.isin(self.cluster_id, allowed)
        return mask

    def row(self, index: int) -> PositionInfo:
        i = int(index)
        z = None if self.coord_z is None else float(self.coord_z[i])
        fen = None
        if self.fen is not None:
            raw = self.fen[i]
            if raw is not None and str(raw) != "" and str(raw) != "nan":
                fen = str(raw)
        et = None if self.eval_target is None or not np.isfinite(self.eval_target[i]) else float(
            self.eval_target[i]
        )
        ep = None if self.eval_pred is None or not np.isfinite(self.eval_pred[i]) else float(
            self.eval_pred[i]
        )
        gn = None if self.grad_norm is None or not np.isfinite(self.grad_norm[i]) else float(
            self.grad_norm[i]
        )
        sid = None if self.slice_id is None else int(self.slice_id[i])
        local = None if self.local_row is None else int(self.local_row[i])
        slice_name = None
        if sid is not None and 0 <= sid < len(self.folders):
            slice_name = self.folders[sid].name
        return PositionInfo(
            index=i,
            sample_id=int(self.sample_id[i]),
            cluster_id=int(self.cluster_id[i]),
            coord_x=float(self.coord_x[i]),
            coord_y=float(self.coord_y[i]),
            coord_z=z,
            fen=fen,
            eval_target=et,
            eval_pred=ep,
            grad_norm=gn,
            slice_id=sid,
            local_row=local,
            slice_name=slice_name,
        )


class SelectionModel:
    """Toggle multi-select. Pin count is unbounded."""

    def __init__(self) -> None:
        self.order: list[int] = []

    def __contains__(self, index: int) -> bool:
        return int(index) in self.order

    def __len__(self) -> int:
        return len(self.order)

    def toggle(self, index: int) -> list[int]:
        idx = int(index)
        if idx in self.order:
            self.order.remove(idx)
        else:
            self.order.append(idx)
        return list(self.order)

    def discard(self, index: int) -> bool:
        idx = int(index)
        if idx in self.order:
            self.order.remove(idx)
            return True
        return False

    def clear(self) -> None:
        self.order.clear()


class FenResolver:
    """Map ``(slice_id, local_row)`` onto the unique-EPD table used at encode time."""

    def __init__(self, folders: Sequence[Path]) -> None:
        self.folders = [Path(p) for p in folders]
        self._frames: dict[int, Any] = {}

    def _frame(self, slice_id: int):
        sid = int(slice_id)
        if sid in self._frames:
            return self._frames[sid]
        if sid < 0 or sid >= len(self.folders):
            self._frames[sid] = None
            return None
        folder = self.folders[sid]
        source = slice_source_json(folder)
        if source is None or not source.is_file():
            self._frames[sid] = None
            return None
        df = load_fen_value_visits(source)
        work = df.copy()
        work["epd"] = work["fen"].map(epd_key)
        unique = work.drop_duplicates(subset=["epd"], keep="first").reset_index(drop=True)
        # features.npz rows match unique EPDs; if no dups, this is identity.
        self._frames[sid] = unique
        return unique

    def lookup(self, slice_id: int | None, local_row: int | None) -> dict[str, Any]:
        if slice_id is None or local_row is None:
            return {}
        frame = self._frame(int(slice_id))
        if frame is None:
            return {}
        row = int(local_row)
        if row < 0 or row >= len(frame):
            return {}
        rec = frame.iloc[row]
        out: dict[str, Any] = {"fen": str(rec["fen"])}
        value = stm_value_from_wdl(rec["wdl"] if "wdl" in rec else None)
        if value is None and "value" in rec and rec["value"] == rec["value"]:
            value = float(rec["value"])
        if value is not None:
            out["eval_target"] = value
        return out


class ModelPredictor:
    """Optional ŷ from a DualHidden NNUE checkpoint + slice feature DB."""

    def __init__(self, checkpoint: Path | None, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.device = device
        self._model = None
        self._datasets: dict[int, Any] = {}
        self._failed = False

    def _load(self):
        if self._model is not None or self._failed or self.checkpoint is None:
            return self._model
        if not self.checkpoint.is_file():
            self._failed = True
            return None
        try:
            from tinymlinternship.nnue.moe import load_dual_hidden_checkpoint

            self._model = load_dual_hidden_checkpoint(self.checkpoint, device=self.device)
            self._model.eval()
        except Exception:
            self._failed = True
            self._model = None
        return self._model

    def predict(
        self,
        folders: Sequence[Path],
        slice_id: int | None,
        local_row: int | None,
    ) -> float | None:
        if slice_id is None or local_row is None:
            return None
        model = self._load()
        if model is None:
            return None
        sid = int(slice_id)
        if sid < 0 or sid >= len(folders):
            return None
        try:
            import torch

            from tinymlinternship.nnue.dataset import FenValueVisitsDataset

            if sid not in self._datasets:
                self._datasets[sid] = FenValueVisitsDataset(Path(folders[sid]), progress=False)
            ds = self._datasets[sid]
            batch = ds.gather(np.array([int(local_row)], dtype=np.int64))
            with torch.no_grad():
                logits = model(
                    batch["white_idx"],
                    batch["black_idx"],
                    batch["stm_white"],
                    batch["white_mask"],
                    batch["black_mask"],
                )
                probs = torch.softmax(logits.float(), dim=-1)[0]
            return float(probs[0] - probs[2])
        except Exception:
            return None


def resolve_position(
    data: ExplorerData,
    index: int,
    *,
    resolver: FenResolver | None = None,
    predictor: ModelPredictor | None = None,
) -> PositionInfo:
    info = data.row(index)
    if info.fen is None and resolver is not None:
        extra = resolver.lookup(info.slice_id, info.local_row)
        info.fen = extra.get("fen")
        if info.eval_target is None:
            info.eval_target = extra.get("eval_target")
    if info.eval_pred is None and predictor is not None:
        info.eval_pred = predictor.predict(data.folders, info.slice_id, info.local_row)
    return info


def _subsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    n = int(n)
    take = n if int(max_points) <= 0 or n <= int(max_points) else int(max_points)
    if take >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(n, size=take, replace=False)
    idx.sort()
    return idx.astype(np.int64, copy=False)


def clustering_features(data: ExplorerData) -> np.ndarray:
    """Matrix k-means should see: stored gradients, else the 2D/3D projection."""
    if data.features is not None:
        return np.ascontiguousarray(data.features, dtype=np.float32)
    cols = [data.coord_x, data.coord_y]
    if data.coord_z is not None:
        cols.append(np.asarray(data.coord_z, dtype=np.float32))
    return np.column_stack(cols).astype(np.float32, copy=False)


def normalize_cluster_algorithm(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "kmeans": "kmeans",
        "minibatchkmeans": "kmeans",
        "kmedoids": "kmedoids",
        "medoids": "kmedoids",
        "dbscan": "dbscan",
    }
    if key not in aliases:
        raise ValueError(f"unknown clustering algorithm {name!r}")
    return aliases[key]


def _label_diagnostics(labels: np.ndarray, n_clusters: int, *, inertia: float | None = None) -> dict[str, Any]:
    labels = np.asarray(labels)
    noise = int(np.sum(labels < 0))
    pos = labels[labels >= 0].astype(np.int64, copy=False)
    k = max(int(n_clusters), int(pos.max()) + 1 if pos.size else 0)
    sizes = np.bincount(pos, minlength=k).astype(np.int64) if k else np.zeros(0, dtype=np.int64)
    return {
        "n_clusters": k,
        "n_rows": int(labels.shape[0]),
        "sizes": sizes.tolist(),
        "noise": noise,
        "empty": int(np.sum(sizes == 0)) if sizes.size else 0,
        "min_size": int(sizes.min()) if sizes.size else 0,
        "max_size": int(sizes.max()) if sizes.size else 0,
        "inertia": None if inertia is None else float(inertia),
    }


def _fit_kmedoids(x: np.ndarray, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """FasterPAM on small n; CLARA (sampled FasterPAM) when a full distance matrix is too big."""
    from sklearn.metrics.pairwise import euclidean_distances

    import kmedoids

    x = np.ascontiguousarray(x, dtype=np.float32)
    n = int(x.shape[0])
    k = int(n_clusters)
    rng = np.random.RandomState(int(seed))

    def _pam(block: np.ndarray, local_seed: int) -> np.ndarray:
        dist = euclidean_distances(block)
        result = kmedoids.fasterpam(dist, k, random_state=int(local_seed))
        return np.asarray(result.medoids, dtype=np.int64)

    if n <= 3_500:
        medoids = _pam(x, seed)
    else:
        sample_n = min(n, 2_500)
        best_medoids = None
        best_cost = float("inf")
        for trial in range(4):
            idx = rng.choice(n, size=sample_n, replace=False)
            local = _pam(x[idx], seed + trial)
            medoids = idx[local].astype(np.int64, copy=False)
            dist = euclidean_distances(x, x[medoids])
            cost = float(dist.min(axis=1).sum())
            if cost < best_cost:
                best_cost = cost
                best_medoids = medoids
        medoids = best_medoids if best_medoids is not None else rng.choice(n, size=k, replace=False)
    dist = euclidean_distances(x, x[medoids])
    labels = dist.argmin(axis=1).astype(np.int16, copy=False)
    return labels, medoids


def _dbscan_cluster_count(labels: np.ndarray) -> int:
    pos = np.asarray(labels)
    pos = pos[pos >= 0]
    return int(pos.max()) + 1 if pos.size else 0


def _fit_dbscan(x: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Tighter ε and larger min_samples than the old 85th-percentile rule.

    Default ε is the 30th percentile of 8-NN distances (was 85th). min_samples
    scales with n and is much higher than before. If that does not yield 3–6
    clusters, ε is searched over lower/higher percentiles.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    x = np.ascontiguousarray(x, dtype=np.float32)
    n = int(x.shape[0])
    n_neighbors = min(8, max(2, n - 1))
    dists, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors(x)
    kth = dists[:, -1]
    min_samples = max(25, min(80, n // 120))
    min_samples = min(min_samples, max(5, n // 8))

    def _run(percentile: float) -> tuple[np.ndarray, int, float]:
        eps = max(float(np.percentile(kth, percentile)), 1e-5)
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(x)
        return labels.astype(np.int16, copy=False), _dbscan_cluster_count(labels), eps

    def _score(n_clusters: int) -> float:
        if 3 <= n_clusters <= 6:
            return 0.0
        if n_clusters <= 0:
            return 100.0
        return abs(n_clusters - 4.5)

    labels, n_clusters, eps = _run(30.0)
    best = (_score(n_clusters), labels, n_clusters, eps, 30.0)
    if best[0] != 0.0:
        for percentile in (12, 16, 20, 24, 28, 32, 36, 42, 48, 55, 65):
            lab, k, e = _run(float(percentile))
            scored = (_score(k), lab, k, e, float(percentile))
            if scored[0] < best[0]:
                best = scored
            if scored[0] == 0.0:
                break
    _score_v, labels, n_clusters, eps, percentile = best
    meta = {
        "dbscan_eps": float(eps),
        "dbscan_min_samples": int(min_samples),
        "dbscan_percentile": float(percentile),
    }
    return labels, meta


def recluster(
    data: ExplorerData,
    n_clusters: int,
    *,
    seed: int | None = None,
    algorithm: str | None = None,
) -> ExplorerData:
    """Fit the chosen algorithm in place. Scatter coordinates are unchanged."""
    from tinymlinternship.nnue.cluster import fit_minibatch_kmeans

    x = clustering_features(data)
    algo = normalize_cluster_algorithm(algorithm or data.algorithm or "kmeans")
    rng_seed = int(data.cluster_seed if seed is None else seed)
    k = int(n_clusters)
    if algo != "dbscan":
        if k < 2:
            raise ValueError(f"n_clusters must be >= 2, got {k}")
        if int(x.shape[0]) < k:
            raise ValueError(f"need at least {k} points to fit {k} clusters, got {x.shape[0]}")

    if algo == "kmeans":
        km = fit_minibatch_kmeans(
            x,
            k,
            batch_size=min(10_000, int(x.shape[0])),
            seed=rng_seed,
        )
        labels = km.predict(x).astype(np.int16, copy=False)
        diag = _label_diagnostics(labels, k, inertia=float(km.inertia_))
    elif algo == "kmedoids":
        labels, medoids = _fit_kmedoids(x, k, rng_seed)
        inertia = float(np.linalg.norm(x - x[medoids[labels]], axis=1).sum())
        diag = _label_diagnostics(labels, k, inertia=inertia)
    else:
        labels, db_meta = _fit_dbscan(x, rng_seed)
        pos = labels[labels >= 0]
        k_found = int(pos.max()) + 1 if pos.size else 0
        diag = _label_diagnostics(labels, k_found)
        diag.update(db_meta)
        k = k_found

    data.cluster_id = labels
    data.algorithm = algo
    data.requested_clusters = int(k)
    data.cluster_seed = rng_seed
    data.diagnostics.update(diag)
    data.diagnostics["algorithm"] = algo
    data.diagnostics["clustered_on"] = "gradients" if data.features is not None else "projection"
    return data


PROJECTION_METHODS: tuple[str, ...] = ("pca", "tsne", "umap", "isomap", "lle")
PROJECTION_LABELS: dict[str, str] = {
    "pca": "PCA",
    "tsne": "t-SNE",
    "umap": "UMAP",
    "isomap": "Isomap",
    "lle": "LLE",
}
_PROJECTION_ALIASES: dict[str, str] = {
    "pca": "pca",
    "tsne": "tsne",
    "t-sne": "tsne",
    "t_sne": "tsne",
    "umap": "umap",
    "isomap": "isomap",
    "lle": "lle",
    "locally-linear-embedding": "lle",
    "locally_linear_embedding": "lle",
}


def normalize_projection_method(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "-")
    if key not in _PROJECTION_ALIASES:
        raise ValueError(
            f"unknown projection {name!r}; expected one of {tuple(PROJECTION_LABELS.values())}"
        )
    return _PROJECTION_ALIASES[key]


def projection_label(method: str) -> str:
    return PROJECTION_LABELS[normalize_projection_method(method)]


def umap_available() -> bool:
    try:
        import umap  # noqa: F401
    except ImportError:
        return False
    return True


def _n_neighbors(n: int, default: int = 12, *, components: int = 2) -> int:
    n = int(n)
    cap = max(2, n - 1)
    floor = max(2, int(components) + 1)
    return int(min(cap, max(floor, int(default))))


def project_gradients(
    gradients: np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 2,
    seed: int = 0,
) -> np.ndarray:
    method = normalize_projection_method(method)
    x = np.ascontiguousarray(gradients, dtype=np.float32)
    n, dim = int(x.shape[0]), int(x.shape[1])
    if n < 3:
        raise ValueError(f"need at least 3 points to project, got {n}")
    k = max(2, int(n_components))
    k = min(k, n - 1)
    if method == "pca":
        from sklearn.decomposition import PCA

        k = min(k, dim)
        coords = PCA(n_components=k, random_state=int(seed)).fit_transform(x)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = float(min(30.0, max(5.0, (n - 1) / 3.0)))
        coords = TSNE(
            n_components=k,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=int(seed),
            max_iter=500 if n >= 2_000 else 750,
            n_jobs=-1,
        ).fit_transform(x)
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP is not installed; pip install umap-learn") from exc
        coords = umap.UMAP(
            n_components=k,
            n_neighbors=_n_neighbors(n, 15, components=k),
            min_dist=0.1,
            metric="euclidean",
            random_state=int(seed),
        ).fit_transform(x)
    elif method == "isomap":
        from sklearn.manifold import Isomap

        coords = Isomap(
            n_neighbors=_n_neighbors(n, 12, components=k),
            n_components=k,
            eigen_solver="auto",
            n_jobs=-1,
        ).fit_transform(x)
    elif method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding

        coords = LocallyLinearEmbedding(
            n_neighbors=_n_neighbors(n, 12, components=k),
            n_components=k,
            random_state=int(seed),
            eigen_solver="auto",
            n_jobs=-1,
        ).fit_transform(x)
    else:
        raise ValueError(f"unknown projection {method!r}")
    return np.ascontiguousarray(coords, dtype=np.float32)


def _apply_coords(data: ExplorerData, coords: np.ndarray, method: str) -> None:
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    if coords.ndim != 2 or int(coords.shape[0]) != len(data) or int(coords.shape[1]) < 2:
        raise ValueError(f"coords must be (n, >=2), got {coords.shape}")
    data.coord_x = coords[:, 0]
    data.coord_y = coords[:, 1]
    data.coord_z = coords[:, 2] if coords.shape[1] > 2 else None
    data.method = normalize_projection_method(method)
    data.projection_cache[data.method] = coords


def reproject(
    data: ExplorerData,
    method: str,
    *,
    seed: int | None = None,
    n_components: int = 2,
) -> ExplorerData:
    """Recompute the 2D layout. Cluster labels are unchanged."""
    method = normalize_projection_method(method)
    cached = data.projection_cache.get(method)
    if cached is not None and int(np.asarray(cached).shape[0]) == len(data):
        _apply_coords(data, cached, method)
        return data
    x = clustering_features(data)
    rng_seed = int(data.cluster_seed if seed is None else seed)
    coords = project_gradients(x, method=method, n_components=n_components, seed=rng_seed)
    _apply_coords(data, coords, method)
    return data


def load_table(path: Path) -> ExplorerData:
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"unsupported table {path.suffix}")
    missing = [c for c in REQUIRED_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    n = len(df)
    sample_id = (
        df["sample_id"].to_numpy(dtype=np.int64)
        if "sample_id" in df.columns
        else np.arange(n, dtype=np.int64)
    )
    fen = df["fen"].astype(str).to_numpy(dtype=object) if "fen" in df.columns else None
    def _opt_float(name: str) -> np.ndarray | None:
        if name not in df.columns:
            return None
        return df[name].to_numpy(dtype=np.float32)

    z = _opt_float("coord_z")
    coord_x = df["coord_x"].to_numpy(dtype=np.float32)
    coord_y = df["coord_y"].to_numpy(dtype=np.float32)
    feat_cols = [coord_x, coord_y]
    if z is not None:
        feat_cols.append(z)
    data = ExplorerData(
        sample_id=sample_id,
        coord_x=coord_x,
        coord_y=coord_y,
        coord_z=z,
        cluster_id=df["cluster_id"].to_numpy(dtype=np.int16),
        fen=fen,
        eval_target=_opt_float("eval_target"),
        eval_pred=_opt_float("eval_pred"),
        grad_norm=_opt_float("grad_norm"),
        slice_id=df["slice_id"].to_numpy(dtype=np.int32) if "slice_id" in df.columns else None,
        local_row=df["local_row"].to_numpy(dtype=np.int64) if "local_row" in df.columns else None,
        features=np.column_stack(feat_cols).astype(np.float32, copy=False),
        method="given",
        source=str(path),
        n_source_rows=n,
    )
    return data


def load_work_dir(
    work_dir: Path,
    *,
    max_points: int = DEFAULT_POOL_SIZE,
    method: str = "pca",
    seed: int = 0,
    n_components: int = 2,
    n_clusters: int | None = 4,
    algorithm: str = "kmeans",
) -> ExplorerData:
    import json

    work_dir = Path(work_dir)
    grad_path = work_dir / "gradients.npy"
    label_path = work_dir / "labels.npy"
    if not grad_path.is_file():
        raise FileNotFoundError(f"{work_dir} needs gradients.npy")
    grads = np.load(grad_path, mmap_mode="r")
    n = int(grads.shape[0])
    saved = None
    if label_path.is_file():
        saved = np.load(label_path)
        if int(saved.shape[0]) != n:
            raise ValueError(f"labels {saved.shape[0]} vs gradients {n}")
    idx = _subsample_indices(n, max_points, seed)
    x = np.ascontiguousarray(grads[idx], dtype=np.float32)
    if saved is not None:
        y = np.asarray(saved[idx], dtype=np.int16)
    else:
        y = np.zeros(int(idx.shape[0]), dtype=np.int16)
    method = normalize_projection_method(method)
    coords = project_gradients(x, method=method, n_components=n_components, seed=seed)
    coord_z = coords[:, 2] if coords.shape[1] > 2 else None
    grad_norm = np.linalg.norm(x, axis=1).astype(np.float32, copy=False)

    slice_id = None
    local_row = None
    sid_path = work_dir / "slice_ids.npy"
    loc_path = work_dir / "local_rows.npy"
    if sid_path.is_file() and loc_path.is_file():
        slice_id = np.asarray(np.load(sid_path)[idx], dtype=np.int32)
        local_row = np.asarray(np.load(loc_path)[idx], dtype=np.int64)

    folders: list[Path] = []
    diagnostics: dict[str, Any] = {}
    meta_path = work_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        folders = [Path(p) for p in meta.get("folders", [])]
        diagnostics["meta"] = {k: v for k, v in meta.items() if k != "folders"}
    diag_path = work_dir / "diagnostics.json"
    if diag_path.is_file():
        diagnostics["saved"] = json.loads(diag_path.read_text(encoding="utf-8"))

    data = ExplorerData(
        sample_id=idx,
        coord_x=coords[:, 0],
        coord_y=coords[:, 1],
        coord_z=coord_z,
        cluster_id=y,
        grad_norm=grad_norm,
        slice_id=slice_id,
        local_row=local_row,
        folders=folders,
        features=x,
        cluster_seed=int(seed),
        method=method,
        algorithm="kmeans",
        source=str(work_dir),
        n_source_rows=n,
        diagnostics=diagnostics,
        projection_cache={method: np.ascontiguousarray(coords, dtype=np.float32)},
    )
    if n_clusters is not None:
        recluster(
            data,
            int(n_clusters),
            seed=int(seed),
            algorithm=algorithm,
        )
    return data


def make_demo_data(
    n: int = 800,
    n_clusters: int = 4,
    seed: int = 0,
) -> ExplorerData:
    """Synthetic 2D blobs + real chess FENs so the window works without a MoE run."""
    import chess

    rng = np.random.RandomState(int(seed))
    n = max(int(n), int(n_clusters))
    k = int(n_clusters)
    means = rng.uniform(-3.0, 3.0, size=(k, 2)).astype(np.float32)
    cluster_id = rng.randint(0, k, size=n).astype(np.int16)
    coord = means[cluster_id] + rng.normal(0.0, 0.35, size=(n, 2)).astype(np.float32)
    coord_z = rng.normal(0.0, 0.4, size=n).astype(np.float32)
    grad_norm = rng.uniform(0.4, 1.6, size=n).astype(np.float32)
    eval_target = rng.uniform(-0.9, 0.9, size=n).astype(np.float32)
    eval_pred = np.clip(eval_target + rng.normal(0.0, 0.12, size=n), -1.0, 1.0).astype(
        np.float32
    )

    start = chess.Board()
    n_fen = min(n, 256)
    pool: list[str] = []
    for _ in range(n_fen):
        board = start.copy()
        ply = int(rng.randint(2, 28))
        for _ in range(ply):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.randint(0, len(moves)))])
        pool.append(board.fen())
    fens = np.empty(n, dtype=object)
    for i in range(n):
        fens[i] = pool[i % n_fen]

    return ExplorerData(
        sample_id=np.arange(n, dtype=np.int64),
        coord_x=coord[:, 0],
        coord_y=coord[:, 1],
        coord_z=coord_z,
        cluster_id=cluster_id,
        fen=fens,
        eval_target=eval_target,
        eval_pred=eval_pred,
        grad_norm=grad_norm,
        features=np.column_stack([coord[:, 0], coord[:, 1], coord_z]).astype(np.float32, copy=False),
        cluster_seed=int(seed),
        requested_clusters=k,
        method="pca",
        algorithm="kmeans",
        source="demo",
        n_source_rows=n,
        diagnostics={"n_clusters": k, "sizes": np.bincount(cluster_id, minlength=k).tolist()},
        projection_cache={
            "pca": np.column_stack([coord[:, 0], coord[:, 1]]).astype(np.float32, copy=False)
        },
    )


def reload_pool(
    data: ExplorerData,
    n_points: int,
    *,
    method: str | None = None,
    n_clusters: int | None = None,
    algorithm: str | None = None,
) -> ExplorerData:
    """Rebuild the working set to ``n_points`` (reprojects and re-clusters)."""
    method = method or data.method
    algorithm = algorithm or data.algorithm
    seed = int(data.cluster_seed)
    k = int(n_clusters if n_clusters is not None else max(int(data.requested_clusters or 2), 2))
    n_points = max(int(n_points), 2)
    if str(data.source) == "demo":
        new = make_demo_data(n=n_points, n_clusters=max(k, 2), seed=seed)
        try:
            if normalize_projection_method(method) != "pca":
                reproject(new, method, seed=seed)
        except ValueError:
            pass
        recluster(new, k, seed=seed, algorithm=algorithm)
        return new
    path = Path(data.source)
    if (path / "gradients.npy").is_file():
        return load_work_dir(
            path,
            max_points=n_points,
            method=method,
            seed=seed,
            n_clusters=k,
            algorithm=algorithm,
        )
    take = min(n_points, len(data))
    if take >= len(data) and take <= n_points:
        recluster(data, k, seed=seed, algorithm=algorithm)
        return data
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(data), size=take, replace=False))

    def _sub(arr):
        if arr is None:
            return None
        return arr[idx]

    new = ExplorerData(
        sample_id=_sub(data.sample_id),
        coord_x=_sub(data.coord_x),
        coord_y=_sub(data.coord_y),
        coord_z=_sub(data.coord_z),
        cluster_id=_sub(data.cluster_id),
        fen=_sub(data.fen),
        eval_target=_sub(data.eval_target),
        eval_pred=_sub(data.eval_pred),
        grad_norm=_sub(data.grad_norm),
        slice_id=_sub(data.slice_id),
        local_row=_sub(data.local_row),
        folders=list(data.folders),
        features=_sub(data.features),
        cluster_seed=seed,
        requested_clusters=k,
        method=data.method,
        algorithm=data.algorithm,
        source=data.source,
        n_source_rows=data.n_source_rows,
    )
    new.projection_cache.clear()
    try:
        reproject(new, method, seed=seed)
    except ValueError:
        pass
    recluster(new, k, seed=seed, algorithm=algorithm)
    return new


def default_work_dir(root: Path) -> Path | None:
    moe = root / "data" / "processed" / "board_eval" / "moe"
    for name in ("moe_b3_1m", "moe_b4_2m", "moe_smoke2", "moe_smoke"):
        path = moe / name
        if (path / "gradients.npy").is_file() and (path / "labels.npy").is_file():
            return path
    if moe.is_dir():
        for child in sorted(moe.iterdir()):
            if (child / "gradients.npy").is_file() and (child / "labels.npy").is_file():
                return child
    return None
