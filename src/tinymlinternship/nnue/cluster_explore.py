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


def cluster_color(cluster_id: int, n_clusters: int | None = None) -> str:
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
    source: str = ""
    n_source_rows: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

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
        if self.requested_clusters > 0:
            return int(self.requested_clusters)
        if self.cluster_id.size == 0:
            return 0
        return int(self.cluster_id.max()) + 1

    def visible_mask(
        self,
        *,
        clusters: Iterable[int] | None = None,
        slice_substr: str = "",
    ) -> np.ndarray:
        mask = np.ones(len(self), dtype=np.bool_)
        if clusters is not None:
            allowed = np.fromiter((int(c) for c in clusters), dtype=np.int16)
            mask &= np.isin(self.cluster_id, allowed)
        needle = str(slice_substr or "").strip().lower()
        if needle and self.folders and self.slice_id is not None:
            names = [p.name.lower() for p in self.folders]
            hit = np.zeros(len(self), dtype=np.bool_)
            for sid, name in enumerate(names):
                if needle in name:
                    hit |= self.slice_id.astype(np.int64) == sid
            mask &= hit
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
    """FIFO multi-select with toggle-off. ``max_n`` is the pin budget."""

    def __init__(self, max_n: int = 5) -> None:
        self._max_n = max(1, int(max_n))
        self.order: list[int] = []

    @property
    def max_n(self) -> int:
        return self._max_n

    @max_n.setter
    def max_n(self, value: int) -> None:
        self._max_n = max(1, int(value))
        while len(self.order) > self._max_n:
            self.order.pop(0)

    def __contains__(self, index: int) -> bool:
        return int(index) in self.order

    def __len__(self) -> int:
        return len(self.order)

    def toggle(self, index: int) -> tuple[list[int], int | None]:
        idx = int(index)
        if idx in self.order:
            self.order.remove(idx)
            return list(self.order), None
        dropped: int | None = None
        if len(self.order) >= self._max_n:
            dropped = self.order.pop(0)
        self.order.append(idx)
        return list(self.order), dropped

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


def recluster(
    data: ExplorerData,
    n_clusters: int,
    *,
    seed: int | None = None,
) -> ExplorerData:
    """Fit mini-batch k-means in place. Scatter coordinates are unchanged."""
    from tinymlinternship.nnue.cluster import cluster_diagnostics, fit_minibatch_kmeans

    x = clustering_features(data)
    k = int(n_clusters)
    if k < 2:
        raise ValueError(f"n_clusters must be >= 2, got {k}")
    if int(x.shape[0]) < k:
        raise ValueError(f"need at least {k} points to fit {k} clusters, got {x.shape[0]}")
    rng_seed = int(data.cluster_seed if seed is None else seed)
    km = fit_minibatch_kmeans(
        x,
        k,
        batch_size=min(10_000, int(x.shape[0])),
        seed=rng_seed,
    )
    labels = km.predict(x).astype(np.int16, copy=False)
    diag = cluster_diagnostics(labels, km.cluster_centers_, inertia=float(km.inertia_))
    data.cluster_id = labels
    data.requested_clusters = k
    data.cluster_seed = rng_seed
    data.diagnostics.update(diag)
    data.diagnostics["clustered_on"] = "gradients" if data.features is not None else "projection"
    return data


def project_gradients(
    gradients: np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 2,
    seed: int = 0,
) -> np.ndarray:
    method = str(method).lower().strip()
    x = np.ascontiguousarray(gradients, dtype=np.float32)
    k = max(2, int(n_components))
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP is not installed; pip install umap-learn") from exc
        reducer = umap.UMAP(n_components=k, random_state=int(seed), metric="euclidean")
        return np.asarray(reducer.fit_transform(x), dtype=np.float32)
    from sklearn.decomposition import PCA

    k = min(k, int(x.shape[1]), int(x.shape[0]))
    coords = PCA(n_components=k, random_state=int(seed)).fit_transform(x)
    return np.asarray(coords, dtype=np.float32)


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
    max_points: int = 30_000,
    method: str = "pca",
    seed: int = 0,
    n_components: int = 2,
    n_clusters: int | None = 4,
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
        method=str(method),
        source=str(work_dir),
        n_source_rows=n,
        diagnostics=diagnostics,
    )
    if n_clusters is not None:
        recluster(data, int(n_clusters), seed=int(seed))
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
    fens = np.empty(n, dtype=object)
    for i in range(n):
        board = start.copy()
        ply = int(rng.randint(2, 28))
        for _ in range(ply):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.randint(0, len(moves)))])
        fens[i] = board.fen()

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
        method="demo",
        source="demo",
        n_source_rows=n,
        diagnostics={"n_clusters": k, "sizes": np.bincount(cluster_id, minlength=k).tolist()},
    )


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
