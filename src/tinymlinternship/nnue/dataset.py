"""``{fen, wdl, visits}`` tables → sparse dual-POV batches for NNUE.

844 features are encoded **once** per source table into a ``*.nnue/`` cache
next to the file, then loaded as in-memory tensors on later runs.
Target is STM ``wdl`` of shape ``(N, 3)``. Slices that only store White-POV
``value`` are converted with the max-entropy WDL map.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from tinymlinternship.data.wdl import wdl_from_row
from tinymlinternship.features import FEATURE_DIM, encode_dual

MAX_ACTIVE_FEATURES = 128
CACHE_VERSION = 2
CACHE_ARRAYS = (
    "white_idx",
    "white_n",
    "black_idx",
    "black_n",
    "stm",
    "wdl",
    "visits",
    "epd_hash",
)
# Per-slice training DB (no visits): sparse 844 + STM WDL.
SLICE_DB_ARRAYS = (
    "white_idx",
    "white_n",
    "black_idx",
    "black_n",
    "stm",
    "wdl",
)
SLICE_FEATURES_NPZ = "features.npz"
SLICE_FEATURES_META = "features.meta.json"
SLICE_DB_VERSION = 2

# fen_value_visits_<dump>_<start>-<end>[_d90][_m10][_draw5]
_DUMP_RANGE_RE = re.compile(
    r"^(?P<prefix>fen_value_visits_.+)_(?P<start>\d+)-(?P<end>\d+)"
    r"(?P<extra>(?:_d\d+)?(?:_m\d+)?(?:_draw\d+)?)?$"
)


def parse_dump_game_range(name: str) -> tuple[str, int, int] | None:
    """``(dump prefix, start, end)`` for a Lichess-dump slice stem, else ``None``."""
    match = _DUMP_RANGE_RE.fullmatch(str(name))
    if match is None:
        return None
    return match.group("prefix"), int(match.group("start")), int(match.group("end"))


def overlapping_dump_slices(holdout: str, names: list[str]) -> set[str]:
    """Holdout folder plus any sibling dump slices whose ``[start, end)`` overlaps.

    ``…_100000-105000_d80_draw5`` and ``…_100000-105000_d99`` are the same games
    with different ply filters; skipping only the exact test name leaks EPDs.
    """
    skip = {str(holdout)}
    parsed = parse_dump_game_range(holdout)
    if parsed is None:
        return skip
    src, start, end = parsed
    for name in names:
        other = parse_dump_game_range(name)
        if other is None:
            continue
        src2, a, b = other
        if src2 == src and start < b and a < end:
            skip.add(name)
    return skip


def apply_holdout_skip(
    folders: list[Path], skip_names: set[str] | None
) -> tuple[list[Path], set[str]]:
    """Drop holdout + overlapping dump-range siblings. Returns (kept, skipped)."""
    if not skip_names:
        return folders, set()
    names = [folder.name for folder in folders]
    skipped: set[str] = set()
    for name in skip_names:
        skipped |= overlapping_dump_slices(name, names)
    kept = [folder for folder in folders if folder.name not in skipped]
    return kept, skipped


def last_fraction_cut(n: int, fraction: float) -> int:
    """First test index when the last ``fraction`` of ``n`` rows is held out.

    Train is ``[0, cut)``, test is ``[cut, n)``. ``n_test = int(n * fraction)``
    (floor). Tiny slices may have ``cut == n`` (no test rows).
    """
    n = int(n)
    if n <= 0:
        return 0
    frac = float(fraction)
    if frac <= 0.0:
        return n
    if frac >= 1.0:
        return 0
    return n - int(n * frac)


def split_last_fraction(
    slices: list[FenValueVisitsDataset],
    fraction: float,
) -> tuple[list[FenValueVisitsDataset], list[FenValueVisitsDataset]]:
    """Per-slice: last ``fraction`` of rows → test, the rest → train.

    Holding out a whole slice made train/test difficulty differ (e.g. extreme
    positions). Splitting every slice keeps the same mix on both sides.
    """
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError(f"test fraction must be in (0, 1), got {fraction}")
    train_parts: list[FenValueVisitsDataset] = []
    test_parts: list[FenValueVisitsDataset] = []
    for dataset in slices:
        n = len(dataset)
        cut = last_fraction_cut(n, fraction)
        if cut > 0:
            train_parts.append(dataset.narrow_rows(0, cut))
        if cut < n:
            test_parts.append(dataset.narrow_rows(cut, n))
    return train_parts, test_parts


def split_random_fraction(
    slices: list[FenValueVisitsDataset],
    fraction: float,
    *,
    seed: int = 0,
) -> tuple[list[FenValueVisitsDataset], list[FenValueVisitsDataset]]:
    """Per-slice: random ``fraction`` of rows → test, the rest → train.

    Stratified i.i.d. split: every slice type appears in both sets, and train
    and test have the same difficulty in expectation.
    """
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError(f"test fraction must be in (0, 1), got {fraction}")
    rng = np.random.RandomState(int(seed))
    train_parts: list[FenValueVisitsDataset] = []
    test_parts: list[FenValueVisitsDataset] = []
    for dataset in slices:
        n = len(dataset)
        n_test = int(n * float(fraction))
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        if train_idx.size > 0:
            train_parts.append(dataset.take_rows(train_idx))
        if test_idx.size > 0:
            test_parts.append(dataset.take_rows(test_idx))
    return train_parts, test_parts


def epd_key(fen: str) -> str:
    """Board + STM + castling + EP (clocks dropped)."""
    parts = str(fen).split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return str(fen).strip()


def epd_hash64(epd: str) -> np.uint64:
    digest = hashlib.blake2b(epd.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def prefer_parquet(path: Path) -> Path:
    """Use the parquet twin when both exist (same stem)."""
    if path.suffix.lower() == ".parquet":
        return path
    twin = path.with_suffix(".parquet")
    if twin.is_file():
        return twin
    nested = path.parent / "fen_value_visits" / f"{path.stem}.parquet"
    if nested.is_file():
        return nested
    return path


def load_fen_value_visits(path: Path) -> pd.DataFrame:
    path = prefer_parquet(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_parquet(path)
    if "fen" not in df.columns:
        raise ValueError(f"{path} missing column fen")
    if "wdl" not in df.columns and "value" not in df.columns:
        raise ValueError(f"{path} missing wdl and value")
    df = df.copy()
    if "visits" not in df.columns:
        df["visits"] = 1
    cols = ["fen", "visits"]
    if "wdl" in df.columns:
        cols.append("wdl")
    if "value" in df.columns:
        cols.append("value")
    return df[cols].copy()


def default_cache_dir(source: Path) -> Path:
    return Path(str(source) + ".nnue")


def _pad_indices(indices: list[int], width: int) -> tuple[np.ndarray, int]:
    if len(indices) > width:
        raise ValueError(f"active features {len(indices)} exceed pad width {width}")
    idx = np.zeros(width, dtype=np.int16)
    n = len(indices)
    if n:
        idx[:n] = np.asarray(indices, dtype=np.int16)
    return idx, n


def _source_stat(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "source": str(path.resolve()),
        "source_size": int(st.st_size),
        "source_mtime_ns": int(st.st_mtime_ns),
    }


def cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "meta.json"


def cache_is_valid(cache_dir: Path, source: Path, *, max_active: int) -> bool:
    meta_path = cache_meta_path(cache_dir)
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = _source_stat(source)
    if int(meta.get("version", -1)) != CACHE_VERSION:
        return False
    if int(meta.get("max_active", -1)) != int(max_active):
        return False
    if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
        return False
    if meta.get("source") != expected["source"]:
        return False
    if int(meta.get("source_size", -1)) != expected["source_size"]:
        return False
    if int(meta.get("source_mtime_ns", -1)) != expected["source_mtime_ns"]:
        return False
    n_rows = int(meta.get("n_rows", -1))
    if n_rows < 0:
        return False
    if "value" in meta.get("columns", []) and "wdl" not in meta.get("columns", []):
        return False
    for name in CACHE_ARRAYS:
        if not (cache_dir / f"{name}.npy").is_file():
            return False
    return True


def encode_frames(
    df: pd.DataFrame,
    *,
    max_active: int = MAX_ACTIVE_FEATURES,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Encode unique EPDs to padded sparse arrays (no cache I/O)."""
    work = df.copy()
    if "visits" not in work.columns:
        work["visits"] = 1
    work["epd"] = work["fen"].map(epd_key)
    work = work.drop_duplicates(subset=["epd"], keep="first").reset_index(drop=True)
    n = len(work)
    white_idx = np.zeros((n, max_active), dtype=np.int16)
    white_n = np.zeros(n, dtype=np.uint8)
    black_idx = np.zeros((n, max_active), dtype=np.int16)
    black_n = np.zeros(n, dtype=np.uint8)
    stm = np.zeros(n, dtype=np.uint8)
    wdl = np.zeros((n, 3), dtype=np.float32)
    visits = work["visits"].to_numpy(dtype=np.float32)
    hashes = np.zeros(n, dtype=np.uint64)

    iterator = range(n)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="encode 844", unit="pos", total=n)

    records = work.to_dict(orient="records")
    epds = work["epd"].tolist()
    for i in iterator:
        rec = records[i]
        board = chess.Board(str(rec["fen"]))
        white, black = encode_dual(board)
        w_idx, w_n = _pad_indices(white, max_active)
        b_idx, b_n = _pad_indices(black, max_active)
        white_idx[i] = w_idx
        white_n[i] = w_n
        black_idx[i] = b_idx
        black_n[i] = b_n
        stm[i] = 1 if board.turn == chess.WHITE else 0
        wdl[i] = np.asarray(wdl_from_row(rec), dtype=np.float32)
        hashes[i] = epd_hash64(epds[i])
    return {
        "white_idx": white_idx,
        "white_n": white_n,
        "black_idx": black_idx,
        "black_n": black_n,
        "stm": stm,
        "wdl": wdl,
        "visits": visits,
        "epd_hash": hashes,
    }


def save_feature_cache(
    cache_dir: Path,
    arrays: dict[str, np.ndarray],
    *,
    source: Path,
    max_active: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in CACHE_ARRAYS:
        np.save(cache_dir / f"{name}.npy", arrays[name])
    meta = {
        "version": CACHE_VERSION,
        "max_active": int(max_active),
        "feature_dim": FEATURE_DIM,
        "n_rows": int(arrays["wdl"].shape[0]),
        "columns": list(CACHE_ARRAYS),
        **_source_stat(source),
    }
    cache_meta_path(cache_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_feature_cache(cache_dir: Path) -> dict[str, np.ndarray]:
    """Load arrays into RAM (not memmap) so later epochs stay off disk."""
    arrays: dict[str, np.ndarray] = {}
    for name in CACHE_ARRAYS:
        arrays[name] = np.load(cache_dir / f"{name}.npy")
    return arrays


def ensure_feature_cache(
    source: Path,
    *,
    cache_dir: Path | None = None,
    max_active: int = MAX_ACTIVE_FEATURES,
    rebuild: bool = False,
    progress: bool = True,
) -> tuple[Path, dict[str, np.ndarray]]:
    """Return (cache_dir, arrays). Encodes and writes only if missing or stale."""
    source = prefer_parquet(source)
    cache_dir = cache_dir or default_cache_dir(source)
    if not rebuild and cache_is_valid(cache_dir, source, max_active=max_active):
        return cache_dir, load_feature_cache(cache_dir)
    df = load_fen_value_visits(source)
    arrays = encode_frames(df, max_active=max_active, progress=progress)
    save_feature_cache(cache_dir, arrays, source=source, max_active=max_active)
    return cache_dir, arrays


def slice_source_json(folder: Path) -> Path | None:
    """JSON (or parquet twin) inside a per-slice folder."""
    named = folder / f"{folder.name}.json"
    if named.is_file():
        return prefer_parquet(named)
    parquet = folder / f"{folder.name}.parquet"
    if parquet.is_file():
        return parquet
    jsons = sorted(
        p
        for p in folder.glob("*.json")
        if p.name not in {SLICE_FEATURES_META, "meta.json"}
    )
    if jsons:
        return prefer_parquet(jsons[0])
    return None


def slice_features_path(folder: Path) -> Path:
    return folder / SLICE_FEATURES_NPZ


def slice_db_is_valid(folder: Path, source: Path, *, max_active: int) -> bool:
    npz = slice_features_path(folder)
    meta_path = folder / SLICE_FEATURES_META
    if not npz.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = _source_stat(source)
    if int(meta.get("version", -1)) != SLICE_DB_VERSION:
        return False
    if int(meta.get("max_active", -1)) != int(max_active):
        return False
    if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
        return False
    if "visits" in meta.get("columns", []):
        return False
    if meta.get("source") != expected["source"]:
        return False
    if int(meta.get("source_size", -1)) != expected["source_size"]:
        return False
    if int(meta.get("source_mtime_ns", -1)) != expected["source_mtime_ns"]:
        return False
    return True


def save_slice_feature_db(
    folder: Path,
    arrays: dict[str, np.ndarray],
    *,
    source: Path,
    max_active: int,
) -> Path:
    payload = {name: arrays[name] for name in SLICE_DB_ARRAYS}
    npz = slice_features_path(folder)
    np.savez(npz, **payload)
    meta = {
        "version": SLICE_DB_VERSION,
        "format": "sparse dual-POV 844 + STM WDL",
        "columns": list(SLICE_DB_ARRAYS),
        "max_active": int(max_active),
        "feature_dim": FEATURE_DIM,
        "n_rows": int(payload["wdl"].shape[0]),
        **_source_stat(source),
    }
    (folder / SLICE_FEATURES_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz


def load_slice_feature_db(folder: Path) -> dict[str, np.ndarray]:
    with np.load(slice_features_path(folder)) as npz:
        missing = set(SLICE_DB_ARRAYS) - set(npz.files)
        if missing:
            raise ValueError(
                f"{folder} features.npz missing {sorted(missing)}; "
                "re-encode with scripts/encode_slice_features.py --rebuild"
            )
        wdl = npz["wdl"]
        if wdl.ndim != 2 or wdl.shape[1] != 3:
            raise ValueError(f"{folder} features.npz wdl must be (N, 3), got {wdl.shape}")
        return {name: np.array(npz[name]) for name in SLICE_DB_ARRAYS}


def ensure_slice_feature_db(
    folder: Path,
    *,
    max_active: int = MAX_ACTIVE_FEATURES,
    rebuild: bool = False,
    progress: bool = True,
) -> tuple[Path, dict[str, np.ndarray]]:
    """Encode ``folder/*.json`` → ``folder/features.npz`` (no visits)."""
    source = slice_source_json(folder)
    if source is None:
        raise FileNotFoundError(f"no slice JSON in {folder}")
    npz = slice_features_path(folder)
    if not rebuild and slice_db_is_valid(folder, source, max_active=max_active):
        return npz, load_slice_feature_db(folder)
    df = load_fen_value_visits(source)
    arrays = encode_frames(df, max_active=max_active, progress=progress)
    save_slice_feature_db(folder, arrays, source=source, max_active=max_active)
    return npz, {name: arrays[name] for name in SLICE_DB_ARRAYS}


def discover_slice_folders(root: Path) -> list[Path]:
    """Immediate child dirs that contain a slice JSON."""
    if not root.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and slice_source_json(child) is not None:
            found.append(child)
    return found


def concat_slice_arrays(array_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not array_list:
        raise ValueError("no slice feature arrays to concatenate")
    return {
        name: np.concatenate([block[name] for block in array_list], axis=0)
        for name in SLICE_DB_ARRAYS
    }


def _mask_from_counts(counts: np.ndarray, width: int) -> np.ndarray:
    return np.arange(width, dtype=np.int32)[None, :] < counts[:, None]


def _subset_arrays(
    arrays: dict[str, np.ndarray],
    *,
    exclude_hashes: set[int] | None = None,
    max_rows: int = 0,
) -> dict[str, np.ndarray]:
    n = int(arrays["wdl"].shape[0])
    keep = np.ones(n, dtype=np.bool_)
    if exclude_hashes:
        if "epd_hash" not in arrays:
            raise ValueError("exclude_epds needs epd_hash in the feature arrays")
        drop_vals = np.fromiter(exclude_hashes, dtype=np.uint64, count=len(exclude_hashes))
        keep &= ~np.isin(arrays["epd_hash"], drop_vals)
    idx = np.flatnonzero(keep)
    if max_rows and max_rows > 0 and idx.size > int(max_rows):
        rng = np.random.RandomState(0)
        idx = rng.choice(idx, size=int(max_rows), replace=False)
        idx.sort()
    return {name: arrays[name][idx] for name in arrays}


class FenValueVisitsDataset(Dataset):
    """In-memory sparse 844 features (from cache or a one-shot encode)."""

    def __init__(
        self,
        table: pd.DataFrame | Path,
        *,
        exclude_epds: set[str] | None = None,
        max_rows: int = 0,
        max_active: int = MAX_ACTIVE_FEATURES,
        progress: bool = False,
        cache_dir: Path | None = None,
        rebuild_cache: bool = False,
        use_cache: bool = True,
    ) -> None:
        self.max_active = max_active
        self.feature_dim = FEATURE_DIM
        exclude_hashes = {int(epd_hash64(e)) for e in exclude_epds} if exclude_epds else None

        if isinstance(table, Path) and table.is_dir():
            _, arrays = ensure_slice_feature_db(
                table,
                max_active=max_active,
                rebuild=rebuild_cache,
                progress=progress,
            )
            if exclude_hashes:
                raise ValueError("slice feature DBs have no EPD key; exclude by omitting the folder")
            arrays = _subset_arrays(arrays, max_rows=max_rows)
            self.cache_dir = table
            self._attach_arrays(arrays)
            return

        if isinstance(table, Path) and use_cache:
            source = prefer_parquet(table)
            _cache_dir, arrays = ensure_feature_cache(
                source,
                cache_dir=cache_dir,
                max_active=max_active,
                rebuild=rebuild_cache,
                progress=progress,
            )
            self.cache_dir = _cache_dir
            arrays = _subset_arrays(arrays, exclude_hashes=exclude_hashes, max_rows=max_rows)
        else:
            self.cache_dir = None
            if isinstance(table, Path):
                df = load_fen_value_visits(table)
            else:
                df = table.copy()
            if exclude_epds:
                df = df.copy()
                df["epd"] = df["fen"].map(epd_key)
                df = df.loc[~df["epd"].isin(exclude_epds)]
            arrays = encode_frames(df, max_active=max_active, progress=progress)
            arrays = _subset_arrays(arrays, max_rows=max_rows)

        self._attach_arrays(arrays)

    def _attach_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        if "wdl" not in arrays:
            raise ValueError(
                "features.npz is missing 'wdl' (N, 3); re-encode with "
                "scripts/encode_slice_features.py --rebuild"
            )
        wdl = np.ascontiguousarray(arrays["wdl"], dtype=np.float32)
        if wdl.ndim != 2 or wdl.shape[1] != 3:
            raise ValueError(f"wdl must have shape (N, 3), got {wdl.shape}")
        wdl = np.clip(wdl, 0.0, None)
        denom = wdl.sum(axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-8)
        wdl = wdl / denom
        self._np = arrays
        width = int(arrays["white_idx"].shape[1])
        self.white_idx = torch.from_numpy(np.ascontiguousarray(arrays["white_idx"]))
        self.black_idx = torch.from_numpy(np.ascontiguousarray(arrays["black_idx"]))
        self.white_mask = torch.from_numpy(_mask_from_counts(arrays["white_n"], width))
        self.black_mask = torch.from_numpy(_mask_from_counts(arrays["black_n"], width))
        self.stm_white = torch.from_numpy(arrays["stm"].astype(np.bool_))
        self.wdl = torch.from_numpy(wdl)
        if "visits" in arrays:
            self.visits = torch.from_numpy(np.ascontiguousarray(arrays["visits"]))
        else:
            self.visits = torch.ones(self.wdl.shape[0], dtype=torch.float32)

    @classmethod
    def from_slice_root(
        cls,
        root: Path,
        *,
        skip_names: set[str] | None = None,
        max_rows: int = 0,
        max_active: int = MAX_ACTIVE_FEATURES,
        rebuild: bool = False,
        progress: bool = True,
    ) -> FenValueVisitsDataset:
        """Load every ``features.npz`` under ``root`` (folders stay on disk, concat in RAM)."""
        folders = discover_slice_folders(root)
        folders, _skipped = apply_holdout_skip(folders, skip_names)
        if not folders:
            raise FileNotFoundError(f"no slice JSON folders in {root}")
        blocks: list[dict[str, np.ndarray]] = []
        for folder in folders:
            _, arrays = ensure_slice_feature_db(
                folder,
                max_active=max_active,
                rebuild=rebuild,
                progress=progress,
            )
            blocks.append(arrays)
        merged = concat_slice_arrays(blocks)
        merged = _subset_arrays(merged, max_rows=max_rows)
        ds = cls.__new__(cls)
        ds.max_active = max_active
        ds.feature_dim = FEATURE_DIM
        ds.cache_dir = root
        ds._attach_arrays(merged)
        return ds

    @classmethod
    def load_slice_datasets(
        cls,
        root: Path,
        *,
        skip_names: set[str] | None = None,
        max_active: int = MAX_ACTIVE_FEATURES,
        rebuild: bool = False,
        progress: bool = True,
    ) -> list[FenValueVisitsDataset]:
        """One in-memory dataset per slice folder (no concat)."""
        folders = discover_slice_folders(root)
        folders, _skipped = apply_holdout_skip(folders, skip_names)
        if not folders:
            raise FileNotFoundError(f"no slice JSON folders in {root}")
        out: list[FenValueVisitsDataset] = []
        for folder in folders:
            out.append(
                cls(
                    folder,
                    max_active=max_active,
                    rebuild_cache=rebuild,
                    progress=progress,
                )
            )
        return out

    def __len__(self) -> int:
        return int(self.wdl.shape[0])

    def take_rows(self, indices: np.ndarray) -> FenValueVisitsDataset:
        """New dataset with the given row indices (order preserved)."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        n = len(self)
        if idx.size == 0:
            raise ValueError("take_rows needs at least one index")
        if idx.min() < 0 or idx.max() >= n:
            raise ValueError(f"row index out of range for n={n}")
        torch_idx = torch.from_numpy(idx)
        clone = FenValueVisitsDataset.__new__(FenValueVisitsDataset)
        clone.max_active = self.max_active
        clone.feature_dim = self.feature_dim
        clone.cache_dir = self.cache_dir
        clone._np = {key: value[idx] for key, value in self._np.items()}
        clone.white_idx = self.white_idx[torch_idx]
        clone.black_idx = self.black_idx[torch_idx]
        clone.white_mask = self.white_mask[torch_idx]
        clone.black_mask = self.black_mask[torch_idx]
        clone.stm_white = self.stm_white[torch_idx]
        clone.wdl = self.wdl[torch_idx]
        clone.visits = self.visits[torch_idx]
        return clone

    def narrow_rows(self, start: int, end: int) -> FenValueVisitsDataset:
        """View of rows ``[start, end)``."""
        start_i = int(start)
        end_i = int(end)
        n = len(self)
        if start_i < 0 or end_i > n or start_i >= end_i:
            raise ValueError(f"invalid row range [{start_i}, {end_i}) for n={n}")
        return self.take_rows(np.arange(start_i, end_i, dtype=np.int64))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "white_idx": self.white_idx[index],
            "white_mask": self.white_mask[index],
            "black_idx": self.black_idx[index],
            "black_mask": self.black_mask[index],
            "stm_white": self.stm_white[index],
            "target": self.wdl[index],
            "weight": self.visits[index],
        }


class MixedSliceDataset(Dataset):
    """Lookup ``(slice, row)`` packed as ``slice << 32 | row``."""

    def __init__(self, slices: list[FenValueVisitsDataset]) -> None:
        if not slices:
            raise ValueError("need at least one train slice")
        self.slices = slices
        self.sizes = [len(item) for item in slices]

    def __len__(self) -> int:
        return int(sum(self.sizes))

    def __getitem__(self, packed: int) -> dict[str, torch.Tensor]:
        slice_id = int(packed) >> 32
        row = int(packed) & 0xFFFFFFFF
        return self.slices[slice_id][row]


class ConcatSliceDataset(Dataset):
    """Linear ``0 .. N-1`` over concatenated slices (sequential test eval)."""

    def __init__(self, slices: list[FenValueVisitsDataset]) -> None:
        if not slices:
            raise ValueError("need at least one slice")
        self.slices = slices
        self.sizes = [len(item) for item in slices]
        offsets = [0]
        acc = 0
        for size in self.sizes:
            acc += int(size)
            offsets.append(acc)
        self._offsets = offsets

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        i = int(index)
        n = len(self)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(index)
        slice_id = bisect.bisect_right(self._offsets, i) - 1
        row = i - self._offsets[slice_id]
        return self.slices[slice_id][row]


class UniformRowBatchSampler(Sampler[list[int]]):
    """Each batch is ``batch_size`` i.i.d. draws uniform over ``0 .. n-1``.

    Large slices appear in proportion to their size (same measure as test CE).
    """

    def __init__(
        self,
        n: int,
        *,
        batch_size: int,
        batches: int,
        seed: int = 0,
    ) -> None:
        if int(n) < 1:
            raise ValueError("n must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batches < 1:
            raise ValueError("batches must be >= 1")
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.batches = int(batches)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches):
            yield rng.randint(0, self.n, size=self.batch_size).tolist()


class AcrossSliceBatchSampler(Sampler[list[int]]):
    """Each batch is ``batch_size`` draws: uniform over slices, then uniform over rows.

    Small slices (puzzles, mini) appear as often as large dump slices.
    """

    def __init__(
        self,
        sizes: list[int],
        *,
        batch_size: int,
        batches: int,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batches < 1:
            raise ValueError("batches must be >= 1")
        if not sizes or min(sizes) < 1:
            raise ValueError("every slice must be non-empty")
        self.sizes = list(sizes)
        self.batch_size = int(batch_size)
        self.batches = int(batches)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        n_slices = len(self.sizes)
        for _ in range(self.batches):
            batch: list[int] = []
            for _ in range(self.batch_size):
                slice_id = int(rng.randint(0, n_slices))
                row = int(rng.randint(0, self.sizes[slice_id]))
                batch.append((slice_id << 32) | row)
            yield batch


def collate_sparse(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    stacked = {
        key: torch.stack([item[key] for item in batch])
        for key in (
            "white_idx",
            "white_mask",
            "black_idx",
            "black_mask",
            "stm_white",
            "target",
            "weight",
        )
    }
    stacked["white_idx"] = stacked["white_idx"].long()
    stacked["black_idx"] = stacked["black_idx"].long()
    return stacked
