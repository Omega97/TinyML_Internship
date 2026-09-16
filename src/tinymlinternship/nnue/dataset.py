"""``{fen, wdl, visits}`` tables → sparse dual-POV batches for NNUE.

844 features are encoded **once** per source table into a ``*.nnue/`` cache
or per-slice ``features.npz``. Training reads those arrays with memory-mapping
so 60M+ rows stay on disk; only the current batch is copied into RAM.
Target is STM ``wdl`` of shape ``(N, 3)``. Slices that only store White-POV
``value`` are converted with the max-entropy WDL map.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pandas as pd
import torch
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic
from torch.utils.data import Dataset, Sampler, Subset

from tinymlinternship.data.wdl import wdl_from_row
from tinymlinternship.features import FEATURE_DIM, encode_dual

# ZIP local-file header (PK\x03\x04). ``np.load(..., mmap_mode="r")`` ignores
# mmap for ``.npz`` (a zip); we parse this header and ``np.memmap`` the
# uncompressed ``.npy`` payload instead.
_ZIP_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_ZIP_LOCAL_HEADER_SIG = 0x04034B50

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
    """Memory-map cache ``.npy`` files (``mmap_mode='r'``); pages fault in on slice."""
    arrays: dict[str, np.ndarray] = {}
    for name in CACHE_ARRAYS:
        arrays[name] = np.load(cache_dir / f"{name}.npy", mmap_mode="r")
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
    return cache_dir, load_feature_cache(cache_dir)


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


def _mmap_npz(path: Path) -> dict[str, np.ndarray]:
    """Memory-map uncompressed ``.npz`` members without copying arrays into RAM.

    ``np.load(path, mmap_mode="r")`` does **not** mmap a zip archive (numpy
    extracts each ``.npy`` into a regular ndarray). ``np.savez`` stores
    uncompressed members, so we can ``np.memmap`` the payload in place.
    Compressed members fall back to an in-memory load of that file only.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if info.filename.endswith(".npy")]
        compressed = [info.filename for info in infos if info.compress_type != zipfile.ZIP_STORED]
    if compressed:
        with np.load(path) as npz:
            return {info.filename[:-4]: np.array(npz[info.filename[:-4]]) for info in infos}

    arrays: dict[str, np.ndarray] = {}
    with path.open("rb") as fh:
        for info in infos:
            fh.seek(int(info.header_offset))
            hdr = fh.read(_ZIP_LOCAL_HEADER.size)
            if len(hdr) != _ZIP_LOCAL_HEADER.size:
                raise ValueError(f"{path} truncated zip header for {info.filename}")
            sig, _ver, flag, _comp, _mt, _md, _crc, _cs, _us, namelen, extralen = (
                _ZIP_LOCAL_HEADER.unpack(hdr)
            )
            if sig != _ZIP_LOCAL_HEADER_SIG:
                raise ValueError(f"{path} bad zip local header for {info.filename}")
            if flag & 1:
                raise ValueError(f"{path} encrypted zip member {info.filename}")
            fh.read(int(namelen))
            fh.read(int(extralen))
            version = read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype = read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, fortran, dtype = read_array_header_2_0(fh)
            else:
                raise ValueError(f"{path} unsupported npy version {version} in {info.filename}")
            key = info.filename[:-4]
            arrays[key] = np.memmap(
                path,
                dtype=dtype,
                mode="r",
                offset=int(fh.tell()),
                shape=shape,
                order="F" if fortran else "C",
            )
    return arrays


def load_slice_feature_db(folder: Path) -> dict[str, np.ndarray]:
    """Memory-map ``folder/features.npz`` (no full-array copy into RAM)."""
    path = slice_features_path(folder)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; encode with scripts/encode_slice_features.py"
        )
    arrays = _mmap_npz(path)
    missing = set(SLICE_DB_ARRAYS) - set(arrays)
    if missing:
        raise ValueError(
            f"{folder} features.npz missing {sorted(missing)}; "
            "re-encode with scripts/encode_slice_features.py --rebuild"
        )
    wdl = arrays["wdl"]
    if wdl.ndim != 2 or wdl.shape[1] != 3:
        raise ValueError(f"{folder} features.npz wdl must be (N, 3), got {wdl.shape}")
    keep = [name for name in arrays if name in SLICE_DB_ARRAYS or name == "visits"]
    return {name: arrays[name] for name in keep}


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
    # Re-open as memmap so the encode buffers can be freed.
    return npz, load_slice_feature_db(folder)


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
    if not exclude_hashes and not (max_rows and max_rows > 0):
        return arrays
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
    if idx.size == n and bool(np.all(idx == np.arange(n, dtype=idx.dtype))):
        return arrays
    return {name: arrays[name][idx] for name in arrays}


def _as_numpy_index(index: torch.Tensor | np.ndarray | int) -> np.ndarray:
    if isinstance(index, torch.Tensor):
        return index.detach().to(dtype=torch.long, device="cpu").numpy().reshape(-1)
    return np.asarray(index, dtype=np.int64).reshape(-1)


class FenValueVisitsDataset(Dataset):
    """Sparse 844 features backed by memory-mapped ``.npy`` / ``features.npz``.

    Arrays stay on disk. ``__getitem__`` / ``gather`` copy only the requested
    rows into tensors. Train/test splits store an index map; they do not pack
    a second copy of the slice.
    """

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

    def _attach_arrays(
        self,
        arrays: dict[str, np.ndarray],
        row_index: np.ndarray | None = None,
    ) -> None:
        if "wdl" not in arrays:
            raise ValueError(
                "features.npz is missing 'wdl' (N, 3); re-encode with "
                "scripts/encode_slice_features.py --rebuild"
            )
        wdl = arrays["wdl"]
        if wdl.ndim != 2 or wdl.shape[1] != 3:
            raise ValueError(f"wdl must have shape (N, 3), got {wdl.shape}")
        self._np = arrays
        self._width = int(arrays["white_idx"].shape[1])
        if row_index is None:
            self._row_index = None
            self._n = int(wdl.shape[0])
        else:
            idx = np.asarray(row_index, dtype=np.int64).reshape(-1)
            n_src = int(wdl.shape[0])
            if idx.size == 0:
                raise ValueError("row index is empty")
            if int(idx.min()) < 0 or int(idx.max()) >= n_src:
                raise ValueError(f"row index out of range for n={n_src}")
            self._row_index = idx
            self._n = int(idx.shape[0])

    def _physical_rows(self, logical: np.ndarray) -> np.ndarray:
        idx = np.asarray(logical, dtype=np.int64).reshape(-1)
        if self._row_index is None:
            return idx
        return self._row_index[idx]

    def _take_np(self, name: str) -> np.ndarray:
        arr = self._np[name]
        if self._row_index is None:
            return arr
        return arr[self._row_index]

    @property
    def white_idx(self) -> torch.Tensor:
        return torch.from_numpy(np.array(self._take_np("white_idx"), copy=True, order="C"))

    @property
    def black_idx(self) -> torch.Tensor:
        return torch.from_numpy(np.array(self._take_np("black_idx"), copy=True, order="C"))

    @property
    def white_mask(self) -> torch.Tensor:
        counts = np.array(self._take_np("white_n"), copy=True, order="C")
        return torch.from_numpy(_mask_from_counts(counts, self._width))

    @property
    def black_mask(self) -> torch.Tensor:
        counts = np.array(self._take_np("black_n"), copy=True, order="C")
        return torch.from_numpy(_mask_from_counts(counts, self._width))

    @property
    def stm_white(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self._take_np("stm")).astype(np.bool_, copy=True))

    @property
    def wdl(self) -> torch.Tensor:
        return torch.from_numpy(_normalize_wdl_np(self._take_np("wdl")))

    @property
    def visits(self) -> torch.Tensor:
        if "visits" in self._np:
            return torch.from_numpy(
                np.array(self._take_np("visits"), copy=True, order="C", dtype=np.float32)
            )
        return torch.ones(len(self), dtype=torch.float32)

    def gather(self, index: torch.Tensor | np.ndarray | int) -> dict[str, torch.Tensor]:
        """Copy ``index`` rows from the memmap into a packed batch dict."""
        logical = _as_numpy_index(index)
        k = int(logical.size)
        if k == 0:
            raise ValueError("gather needs at least one index")
        n = len(self)
        if int(logical.min()) < 0 or int(logical.max()) >= n:
            raise IndexError(f"row index out of range for n={n}")
        phys = self._physical_rows(logical)
        order = np.argsort(phys, kind="mergesort")
        sorted_phys = phys[order]
        inv = np.empty(k, dtype=np.int64)
        inv[order] = np.arange(k, dtype=np.int64)

        def _rows(name: str, dtype: np.dtype | None = None) -> np.ndarray:
            arr = np.ascontiguousarray(self._np[name][sorted_phys], dtype=dtype)
            return arr[inv]

        white_n = torch.from_numpy(np.ascontiguousarray(_rows("white_n", np.uint8))).long()
        black_n = torch.from_numpy(np.ascontiguousarray(_rows("black_n", np.uint8))).long()
        ar = torch.arange(self._width, dtype=torch.long)
        stm = np.asarray(_rows("stm")).astype(np.bool_, copy=False)
        if "visits" in self._np:
            weight = np.ascontiguousarray(_rows("visits"), dtype=np.float32)
        else:
            weight = np.ones(k, dtype=np.float32)
        return {
            "white_idx": torch.from_numpy(np.ascontiguousarray(_rows("white_idx", np.int16))).long(),
            "white_mask": ar < white_n.unsqueeze(1),
            "black_idx": torch.from_numpy(np.ascontiguousarray(_rows("black_idx", np.int16))).long(),
            "black_mask": ar < black_n.unsqueeze(1),
            "stm_white": torch.from_numpy(np.ascontiguousarray(stm)),
            "target": torch.from_numpy(_normalize_wdl_np(_rows("wdl"))),
            "weight": torch.from_numpy(weight),
        }

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
            blocks.append({name: np.ascontiguousarray(arrays[name]) for name in arrays})
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
        """One memmapped dataset per slice folder (no concat, no tensor pack)."""
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
        return int(self._n)

    def take_rows(self, indices: np.ndarray) -> FenValueVisitsDataset:
        """New dataset with the given row indices (order preserved, arrays shared)."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        n = len(self)
        if idx.size == 0:
            raise ValueError("take_rows needs at least one index")
        if int(idx.min()) < 0 or int(idx.max()) >= n:
            raise ValueError(f"row index out of range for n={n}")
        clone = FenValueVisitsDataset.__new__(FenValueVisitsDataset)
        clone.max_active = self.max_active
        clone.feature_dim = self.feature_dim
        clone.cache_dir = self.cache_dir
        clone._attach_arrays(self._np, row_index=self._physical_rows(idx))
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
        i = int(index)
        n = len(self)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(index)
        phys = int(self._physical_rows(np.asarray([i], dtype=np.int64))[0])
        width = self._width
        white_n = int(self._np["white_n"][phys])
        black_n = int(self._np["black_n"][phys])
        white_mask = torch.zeros(width, dtype=torch.bool)
        black_mask = torch.zeros(width, dtype=torch.bool)
        if white_n:
            white_mask[:white_n] = True
        if black_n:
            black_mask[:black_n] = True
        wdl = _normalize_wdl_np(np.asarray(self._np["wdl"][phys : phys + 1]))[0]
        if "visits" in self._np:
            weight = np.float32(self._np["visits"][phys])
        else:
            weight = np.float32(1.0)
        stm = bool(self._np["stm"][phys])
        return {
            "white_idx": torch.from_numpy(
                np.ascontiguousarray(self._np["white_idx"][phys], dtype=np.int16)
            ),
            "white_mask": white_mask,
            "black_idx": torch.from_numpy(
                np.ascontiguousarray(self._np["black_idx"][phys], dtype=np.int16)
            ),
            "black_mask": black_mask,
            "stm_white": torch.tensor(stm, dtype=torch.bool),
            "target": torch.from_numpy(wdl),
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


_PACK_KEYS = (
    "white_idx",
    "white_mask",
    "black_idx",
    "black_mask",
    "stm_white",
    "target",
    "weight",
)


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

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

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

    def gather(self, index: torch.Tensor | np.ndarray | int) -> dict[str, torch.Tensor]:
        """Batch gather from memmapped slices; copies only the requested rows."""
        idx = _as_numpy_index(index)
        k = int(idx.size)
        if k == 0:
            raise ValueError("gather needs at least one index")
        n = len(self)
        if int(idx.min()) < 0 or int(idx.max()) >= n:
            raise IndexError(f"row index out of range for n={n}")
        offsets = np.asarray(self._offsets, dtype=np.int64)
        slice_ids = np.searchsorted(offsets[1:], idx, side="right")
        local = idx - offsets[slice_ids]
        order = np.argsort(slice_ids, kind="mergesort")
        sorted_sids = slice_ids[order]
        sorted_local = local[order]
        inv = np.empty(k, dtype=np.int64)
        inv[order] = np.arange(k, dtype=np.int64)
        inv_t = torch.from_numpy(inv)

        pieces: dict[str, list[torch.Tensor]] = {key: [] for key in _PACK_KEYS}
        start = 0
        while start < k:
            sid = int(sorted_sids[start])
            end = start + 1
            while end < k and int(sorted_sids[end]) == sid:
                end += 1
            part = self.slices[sid].gather(sorted_local[start:end])
            for key in _PACK_KEYS:
                pieces[key].append(part[key])
            start = end
        return {key: torch.cat(parts, dim=0)[inv_t] for key, parts in pieces.items()}

    def iter_batches(self, batch_size: int, *, pad: bool = False):
        n = len(self)
        width = int(batch_size)
        if width < 1:
            raise ValueError("batch_size must be >= 1")
        for start in range(0, n, width):
            end = min(start + width, n)
            batch = self.gather(np.arange(start, end, dtype=np.int64))
            got = end - start
            if pad and got < width:
                extra = width - got
                padded: dict[str, torch.Tensor] = {}
                for key, tensor in batch.items():
                    zeros = torch.zeros(
                        (extra, *tensor.shape[1:]),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    padded[key] = torch.cat([tensor, zeros], dim=0)
                batch = padded
            yield batch


class PackedSparseTensors:
    """Column-major sparse NNUE table for batched integer gathers.

    ``ConcatSliceDataset.__getitem__`` plus ``collate_sparse`` builds each
    batch as thousands of Python dicts. This concatenates the underlying
    tensors once so a batch is one gather per column — the path that can
    stay on GPU for the whole run.
    """

    __slots__ = ("_t",)

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        missing = [key for key in _PACK_KEYS if key not in tensors]
        if missing:
            raise ValueError(f"packed tensors missing {missing}")
        n = int(tensors["target"].shape[0])
        if n < 1:
            raise ValueError("packed tensors need at least one row")
        for key in _PACK_KEYS:
            if int(tensors[key].shape[0]) != n:
                raise ValueError(f"{key} rows {tensors[key].shape[0]} != {n}")
        normalized = {
            "white_idx": tensors["white_idx"].long(),
            "white_mask": tensors["white_mask"].bool(),
            "black_idx": tensors["black_idx"].long(),
            "black_mask": tensors["black_mask"].bool(),
            "stm_white": tensors["stm_white"].bool(),
            "target": tensors["target"].float(),
            "weight": tensors["weight"].float(),
        }
        self._t = normalized

    def __len__(self) -> int:
        return int(self._t["target"].shape[0])

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._t[key]

    @property
    def device(self) -> torch.device:
        return self._t["target"].device

    def pin_memory(self) -> PackedSparseTensors:
        if self.device.type != "cpu":
            return self
        return PackedSparseTensors({key: tensor.pin_memory() for key, tensor in self._t.items()})

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> PackedSparseTensors:
        dev = torch.device(device)
        moved: dict[str, torch.Tensor] = {}
        for key, tensor in self._t.items():
            out = tensor.to(device=dev, non_blocking=non_blocking)
            if key in {"white_idx", "black_idx"} and out.dtype != torch.int64:
                out = out.long()
            elif key in {"white_mask", "black_mask", "stm_white"} and out.dtype != torch.bool:
                out = out.bool()
            moved[key] = out
        return PackedSparseTensors(moved)

    def index_select(self, index: torch.Tensor) -> PackedSparseTensors:
        index = index.to(device=self.device, dtype=torch.long)
        return PackedSparseTensors(
            {key: tensor.index_select(0, index) for key, tensor in self._t.items()}
        )

    def gather(self, index: torch.Tensor) -> dict[str, torch.Tensor]:
        index = index.to(device=self.device, dtype=torch.long)
        return {key: tensor[index] for key, tensor in self._t.items()}

    def slice_rows(self, start: int, end: int) -> dict[str, torch.Tensor]:
        sl = slice(int(start), int(end))
        return {key: tensor[sl] for key, tensor in self._t.items()}

    def iter_batches(self, batch_size: int, *, pad: bool = False):
        n = len(self)
        width = int(batch_size)
        if width < 1:
            raise ValueError("batch_size must be >= 1")
        for start in range(0, n, width):
            end = min(start + width, n)
            batch = self.slice_rows(start, end)
            got = end - start
            if pad and got < width:
                extra = width - got
                padded: dict[str, torch.Tensor] = {}
                for key, tensor in batch.items():
                    zeros = torch.zeros(
                        (extra, *tensor.shape[1:]),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    padded[key] = torch.cat([tensor, zeros], dim=0)
                batch = padded
            yield batch

    @classmethod
    def from_dataset(cls, dataset: object) -> PackedSparseTensors:
        if isinstance(dataset, PackedSparseTensors):
            return dataset
        if isinstance(dataset, Subset):
            idx = np.asarray(dataset.indices, dtype=np.int64)
            base = dataset.dataset
            if hasattr(base, "gather"):
                return cls(base.gather(idx))
            return cls.from_dataset(base).index_select(torch.as_tensor(idx, dtype=torch.long))
        if isinstance(dataset, (ConcatSliceDataset, FenValueVisitsDataset)):
            n = len(dataset)
            if n < 1:
                raise ValueError("packed tensors need at least one row")
            return cls(dataset.gather(np.arange(n, dtype=np.int64)))
        raise TypeError(f"cannot pack sparse tensors from {type(dataset)!r}")


def _slice_n_rows(folder: Path) -> int:
    meta_path = folder / SLICE_FEATURES_META
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            n = int(meta.get("n_rows", -1))
            if n >= 0:
                return n
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    path = slice_features_path(folder)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; encode with scripts/encode_slice_features.py")
    return int(_mmap_npz(path)["wdl"].shape[0])


def _open_slice_npz(folder: Path) -> dict[str, np.ndarray]:
    """Memory-map ``features.npz`` (dict of arrays; no zip extraction of full tables)."""
    return load_slice_feature_db(folder)


def _normalize_wdl_np(wdl: np.ndarray) -> np.ndarray:
    out = np.clip(np.ascontiguousarray(wdl, dtype=np.float32), 0.0, None)
    denom = np.maximum(out.sum(axis=1, keepdims=True), 1e-8)
    return out / denom


def _alloc_compact(
    n: int,
    width: int,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    n = int(n)
    width = int(width)
    dev = torch.device(device or "cpu")
    return {
        "white_idx": torch.empty((n, width), dtype=torch.int16, device=dev),
        "white_n": torch.empty((n,), dtype=torch.uint8, device=dev),
        "black_idx": torch.empty((n, width), dtype=torch.int16, device=dev),
        "black_n": torch.empty((n,), dtype=torch.uint8, device=dev),
        "stm": torch.empty((n,), dtype=torch.bool, device=dev),
        "wdl": torch.empty((n, 3), dtype=torch.float32, device=dev),
        "weight": torch.empty((n,), dtype=torch.float32, device=dev),
    }


def _copy_slice_rows(
    dst: dict[str, torch.Tensor],
    start: int,
    npz: dict[str, np.ndarray],
    rows: np.ndarray,
) -> int:
    k = int(rows.size)
    if k == 0:
        return 0
    end = start + k
    dst["white_idx"][start:end].copy_(
        torch.from_numpy(np.ascontiguousarray(npz["white_idx"][rows], dtype=np.int16))
    )
    dst["black_idx"][start:end].copy_(
        torch.from_numpy(np.ascontiguousarray(npz["black_idx"][rows], dtype=np.int16))
    )
    dst["white_n"][start:end].copy_(
        torch.from_numpy(np.ascontiguousarray(npz["white_n"][rows], dtype=np.uint8))
    )
    dst["black_n"][start:end].copy_(
        torch.from_numpy(np.ascontiguousarray(npz["black_n"][rows], dtype=np.uint8))
    )
    dst["stm"][start:end].copy_(
        torch.from_numpy(np.asarray(npz["stm"][rows]).astype(np.bool_, copy=False))
    )
    dst["wdl"][start:end].copy_(torch.from_numpy(_normalize_wdl_np(npz["wdl"][rows])))
    if "visits" in npz:
        vis = np.ascontiguousarray(npz["visits"][rows], dtype=np.float32)
    else:
        vis = np.ones(k, dtype=np.float32)
    dst["weight"][start:end].copy_(torch.from_numpy(vis))
    return k


class CompactPackedTensors:
    """Sparse tables in on-disk dtypes; promote only the gathered minibatch.

    Stores int16 indices and uint8 counts (no ``(N, 128)`` masks, no int64
    index tables). Lives on CPU or CUDA. ``gather`` / ``iter_batches`` return
    the same dict as ``PackedSparseTensors`` so the train loop is unchanged.

    On CUDA this is the path that keeps the GPU busy: ``torch.randint`` and
    gather run on-device, and only the batch is promoted to int64 / bool masks.
    61M rows is ~32 GiB (vs ~130 GiB for int64 + masks).
    """

    __slots__ = ("_t", "_width", "_arange")

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        n = int(tensors["wdl"].shape[0])
        if n < 1:
            raise ValueError("compact pack needs at least one row")
        width = int(tensors["white_idx"].shape[1])
        self._width = width
        device = tensors["wdl"].device

        def _keep(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            if tensor.dtype == dtype and tensor.device == device and tensor.is_contiguous():
                return tensor
            return tensor.to(device=device, dtype=dtype).contiguous()

        self._t = {
            "white_idx": _keep(tensors["white_idx"], torch.int16),
            "white_n": _keep(tensors["white_n"], torch.uint8),
            "black_idx": _keep(tensors["black_idx"], torch.int16),
            "black_n": _keep(tensors["black_n"], torch.uint8),
            "stm": _keep(tensors["stm"], torch.bool),
            "wdl": _keep(tensors["wdl"], torch.float32),
            "weight": _keep(tensors["weight"], torch.float32),
        }
        for key, tensor in self._t.items():
            if int(tensor.shape[0]) != n:
                raise ValueError(f"{key} rows {tensor.shape[0]} != {n}")
            if tensor.device != device:
                raise ValueError(f"{key} device {tensor.device} != {device}")
        self._arange = torch.arange(width, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return int(self._t["wdl"].shape[0])

    @property
    def device(self) -> torch.device:
        return self._t["wdl"].device

    @property
    def width(self) -> int:
        return int(self._width)

    @property
    def nbytes(self) -> int:
        return int(sum(t.nbytes for t in self._t.values()))

    def gather(self, index: torch.Tensor) -> dict[str, torch.Tensor]:
        index = index.to(device=self.device, dtype=torch.long)
        white_n = self._t["white_n"][index].long()
        black_n = self._t["black_n"][index].long()
        ar = self._arange
        return {
            "white_idx": self._t["white_idx"][index].long(),
            "white_mask": ar < white_n.unsqueeze(1),
            "black_idx": self._t["black_idx"][index].long(),
            "black_mask": ar < black_n.unsqueeze(1),
            "stm_white": self._t["stm"][index],
            "target": self._t["wdl"][index],
            "weight": self._t["weight"][index],
        }

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> CompactPackedTensors:
        dev = torch.device(device)
        if self.device == dev:
            return self
        return CompactPackedTensors(
            {key: tensor.to(device=dev, non_blocking=non_blocking) for key, tensor in self._t.items()}
        )

    def slice_rows(self, start: int, end: int) -> dict[str, torch.Tensor]:
        return self.gather(
            torch.arange(int(start), int(end), dtype=torch.long, device=self.device)
        )

    def iter_batches(self, batch_size: int, *, pad: bool = False):
        n = len(self)
        width = int(batch_size)
        if width < 1:
            raise ValueError("batch_size must be >= 1")
        for start in range(0, n, width):
            end = min(start + width, n)
            batch = self.slice_rows(start, end)
            got = end - start
            if pad and got < width:
                extra = width - got
                padded: dict[str, torch.Tensor] = {}
                for key, tensor in batch.items():
                    zeros = torch.zeros(
                        (extra, *tensor.shape[1:]),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    padded[key] = torch.cat([tensor, zeros], dim=0)
                batch = padded
            yield batch

    def index_select(self, index: torch.Tensor) -> CompactPackedTensors:
        index = index.to(device=self.device, dtype=torch.long)
        return CompactPackedTensors(
            {
                "white_idx": self._t["white_idx"][index],
                "white_n": self._t["white_n"][index],
                "black_idx": self._t["black_idx"][index],
                "black_n": self._t["black_n"][index],
                "stm": self._t["stm"][index],
                "wdl": self._t["wdl"][index],
                "weight": self._t["weight"][index],
            }
        )


def load_compact_split(
    folders: list[Path],
    test_fraction: float,
    *,
    seed: int = 0,
    rebuild: bool = False,
    progress: bool = True,
    max_active: int = MAX_ACTIVE_FEATURES,
    device: torch.device | str | None = None,
) -> tuple[CompactPackedTensors, CompactPackedTensors]:
    """Load ``features.npz`` sequentially into compact train/test tables.

    Split matches ``split_random_fraction`` (per-slice i.i.d., same seed).
    Peak extra RAM is one memmapped slice (row copies only) plus the compact
    tables on ``device`` (CPU or CUDA). CUDA fill is slice-by-slice so the
    host never holds a second full copy.
    """
    frac = float(test_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"test fraction must be in (0, 1), got {test_fraction}")
    if not folders:
        raise FileNotFoundError("no slice folders to load")
    dest = torch.device(device or "cpu")

    if rebuild:
        for folder in folders:
            ensure_slice_feature_db(
                folder, max_active=max_active, rebuild=True, progress=progress
            )

    rng = np.random.RandomState(int(seed))
    counts: list[tuple[int, int, int]] = []
    n_train = 0
    n_test = 0
    for folder in folders:
        n = _slice_n_rows(folder)
        n_te = int(n * frac)
        n_tr = n - n_te
        counts.append((n, n_tr, n_te))
        n_train += n_tr
        n_test += n_te
    if n_train < 1:
        raise ValueError("train split is empty; lower --test-fraction")
    if n_test < 1:
        raise ValueError("test split is empty; raise --test-fraction or use larger slices")

    width = MAX_ACTIVE_FEATURES
    first = _open_slice_npz(folders[0])
    width = int(first["white_idx"].shape[1])
    del first

    train_buf = _alloc_compact(n_train, width, device=dest)
    test_buf = _alloc_compact(n_test, width, device=dest)
    i_tr = 0
    i_te = 0
    folder_iter: Any = folders
    if progress:
        try:
            from tqdm import tqdm

            folder_iter = tqdm(folders, desc=f"load slices → {dest}", unit="slice")
        except ImportError:
            folder_iter = folders

    for folder, (n, _n_tr, n_te) in zip(folder_iter, counts):
        perm = rng.permutation(n)
        test_rows = perm[:n_te]
        train_rows = perm[n_te:]
        npz = _open_slice_npz(folder)
        if int(npz["white_idx"].shape[1]) != width:
            raise ValueError(
                f"{folder} white_idx width {npz['white_idx'].shape[1]} != {width}"
            )
        i_tr += _copy_slice_rows(train_buf, i_tr, npz, train_rows)
        i_te += _copy_slice_rows(test_buf, i_te, npz, test_rows)
        del npz

    if i_tr != n_train or i_te != n_test:
        raise RuntimeError(f"compact fill mismatch train {i_tr}/{n_train} test {i_te}/{n_test}")
    return CompactPackedTensors(train_buf), CompactPackedTensors(test_buf)


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
