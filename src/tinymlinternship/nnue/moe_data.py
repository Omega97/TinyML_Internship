"""Train/test row plans for MoE scripts (matches ``split_random_fraction``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tinymlinternship.nnue.dataset import (
    CompactPackedTensors,
    FenValueVisitsDataset,
    MAX_ACTIVE_FEATURES,
    _alloc_compact,
    _copy_slice_rows,
    _slice_n_rows,
    discover_slice_folders,
    load_slice_feature_db,
    slice_source_json,
)


def slice_folders(root: Path) -> list[Path]:
    folders = discover_slice_folders(root)
    if folders:
        return folders
    if slice_source_json(root) is not None:
        return [root]
    return []


def plan_split_indices(
    folders: list[Path],
    test_fraction: float,
    *,
    seed: int = 0,
) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    """Per folder ``(train_idx, test_idx)`` with the trainer's RNG sequence."""
    frac = float(test_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    rng = np.random.RandomState(int(seed))
    planned: list[tuple[Path, np.ndarray, np.ndarray]] = []
    for folder in folders:
        n = int(_slice_n_rows(folder))
        n_test = int(n * frac)
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        planned.append((folder, train_idx.astype(np.int64, copy=False), test_idx.astype(np.int64, copy=False)))
    return planned


def subsample_parts(
    parts: list[np.ndarray],
    max_rows: int,
    *,
    seed: int = 0,
) -> list[np.ndarray]:
    """Keep a global random subset of concatenated row indices (sorted by slice)."""
    arrays = [np.asarray(p, dtype=np.int64) for p in parts]
    sizes = [int(arr.size) for arr in arrays]
    total = int(sum(sizes))
    if total < 1:
        raise ValueError("no rows to subsample")
    take = total if int(max_rows) <= 0 else min(int(max_rows), total)
    if take == total:
        return arrays
    rng = np.random.RandomState(int(seed))
    chosen = np.sort(rng.choice(total, size=take, replace=False).astype(np.int64, copy=False))
    out: list[np.ndarray] = []
    offset = 0
    cursor = 0
    n_chosen = int(chosen.size)
    for arr, size in zip(arrays, sizes):
        end = offset + size
        start_c = cursor
        while cursor < n_chosen and int(chosen[cursor]) < end:
            cursor += 1
        local = chosen[start_c:cursor] - offset
        out.append(arr[local])
        offset = end
    return out


def count_rows(parts: list[np.ndarray]) -> int:
    return int(sum(int(np.asarray(p).size) for p in parts))


def iter_folder_batches(
    folder: Path,
    local_rows: np.ndarray,
    *,
    batch_size: int,
    dataset: FenValueVisitsDataset | None = None,
) -> tuple[FenValueVisitsDataset, object]:
    """Yield packed batch dicts for ``local_rows`` of one slice."""
    ds = dataset if dataset is not None else FenValueVisitsDataset(folder, progress=False)
    idx = np.asarray(local_rows, dtype=np.int64).reshape(-1)
    width = int(batch_size)
    if width < 1:
        raise ValueError("batch_size must be >= 1")

    def _gen():
        for start in range(0, int(idx.size), width):
            yield ds.gather(idx[start : start + width])

    return ds, _gen()


def batches_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}


def pack_parts(
    folders: list[Path],
    parts: list[np.ndarray],
    *,
    max_active: int = MAX_ACTIVE_FEATURES,
    log=None,
) -> tuple[CompactPackedTensors, np.ndarray, np.ndarray]:
    """Copy selected local rows into one compact CPU table (one mmap open per slice)."""
    if len(folders) != len(parts):
        raise ValueError("folders and parts length mismatch")
    n = count_rows(parts)
    if n < 1:
        raise ValueError("no rows to pack")
    width = int(max_active)
    for folder, rows in zip(folders, parts):
        if int(np.asarray(rows).size) == 0:
            continue
        npz = load_slice_feature_db(folder)
        width = int(npz["white_idx"].shape[1])
        del npz
        break
    buf = _alloc_compact(n, width, device="cpu")
    slice_ids = np.empty(n, dtype=np.uint16)
    local_rows = np.empty(n, dtype=np.uint32)
    cursor = 0
    nonempty = 0
    for sid, (folder, rows) in enumerate(zip(folders, parts)):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows.size == 0:
            continue
        nonempty += 1
        npz = load_slice_feature_db(folder)
        if int(npz["white_idx"].shape[1]) != width:
            raise ValueError(
                f"{folder} width {npz['white_idx'].shape[1]} != packed width {width}"
            )
        _copy_slice_rows(buf, cursor, npz, rows)
        end = cursor + int(rows.size)
        slice_ids[cursor:end] = np.uint16(sid)
        local_rows[cursor:end] = rows.astype(np.uint32, copy=False)
        cursor = end
        del npz
        if log and nonempty % 25 == 0:
            log(f"  packed {cursor:,}/{n:,} rows from {nonempty} slices")
    if cursor != n:
        raise RuntimeError(f"pack fill mismatch {cursor}/{n}")
    return CompactPackedTensors(buf), slice_ids, local_rows


def save_train_pack(
    work_dir: Path,
    pack: CompactPackedTensors,
    slice_ids: np.ndarray,
    local_rows: np.ndarray,
) -> None:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save({key: tensor.cpu() for key, tensor in pack._t.items()}, work_dir / "train_pack.pt")
    np.save(work_dir / "slice_ids.npy", np.asarray(slice_ids, dtype=np.uint16))
    np.save(work_dir / "local_rows.npy", np.asarray(local_rows, dtype=np.uint32))


def load_train_pack(work_dir: Path) -> tuple[CompactPackedTensors, np.ndarray, np.ndarray]:
    work_dir = Path(work_dir)
    tensors = torch.load(work_dir / "train_pack.pt", map_location="cpu", weights_only=False)
    pack = CompactPackedTensors(tensors)
    slice_ids = np.load(work_dir / "slice_ids.npy")
    local_rows = np.load(work_dir / "local_rows.npy")
    if int(slice_ids.shape[0]) != len(pack):
        raise ValueError("train_pack and slice_ids length mismatch")
    return pack, slice_ids, local_rows
