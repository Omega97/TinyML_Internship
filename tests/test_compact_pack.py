"""Compact CPU pack: same split as FenValueVisitsDataset, gather matches PackedSparseTensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tinymlinternship.nnue.dataset import (
    CompactPackedTensors,
    FenValueVisitsDataset,
    PackedSparseTensors,
    load_compact_split,
    split_random_fraction,
)


def _write_slice(folder: Path, *, n: int, width: int = 8, seed: int = 0) -> None:
    folder.mkdir(parents=True)
    rng = np.random.RandomState(seed)
    white_n = rng.randint(1, width + 1, size=n, dtype=np.uint8)
    black_n = rng.randint(1, width + 1, size=n, dtype=np.uint8)
    white_idx = rng.randint(0, 844, size=(n, width), dtype=np.int16)
    black_idx = rng.randint(0, 844, size=(n, width), dtype=np.int16)
    for i in range(n):
        white_idx[i, white_n[i] :] = 0
        black_idx[i, black_n[i] :] = 0
    wdl = rng.dirichlet((2.0, 2.0, 2.0), size=n).astype(np.float32)
    stm = rng.randint(0, 2, size=n, dtype=np.uint8)
    np.savez(
        folder / "features.npz",
        white_idx=white_idx,
        white_n=white_n,
        black_idx=black_idx,
        black_n=black_n,
        stm=stm,
        wdl=wdl,
    )
    source = folder / f"{folder.name}.json"
    source.write_text("[]\n", encoding="utf-8")
    (folder / "features.meta.json").write_text(
        json.dumps(
            {
                "version": 2,
                "format": "sparse dual-POV 844 + STM WDL",
                "columns": ["white_idx", "white_n", "black_idx", "black_n", "stm", "wdl"],
                "max_active": width,
                "feature_dim": 844,
                "n_rows": n,
                "source": str(source.resolve()),
                "source_size": source.stat().st_size,
                "source_mtime_ns": source.stat().st_mtime_ns,
            }
        ),
        encoding="utf-8",
    )


def test_compact_split_matches_dataset_counts(tmp_path: Path):
    a = tmp_path / "slice_a"
    b = tmp_path / "slice_b"
    _write_slice(a, n=40, seed=1)
    _write_slice(b, n=25, seed=2)
    frac, seed = 0.20, 7
    train_c, test_c = load_compact_split([a, b], frac, seed=seed, progress=False)
    slices = [
        FenValueVisitsDataset(a, progress=False, max_active=8),
        FenValueVisitsDataset(b, progress=False, max_active=8),
    ]
    train_d, test_d = split_random_fraction(slices, frac, seed=seed)
    assert len(train_c) == sum(len(p) for p in train_d)
    assert len(test_c) == sum(len(p) for p in test_d)
    assert len(train_c) + len(test_c) == 65


def test_compact_gather_matches_packed(tmp_path: Path):
    folder = tmp_path / "slice_g"
    _write_slice(folder, n=16, width=8, seed=3)
    ds = FenValueVisitsDataset(folder, progress=False, max_active=8)
    packed = PackedSparseTensors.from_dataset(ds)
    compact = CompactPackedTensors(
        {
            "white_idx": ds.white_idx,
            "white_n": torch.from_numpy(np.array(ds._np["white_n"], copy=True)),
            "black_idx": ds.black_idx,
            "black_n": torch.from_numpy(np.array(ds._np["black_n"], copy=True)),
            "stm": ds.stm_white,
            "wdl": ds.wdl,
            "weight": ds.visits,
        }
    )
    idx = torch.tensor([0, 3, 7, 15])
    a = packed.gather(idx)
    b = compact.gather(idx)
    assert torch.equal(a["white_idx"], b["white_idx"])
    assert torch.equal(a["black_idx"], b["black_idx"])
    assert torch.equal(a["white_mask"], b["white_mask"])
    assert torch.equal(a["black_mask"], b["black_mask"])
    assert torch.equal(a["stm_white"], b["stm_white"])
    assert torch.allclose(a["target"], b["target"])
    assert torch.equal(a["weight"], b["weight"])


def test_prefetch_yields_requested_batches(tmp_path: Path):
    import importlib.util

    folder = tmp_path / "slice_p"
    _write_slice(folder, n=64, width=8, seed=5)
    train, _test = load_compact_split([folder], 0.10, seed=0, progress=False)
    path = Path(__file__).parent.parent / "scripts" / "train_nnue-gpu-compact.py"
    spec = importlib.util.spec_from_file_location("train_nnue_gpu_compact", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = torch.Generator().manual_seed(0)
    batches = list(
        mod.iter_prefetched_train_batches(
            train, batch_size=8, n_batches=5, generator=g, workers=3
        )
    )
    assert len(batches) == 5
    assert batches[0]["white_idx"].shape == (8, 8)
    assert batches[0]["white_idx"].dtype == torch.int64
    assert batches[0]["white_mask"].dtype == torch.bool


def test_train_val_index_select_is_not_a_second_full_pack(tmp_path: Path):
    folder = tmp_path / "slice_v"
    _write_slice(folder, n=100, width=8, seed=4)
    train, test = load_compact_split([folder], 0.10, seed=0, progress=False)
    assert len(train) == 90
    assert len(test) == 10
    subset = train.index_select(torch.arange(5))
    assert len(subset) == 5
    assert subset.nbytes < train.nbytes / 4
    assert len(train) == 90


def test_slice_npz_is_memmapped(tmp_path: Path):
    from tinymlinternship.nnue.dataset import load_slice_feature_db

    folder = tmp_path / "slice_mmap"
    _write_slice(folder, n=32, width=8, seed=6)
    arrays = load_slice_feature_db(folder)
    assert isinstance(arrays["white_idx"], np.memmap)
    assert isinstance(arrays["wdl"], np.memmap)
    ds = FenValueVisitsDataset(folder, progress=False, max_active=8)
    assert isinstance(ds._np["white_idx"], np.memmap)
    part = ds.take_rows(np.array([0, 2, 5, 7]))
    assert part._np["white_idx"] is ds._np["white_idx"]
    assert len(part) == 4
    assert "white_mask" not in vars(ds)


def test_mmap_gather_matches_packed(tmp_path: Path):
    folder = tmp_path / "slice_g2"
    _write_slice(folder, n=24, width=8, seed=8)
    ds = FenValueVisitsDataset(folder, progress=False, max_active=8)
    packed = PackedSparseTensors.from_dataset(ds)
    idx = torch.tensor([0, 5, 11, 23])
    a = packed.gather(idx)
    b = ds.gather(idx)
    for key in a:
        if a[key].dtype.is_floating_point:
            assert torch.allclose(a[key], b[key]), key
        else:
            assert torch.equal(a[key], b[key]), key


def test_compact_to_cpu_roundtrip(tmp_path: Path):
    folder = tmp_path / "slice_dev"
    _write_slice(folder, n=12, width=8, seed=10)
    train, _test = load_compact_split([folder], 0.25, seed=0, progress=False)
    again = train.to("cpu")
    idx = torch.tensor([0, 3, 5])
    a = train.gather(idx)
    b = again.gather(idx)
    assert torch.equal(a["white_idx"], b["white_idx"])
    assert torch.equal(a["white_mask"], b["white_mask"])


def test_compact_cuda_gather_matches_cpu(tmp_path: Path):
    if not torch.cuda.is_available():
        return
    folder = tmp_path / "slice_cuda"
    _write_slice(folder, n=20, width=8, seed=11)
    cpu_train, _cpu_test = load_compact_split(
        [folder], 0.20, seed=1, progress=False, device="cpu"
    )
    gpu_train, _gpu_test = load_compact_split(
        [folder], 0.20, seed=1, progress=False, device="cuda"
    )
    assert gpu_train.device.type == "cuda"
    idx = torch.tensor([0, 2, 7, len(cpu_train) - 1])
    a = cpu_train.gather(idx)
    b = gpu_train.gather(idx.to("cuda"))
    for key in a:
        b_cpu = b[key].cpu()
        if a[key].dtype.is_floating_point:
            assert torch.allclose(a[key], b_cpu), key
        else:
            assert torch.equal(a[key], b_cpu), key


def test_from_dataset_subset_gathers_only_indices(tmp_path: Path):
    from torch.utils.data import Subset

    folder = tmp_path / "slice_sub"
    _write_slice(folder, n=20, width=8, seed=9)
    ds = FenValueVisitsDataset(folder, progress=False, max_active=8)

    class _Spy:
        def __init__(self, inner):
            self.inner = inner
            self.gathered = None

        def __len__(self):
            return len(self.inner)

        def gather(self, index):
            self.gathered = np.asarray(index, dtype=np.int64).reshape(-1)
            return self.inner.gather(index)

    spy = _Spy(ds)
    packed = PackedSparseTensors.from_dataset(Subset(spy, [1, 4, 9]))
    assert packed is not None
    assert list(spy.gathered) == [1, 4, 9]
    assert len(packed) == 3
