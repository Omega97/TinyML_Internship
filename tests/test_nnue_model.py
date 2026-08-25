"""Dual-hidden NNUE forward and sparse L1 equivalence."""

from __future__ import annotations

import chess
import pytest
import torch

import numpy as np

from tinymlinternship.features import FEATURE_DIM, encode_dual
from tinymlinternship.nnue.dataset import FenValueVisitsDataset, _pad_indices
from tinymlinternship.nnue.model import DualHiddenNNUE, LinearWDLNNUE, MediumWDLNNUE


def _dense_from_indices(indices: list[int]) -> torch.Tensor:
    x = torch.zeros(FEATURE_DIM)
    if indices:
        x[torch.tensor(indices, dtype=torch.long)] = 1.0
    return x


def test_dual_hidden_output_range_startpos():
    model = DualHiddenNNUE(hidden_dim=64, hidden2_dim=128)
    white_idx, black_idx = encode_dual(chess.Board())
    white = _dense_from_indices(white_idx).unsqueeze(0)
    black = _dense_from_indices(black_idx).unsqueeze(0)
    stm = torch.tensor([True])
    logits = model(white, black, stm)
    assert logits.shape == (1, 3)
    probs = model.probabilities(logits)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert (probs >= 0).all()
    v = model.stm_value(logits)
    assert -1.0 <= v.item() <= 1.0


def test_linear_wdl_sparse_matches_dense_and_softmax():
    torch.manual_seed(0)
    model = LinearWDLNNUE()
    assert model.count_parameters() == 844 * 2 * 3 + 3
    white_idx, black_idx = encode_dual(chess.Board())
    w_pad, w_n = _pad_indices(white_idx, 128)
    b_pad, b_n = _pad_indices(black_idx, 128)
    w_mask = np.arange(128) < w_n
    b_mask = np.arange(128) < b_n
    stm = torch.tensor([True])
    dense = model(
        _dense_from_indices(white_idx).unsqueeze(0),
        _dense_from_indices(black_idx).unsqueeze(0),
        stm,
    )
    sparse = model.forward_sparse(
        torch.from_numpy(w_pad).unsqueeze(0),
        torch.from_numpy(w_mask).unsqueeze(0),
        torch.from_numpy(b_pad).unsqueeze(0),
        torch.from_numpy(b_mask).unsqueeze(0),
        stm,
    )
    assert dense.shape == (1, 3)
    assert torch.allclose(dense, sparse, atol=1e-5)
    probs = model.probabilities(dense)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_medium_wdl_sparse_matches_dense_and_softmax():
    torch.manual_seed(0)
    model = MediumWDLNNUE(hidden_dim=20)
    assert model.count_parameters() == 844 * 2 * 20 + 20 + 20 * 3 + 3
    white_idx, black_idx = encode_dual(chess.Board())
    w_pad, w_n = _pad_indices(white_idx, 128)
    b_pad, b_n = _pad_indices(black_idx, 128)
    w_mask = np.arange(128) < w_n
    b_mask = np.arange(128) < b_n
    stm = torch.tensor([True])
    dense = model(
        _dense_from_indices(white_idx).unsqueeze(0),
        _dense_from_indices(black_idx).unsqueeze(0),
        stm,
    )
    sparse = model.forward_sparse(
        torch.from_numpy(w_pad).unsqueeze(0),
        torch.from_numpy(w_mask).unsqueeze(0),
        torch.from_numpy(b_pad).unsqueeze(0),
        torch.from_numpy(b_mask).unsqueeze(0),
        stm,
    )
    assert dense.shape == (1, 3)
    assert torch.allclose(dense, sparse, atol=1e-5)
    probs = model.probabilities(dense)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)
    v = model.stm_value(dense)
    assert -1.0 <= v.item() <= 1.0


def test_soft_ce_loss_on_wdl_batch():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "scripts" / "train_nnue.py"
    spec = importlib.util.spec_from_file_location("train_nnue", path)
    assert spec is not None and spec.loader is not None
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    torch.manual_seed(0)
    model = DualHiddenNNUE(hidden_dim=8, hidden2_dim=16)
    white_idx, black_idx = encode_dual(chess.Board())
    w_pad, w_n = _pad_indices(white_idx, 128)
    b_pad, b_n = _pad_indices(black_idx, 128)
    w_mask = np.arange(128) < w_n
    b_mask = np.arange(128) < b_n
    stm = torch.tensor([True, True])
    logits = model.forward_sparse(
        torch.from_numpy(np.stack([w_pad, w_pad])),
        torch.from_numpy(np.stack([w_mask, w_mask])),
        torch.from_numpy(np.stack([b_pad, b_pad])),
        torch.from_numpy(np.stack([b_mask, b_mask])),
        stm,
    )
    assert logits.shape == (2, 3)
    target = torch.tensor([[0.312, 0.469, 0.219], [0.8, 0.1, 0.1]])
    weight = torch.ones(2)
    loss = train.ce_loss(logits, target, weight)
    assert loss.ndim == 0
    assert loss.detach().item() > 0
    probs = model.probabilities(logits)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_sparse_l1_matches_dense():
    torch.manual_seed(0)
    model = DualHiddenNNUE(hidden_dim=64, hidden2_dim=128)
    white_idx, black_idx = encode_dual(chess.Board())
    w_pad, w_n = _pad_indices(white_idx, 128)
    b_pad, b_n = _pad_indices(black_idx, 128)
    w_mask = np.arange(128) < w_n
    b_mask = np.arange(128) < b_n
    stm = torch.tensor([True])
    dense = model(
        _dense_from_indices(white_idx).unsqueeze(0),
        _dense_from_indices(black_idx).unsqueeze(0),
        stm,
    )
    sparse = model.forward_sparse(
        torch.from_numpy(w_pad).unsqueeze(0),
        torch.from_numpy(w_mask).unsqueeze(0),
        torch.from_numpy(b_pad).unsqueeze(0),
        torch.from_numpy(b_mask).unsqueeze(0),
        stm,
    )
    assert torch.allclose(dense, sparse, atol=1e-5)
    assert dense.shape == (1, 3)


def test_concat_order_uses_stm():
    """Black to move uses black accumulator first; softmax is a 3-simplex."""
    model = DualHiddenNNUE(hidden_dim=8, hidden2_dim=16)
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    white_idx, black_idx = encode_dual(chess.Board(fen))
    logits = model(
        _dense_from_indices(white_idx).unsqueeze(0),
        _dense_from_indices(black_idx).unsqueeze(0),
        torch.tensor([False]),
    )
    probs = model.probabilities(logits)
    assert logits.shape == (1, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_dataset_excludes_test_epds():
    import pandas as pd

    df = pd.DataFrame(
        {
            "fen": [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            ],
            "value": [0.0, 0.1],
            "visits": [2, 1],
        }
    )
    holdout = {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"}
    ds = FenValueVisitsDataset(df, exclude_epds=holdout)
    assert len(ds) == 1
    assert ds[0]["target"].shape == (3,)
    assert float(ds[0]["target"].sum()) == pytest.approx(1.0, abs=1e-5)


def test_feature_cache_roundtrip(tmp_path):
    import pandas as pd

    from tinymlinternship.nnue.dataset import (
        cache_is_valid,
        ensure_feature_cache,
        load_feature_cache,
    )

    rows = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "value": 0.0,
            "visits": 3,
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "value": 0.1,
            "visits": 1,
        },
    ]
    table = tmp_path / "tiny.parquet"
    pd.DataFrame(rows).to_parquet(table)
    cache_dir, first = ensure_feature_cache(table, progress=False)
    assert cache_is_valid(cache_dir, table, max_active=128)
    assert first["wdl"].shape == (2, 3)
    second = load_feature_cache(cache_dir)
    assert (first["white_idx"] == second["white_idx"]).all()
    assert (first["epd_hash"] == second["epd_hash"]).all()

    holdout = {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"}
    ds = FenValueVisitsDataset(table, exclude_epds=holdout, progress=False)
    assert ds.cache_dir == cache_dir
    assert len(ds) == 1
    assert ds[0]["target"].shape == (3,)
    assert ds[0]["white_idx"].dtype == torch.int16


def test_slice_feature_db_omits_visits(tmp_path):
    import json

    from tinymlinternship.nnue.dataset import (
        SLICE_DB_ARRAYS,
        FenValueVisitsDataset,
        ensure_slice_feature_db,
        load_slice_feature_db,
        slice_db_is_valid,
    )

    folder = tmp_path / "fen_value_visits_tiny"
    folder.mkdir()
    rows = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "value": 0.25,
            "visits": 9,
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "value": -0.5,
            "visits": 2,
        },
    ]
    (folder / "fen_value_visits_tiny.json").write_text(json.dumps(rows), encoding="utf-8")
    npz, arrays = ensure_slice_feature_db(folder, progress=False)
    assert npz.name == "features.npz"
    assert slice_db_is_valid(folder, folder / "fen_value_visits_tiny.json", max_active=128)
    assert set(arrays) == set(SLICE_DB_ARRAYS)
    assert "visits" not in arrays
    loaded = load_slice_feature_db(folder)
    assert loaded["wdl"].shape == (2, 3)
    assert loaded["wdl"].sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)
    ds = FenValueVisitsDataset(folder, progress=False)
    assert len(ds) == 2
    assert ds[0]["weight"].item() == 1.0
    assert ds[0]["target"].shape == (3,)
    assert ds[0]["white_mask"].any()


def test_from_slice_root_skips_test_folder(tmp_path):
    import json

    def _write_slice(name: str, value: float) -> None:
        folder = tmp_path / name
        folder.mkdir()
        rows = [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "value": value,
                "visits": 1,
            }
        ]
        (folder / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")

    _write_slice("fen_value_visits_train_a", 0.1)
    _write_slice("fen_value_visits_lichess_db_standard_rated_2026-07_100000-101000", 0.9)
    ds = FenValueVisitsDataset.from_slice_root(
        tmp_path,
        skip_names={"fen_value_visits_lichess_db_standard_rated_2026-07_100000-101000"},
        progress=False,
    )
    assert len(ds) == 1
    assert ds[0]["target"].shape == (3,)
    assert float(ds[0]["target"][0] - ds[0]["target"][2]) == pytest.approx(0.1, abs=5e-3)


def test_across_slice_batch_samples_every_folder(tmp_path):
    import json

    from tinymlinternship.nnue.dataset import (
        AcrossSliceBatchSampler,
        FenValueVisitsDataset,
        MixedSliceDataset,
    )

    def _write_slice(name: str, value: float) -> None:
        folder = tmp_path / name
        folder.mkdir()
        rows = [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "value": value,
                "visits": 1,
            }
        ]
        (folder / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")

    _write_slice("slice_a", 0.1)
    _write_slice("slice_b", 0.9)
    slices = FenValueVisitsDataset.load_slice_datasets(tmp_path, progress=False)
    mixed = MixedSliceDataset(slices)
    sampler = AcrossSliceBatchSampler(mixed.sizes, batch_size=8, batches=4, seed=0)
    seen_values = set()
    for packed_batch in sampler:
        assert len(packed_batch) == 8
        for packed in packed_batch:
            tgt = mixed[packed]["target"]
            seen_values.add(round(float(tgt[0] - tgt[2]), 1))
    assert seen_values == {0.1, 0.9}


def test_rank_rows_best_and_worst():
    import importlib.util
    from pathlib import Path

    import numpy as np

    path = Path(__file__).parent.parent / "scripts" / "inspect_nnue_positions.py"
    spec = importlib.util.spec_from_file_location("inspect_nnue_positions", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fens = ["a", "b", "c", "d"]
    teacher = np.array(
        [
            [1.00, 0.00, 0.00],
            [0.50, 0.40, 0.10],
            [0.20, 0.60, 0.20],
            [0.90, 0.05, 0.05],
        ]
    )
    pred = np.array(
        [
            [0.98, 0.01, 0.01],
            [0.48, 0.42, 0.10],
            [0.70, 0.20, 0.10],
            [0.05, 0.05, 0.90],
        ]
    )
    best, worst = mod.rank_rows(fens, teacher, pred, n=2)
    assert [row["fen"] for row in best] == ["a", "b"]
    assert [row["fen"] for row in worst] == ["d", "c"]


def test_maybe_subset_dataset_is_fixed_and_smaller():
    import importlib.util
    from pathlib import Path

    from torch.utils.data import Dataset, Subset

    path = Path(__file__).parent.parent / "scripts" / "train_nnue.py"
    spec = importlib.util.spec_from_file_location("train_nnue", path)
    assert spec is not None and spec.loader is not None
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    class _Idx(Dataset):
        def __len__(self) -> int:
            return 100

        def __getitem__(self, i: int) -> int:
            return i

    ds = _Idx()
    a = train.maybe_subset_dataset(ds, 10, seed=0)
    b = train.maybe_subset_dataset(ds, 10, seed=0)
    c = train.maybe_subset_dataset(ds, 10, seed=1)
    assert isinstance(a, Subset)
    assert len(a) == 10
    assert list(a.indices) == list(b.indices)
    assert list(a.indices) != list(c.indices)
    assert train.maybe_subset_dataset(ds, 0) is ds
    assert train.maybe_subset_dataset(ds, 1000) is ds


def test_inspect_sample_subset_is_deterministic():
    import importlib.util
    from pathlib import Path

    import numpy as np

    path = Path(__file__).parent.parent / "scripts" / "inspect_nnue_positions.py"
    spec = importlib.util.spec_from_file_location("inspect_nnue_positions", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fens = [f"f{i}" for i in range(20)]
    teacher = np.eye(3, dtype=np.float64)[np.arange(20) % 3]
    pred = teacher.copy()
    a = mod.sample_subset(fens, teacher, pred, n=5, seed=0)
    b = mod.sample_subset(fens, teacher, pred, n=5, seed=0)
    c = mod.sample_subset(fens, teacher, pred, n=5, seed=1)
    assert a[0] == b[0]
    assert a[0] != c[0]
    assert len(a[0]) == 5

