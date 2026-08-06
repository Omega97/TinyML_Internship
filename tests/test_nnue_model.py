"""Tests for bucketed NNUE model and dataset."""

from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch

from tinymlinternship.config.settings import CHESSBENCH_PROCESSED_DIR
from tinymlinternship.data.chessbench_preprocess import parse_chessbench_row
from tinymlinternship.features import FEATURE_DIM, NUM_BUCKETS
from tinymlinternship.nnue import (
    BucketedNNUE,
    ChessbenchDataset,
    SingleHeadNNUE,
    build_nnue,
    indices_to_binary,
    infer_architecture,
)


def test_indices_to_binary():
    x = indices_to_binary([0, 5, 10])
    assert x.shape == (FEATURE_DIM,)
    assert x[0] == 1.0
    assert x[5] == 1.0
    assert x[10] == 1.0
    assert x[1] == 0.0


def test_single_head_nnue_forward_startpos():
    """F3 production path: one head, ignores bucket routing."""
    row = parse_chessbench_row(chess.STARTING_FEN, 0.5)
    assert row is not None

    model = SingleHeadNNUE(hidden_dim=16)
    white = indices_to_binary(row.white_features).unsqueeze(0)
    black = indices_to_binary(row.black_features).unsqueeze(0)
    bucket_ids = torch.tensor([row.bucket_id], dtype=torch.long)
    stm_white = torch.tensor([row.stm_white], dtype=torch.bool)

    out = model(white, black, bucket_ids, stm_white)
    assert out.shape == (1,)
    assert -1.0 <= out.item() <= 1.0


def test_single_head_ignores_bucket_id():
    model = SingleHeadNNUE(hidden_dim=8)
    white = torch.zeros(1, FEATURE_DIM)
    black = torch.zeros(1, FEATURE_DIM)
    white[:, 0] = 1.0
    black[:, 1] = 1.0
    stm = torch.tensor([True], dtype=torch.bool)
    a = model(white, black, torch.tensor([0]), stm)
    b = model(white, black, torch.tensor([7]), stm)
    assert torch.allclose(a, b)


def test_bucketed_nnue_forward_startpos():
    row = parse_chessbench_row(chess.STARTING_FEN, 0.5)
    assert row is not None

    model = BucketedNNUE(hidden_dim=16)
    white = indices_to_binary(row.white_features).unsqueeze(0)
    black = indices_to_binary(row.black_features).unsqueeze(0)
    bucket_ids = torch.tensor([row.bucket_id], dtype=torch.long)
    stm_white = torch.tensor([row.stm_white], dtype=torch.bool)

    out = model(white, black, bucket_ids, stm_white)
    assert out.shape == (1,)
    assert out.item() == pytest.approx(0.0, abs=0.5)


def test_bucketed_nnue_routes_by_bucket():
    model = BucketedNNUE(hidden_dim=8, num_buckets=NUM_BUCKETS)
    white = torch.zeros(2, FEATURE_DIM)
    black = torch.zeros(2, FEATURE_DIM)
    white[:, 0] = 1.0
    black[:, 1] = 1.0
    bucket_ids = torch.tensor([0, 3], dtype=torch.long)
    stm_white = torch.tensor([True, False], dtype=torch.bool)

    out = model(white, black, bucket_ids, stm_white)
    assert out.shape == (2,)


def test_build_nnue_and_infer_architecture():
    single = build_nnue("single_head", hidden_dim=16)
    bucketed = build_nnue("bucketed", hidden_dim=16)
    assert isinstance(single, SingleHeadNNUE)
    assert isinstance(bucketed, BucketedNNUE)
    assert infer_architecture(single.state_dict()) == "single_head"
    assert infer_architecture(bucketed.state_dict()) == "bucketed"
    # single-head is strictly smaller than 8-expert bucketed at same W
    assert single.count_parameters() < bucketed.count_parameters()


@pytest.mark.skipif(
    not (CHESSBENCH_PROCESSED_DIR / "splits" / "train.parquet").exists(),
    reason="ChessBench train split not prepared",
)
def test_chessbench_dataset_loads_row():
    ds = ChessbenchDataset(CHESSBENCH_PROCESSED_DIR / "splits" / "train.parquet")
    item = ds[0]
    assert item["white_features"].shape == (FEATURE_DIM,)
    assert 0 <= item["bucket_id"] < NUM_BUCKETS
    assert -1.0 <= item["target"] <= 1.0


def test_dataset_encodes_from_fen(tmp_path: Path):
    """Production labeled rows (fen only) encode features on the fly."""
    import pandas as pd

    path = tmp_path / "labeled.parquet"
    pd.DataFrame(
        [
            {
                "fen": chess.STARTING_FEN,
                "bucket_id": 7,
                "stm_white": True,
                "expected_reward": 0.0,
            }
        ]
    ).to_parquet(path, index=False)

    ds = ChessbenchDataset(path)
    item = ds[0]
    assert item["white_features"].shape == (FEATURE_DIM,)
    assert item["black_features"].shape == (FEATURE_DIM,)
    assert item["white_features"].sum().item() > 0
    assert item["bucket_id"] == 7
    assert item["target"] == 0.0