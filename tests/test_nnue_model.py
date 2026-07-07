"""Tests for bucketed NNUE model and dataset."""

from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch

from tinymlinternship.config.settings import CHESSBENCH_PROCESSED_DIR
from tinymlinternship.data.chessbench_preprocess import parse_chessbench_row
from tinymlinternship.features import FEATURE_DIM, NUM_BUCKETS
from tinymlinternship.nnue import BucketedNNUE, ChessbenchDataset, indices_to_binary


def test_indices_to_binary():
    x = indices_to_binary([0, 5, 10])
    assert x.shape == (FEATURE_DIM,)
    assert x[0] == 1.0
    assert x[5] == 1.0
    assert x[10] == 1.0
    assert x[1] == 0.0


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