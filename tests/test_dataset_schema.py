"""Tests for ASSETS training-set schema helpers."""

import pandas as pd
import pytest

from tinymlinternship.data.schema import (
    LABELED_REQUIRED_COLUMNS,
    PRELABEL_COLUMNS,
    build_manifest,
    ensure_prelabel_columns,
    split_by_game,
    stm_white_from_fen,
    validate_labeled_frame,
    validate_rewards_series,
)


def test_stm_white_from_fen():
    assert stm_white_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert not stm_white_from_fen(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )


def test_ensure_prelabel_renames_bucket():
    df = pd.DataFrame(
        {
            "fen": ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
            "bucket": [7],
            "piece_count": [32],
            "has_queen": [True],
            "ply": [0],
            "source": ["lc0"],
            "game_id": ["g0"],
        }
    )
    out = ensure_prelabel_columns(df)
    assert "bucket_id" in out.columns
    assert list(PRELABEL_COLUMNS) == [c for c in PRELABEL_COLUMNS if c in out.columns]


def test_split_by_game_no_leakage():
    rows = []
    for g in range(10):
        for p in range(5):
            rows.append(
                {
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "bucket_id": 7,
                    "piece_count": 32,
                    "has_queen": True,
                    "stm_white": True,
                    "ply": p,
                    "source": "lichess",
                    "game_id": f"game_{g}",
                    "expected_reward": 0.0,
                }
            )
    df = pd.DataFrame(rows)
    train, val = split_by_game(df, val_fraction=0.2, seed=0)
    assert len(train) + len(val) == len(df)
    assert set(train["game_id"]).isdisjoint(set(val["game_id"]))
    assert len(val) > 0


def test_validate_rewards_and_labeled():
    validate_rewards_series(pd.Series([-1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        validate_rewards_series(pd.Series([1.01]))

    good = pd.DataFrame(
        {
            col: [0 if col != "fen" else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
            for col in LABELED_REQUIRED_COLUMNS
        }
    )
    good["expected_reward"] = 0.1
    good["bucket_id"] = 7
    good["piece_count"] = 32
    good["has_queen"] = True
    good["stm_white"] = True
    good["ply"] = 0
    good["source"] = "lichess"
    good["game_id"] = "g"
    validate_labeled_frame(good)


def test_build_manifest_counts():
    train = pd.DataFrame(
        {
            "fen": ["f1", "f2"],
            "bucket_id": [7, 6],
            "piece_count": [32, 30],
            "has_queen": [True, True],
            "stm_white": [True, False],
            "ply": [0, 1],
            "source": ["lichess", "lichess"],
            "game_id": ["a", "b"],
            "expected_reward": [0.1, -0.2],
        }
    )
    val = train.iloc[:1].copy()
    m = build_manifest(
        train=train,
        val=val,
        sources=["lichess"],
        teacher_network="791556.pb.gz",
    )
    assert m["train_rows"] == 2
    assert m["val_rows"] == 1
    assert "bucket_counts_all" in m
