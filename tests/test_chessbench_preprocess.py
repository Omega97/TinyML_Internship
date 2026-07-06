"""Tests for ChessBench → SARDINE training row conversion."""

from __future__ import annotations

import chess
import pytest

from tinymlinternship.data.chessbench_preprocess import (
    parse_chessbench_row,
    win_prob_to_expected_reward,
)
from tinymlinternship.features import FEATURE_DIM, encode_dual


def test_win_prob_to_expected_reward_mapping():
    assert win_prob_to_expected_reward(0.0) == -1.0
    assert win_prob_to_expected_reward(0.5) == 0.0
    assert win_prob_to_expected_reward(1.0) == 1.0
    assert win_prob_to_expected_reward(0.7218503805722687) == pytest.approx(0.4437007611445374)


def test_parse_chessbench_row_startpos():
    row = parse_chessbench_row(chess.STARTING_FEN, 0.5)
    assert row is not None
    assert row.expected_reward == 0.0
    assert row.bucket_id == 7
    assert row.stm_white is True
    white, black = encode_dual(chess.Board())
    assert row.white_features == white
    assert row.black_features == black
    assert all(0 <= idx < FEATURE_DIM for idx in row.white_features)


def test_parse_chessbench_row_invalid_fen():
    assert parse_chessbench_row("not-a-fen", 0.5) is None