"""Tests for Lc0 teacher eval helpers (no lc0 binary required)."""

import chess

from tinymlinternship.engine.eval_lc0 import (
    expected_reward_to_cp,
    parse_wdl_permille,
    wdl_to_expected_reward_white,
)


def test_parse_wdl_permille():
    line = "info depth 1 score cp 5 wdl 191 642 167 pv e2e4"
    assert parse_wdl_permille(line) == (191, 642, 167)


def test_wdl_white_startpos_slight_edge():
    board = chess.Board()
    reward = wdl_to_expected_reward_white(board, 191, 642, 167)
    assert 0.0 < reward < 0.1


def test_wdl_black_to_move_flips_sign():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    white_reward = wdl_to_expected_reward_white(chess.Board(), 191, 642, 167)
    black_reward = wdl_to_expected_reward_white(board, 167, 642, 191)
    assert abs(white_reward + black_reward) < 0.05


def test_expected_reward_to_cp_scale():
    assert expected_reward_to_cp(0.024) == 24
    assert expected_reward_to_cp(-1.0) == -1000