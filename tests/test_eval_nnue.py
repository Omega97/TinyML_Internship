"""Tests for NNUE engine eval (requires trained checkpoint)."""

from __future__ import annotations

import chess
import pytest

from tinymlinternship.config.settings import NNUE_CHECKPOINT_DEFAULT
from tinymlinternship.engine import evaluate_nnue, search, search_best_move
from tinymlinternship.engine.eval_nnue import NnueEvaluator, stm_reward_to_white
from tinymlinternship.features import FEATURE_DIM


def _checkpoint_matches_feature_dim() -> bool:
    if not NNUE_CHECKPOINT_DEFAULT.exists():
        return False
    import torch

    payload = torch.load(NNUE_CHECKPOINT_DEFAULT, map_location="cpu", weights_only=True)
    weight = payload["model_state_dict"]["l1.weight"]
    return weight.shape[1] == FEATURE_DIM


pytestmark = pytest.mark.skipif(
    not _checkpoint_matches_feature_dim(),
    reason="NNUE checkpoint missing or trained for old feature dim",
)


def test_stm_reward_to_white_flips_for_black():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert stm_reward_to_white(board, 0.2) == pytest.approx(-0.2)


def test_evaluate_nnue_startpos_reasonable():
    score = evaluate_nnue(chess.Board())
    assert -150 <= score <= 150


def test_evaluate_nnue_material_advantage_positive():
    board = chess.Board("8/8/8/8/8/8/8/4Q2K w - - 0 1")
    assert evaluate_nnue(board) > 200


def test_nnue_evaluator_matches_module_fn():
    board = chess.Board()
    ev = NnueEvaluator()
    assert ev.evaluate_cp(board) == evaluate_nnue(board)


def test_search_with_nnue_finds_move():
    result = search_best_move(chess.Board(), eval_fn=evaluate_nnue)
    assert result is not None
    assert result.move in chess.Board().legal_moves
    assert result.nodes >= 20


def test_search_nnue_depth_two_startpos():
    result = search(chess.Board(), 2, eval_fn=evaluate_nnue)
    assert result is not None
    assert -300 <= result.score <= 300