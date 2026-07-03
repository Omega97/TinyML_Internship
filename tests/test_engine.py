"""Tests for SARDINE engine v0.1 (HCE + 1-ply search)."""

import chess

from tinymlinternship.engine import evaluate_hce, search_best_move


def test_eval_startpos_near_equal():
    score = evaluate_hce(chess.Board())
    assert -50 <= score <= 50


def test_eval_material_advantage():
    board = chess.Board("8/8/8/8/8/8/8/4Q2K w - - 0 1")
    assert evaluate_hce(board) >= 850


def test_eval_mate_for_side_to_move():
    # Black to move, checkmated.
    board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    assert board.is_checkmate()
    assert evaluate_hce(board) == 30_000


def test_search_startpos_one_ply():
    board = chess.Board()
    result = search_best_move(board)
    assert result is not None
    assert result.move in board.legal_moves
    assert result.nodes == 20


def test_search_black_minimizes_white_score():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    result = search_best_move(board)
    assert result is not None
    assert result.move in board.legal_moves
    assert result.nodes == 20
    # Black should not blunder into a huge White advantage.
    assert result.score <= 100


def test_search_handles_mate_threat():
    # Queen on h2; best reply captures it or leaves White well ahead.
    board = chess.Board("8/8/8/8/8/8/7q/6K1 w - - 0 1")
    result = search_best_move(board)
    assert result is not None
    assert result.score >= -100


def test_search_picks_winning_capture():
    board = chess.Board("4qk2/8/8/8/8/8/8/4R1K1 w - - 0 1")
    result = search_best_move(board)
    assert result is not None
    assert result.move.uci() == "e1e8"
    assert result.score >= 500


def test_search_no_legal_moves():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert search_best_move(board) is None