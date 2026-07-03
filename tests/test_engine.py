"""Tests for SARDINE engine (HCE + alpha-beta search)."""

import chess

from tinymlinternship.engine import evaluate_hce, search, search_best_move
from tinymlinternship.engine.eval_hce import MATE_SCORE


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
    assert result.nodes >= 20


def test_search_black_minimizes_white_score():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    result = search_best_move(board)
    assert result is not None
    assert result.move in board.legal_moves
    assert result.nodes >= 20
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


def test_search_depth_one_matches_best_move():
    board = chess.Board()
    one = search_best_move(board)
    two = search(board, 1)
    assert one is not None and two is not None
    assert one.move == two.move
    assert one.score == two.score
    assert two.depth == 1


def test_search_finds_mating_move_depth_two():
    # Qg7# (HCE sees mate at the leaf).
    board = chess.Board("7k/5Q2/5K2/8/8/8/8/8 w - - 0 1")
    result = search(board, 2)
    assert result is not None
    assert result.move.uci() == "f7g7"
    assert result.score >= MATE_SCORE - 100


def test_search_depth_two_winning_capture():
    board = chess.Board("4qk2/8/8/8/8/8/8/4R1K1 w - - 0 1")
    result = search(board, 2)
    assert result is not None
    assert result.move.uci() == "e1e8"
    assert result.score >= 500


def test_search_startpos_depth_two_not_blunder():
    board = chess.Board()
    result = search(board, 2)
    assert result is not None
    assert result.move in board.legal_moves
    assert -200 <= result.score <= 200


def test_search_pluggable_eval():
    board = chess.Board("4qk2/8/8/8/8/8/8/4R1K1 w - - 0 1")

    def constant_eval(_: chess.Board) -> int:
        return 0

    result = search(board, 2, eval_fn=constant_eval)
    assert result is not None
    assert result.score == 0


def test_search_depth_one_without_quiescence_nodes():
    board = chess.Board()
    result = search(board, 1, quiescence=False)
    assert result is not None
    assert result.nodes == 20


def test_quiescence_avoids_hanging_promotion():
    # Black promotes; without qsearch the h1 queen is lost to Kxh1 at the horizon.
    board = chess.Board("8/8/8/8/8/8/7p/6K1 b - - 0 1")
    with_q = search(board, 1, quiescence=True)
    without_q = search(board, 1, quiescence=False)
    assert with_q is not None and without_q is not None
    assert with_q.move.uci() == "h2g1q"
    assert without_q.move.uci() == "h2h1q"
    assert with_q.score >= -100
    assert without_q.score <= -500


def test_quiescence_default_on():
    board = chess.Board("8/8/8/8/8/8/7p/6K1 b - - 0 1")
    result = search(board, 1)
    assert result is not None
    assert result.move.uci() == "h2g1q"