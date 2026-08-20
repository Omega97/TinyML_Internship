"""Lichess puzzle FEN is before the opponent's setup move."""

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_lichess_puzzles import puzzle_player_fen  # noqa: E402

RAW = "r1bqk2r/pp1nbNp1/2p1p2p/8/2BP4/1PN3P1/P3QP1P/3R1RK1 b kq - 0 19"
MOVES = "e8f7 e2e6 f7f8 e6f7"


def test_puzzle_player_fen_pushes_setup_move():
    expected = chess.Board(RAW)
    expected.push_uci("e8f7")
    assert puzzle_player_fen(RAW, MOVES) == expected.fen()
    assert chess.Board(puzzle_player_fen(RAW, MOVES)).turn == chess.WHITE
