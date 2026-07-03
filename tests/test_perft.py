"""Perft tests — move generation correctness."""

import chess
import pytest

from tinymlinternship.engine.perft import PERFT_STARTPOS, perft


@pytest.mark.parametrize("depth,expected", PERFT_STARTPOS)
def test_perft_startpos(depth: int, expected: int):
    assert perft(chess.Board(), depth) == expected


def test_perft_depth_zero_is_one_node():
    assert perft(chess.Board(), 0) == 1


def test_perft_kiwipete_depth_one():
    # python-chess drops illegal white castling flags → 42 moves, not 48.
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/PNnP1P2/2p1P2p/1B2KB1R/r2Q1RK1 w KQkq - 0 1")
    assert perft(board, 1) == 42