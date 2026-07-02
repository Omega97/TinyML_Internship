"""
NNUE output-bucket selector (SARDINE B-C scheme).

Routes a position to one of 8 expert heads using piece count and queen presence only.
"""

from __future__ import annotations

import chess
from typing import Union

BoardLike = Union[str, chess.Board]

NUM_BUCKETS = 8


def _as_board(board: BoardLike) -> chess.Board:
    if isinstance(board, str):
        return chess.Board(board)
    return board


def piece_count(board: BoardLike) -> int:
    """Number of pieces on the board (kings included)."""
    return len(_as_board(board).piece_map())


def has_queen(board: BoardLike) -> bool:
    """True if either side still has a queen."""
    return any(p.piece_type == chess.QUEEN for p in _as_board(board).piece_map().values())


def bucket_id(board: BoardLike) -> int:
    """
    Return the training/inference bucket index in ``[0, 7]``.

    Piece count ``p`` includes both kings. Queen presence is OR across colors.
    """
    b = _as_board(board)
    p = piece_count(b)

    if p <= 12:
        return 0
    if p == 32:
        return 7

    queen = has_queen(b)
    if 13 <= p <= 21:
        return 2 if queen else 1
    if 22 <= p <= 27:
        return 4 if queen else 3
    if 28 <= p <= 31:
        return 6 if queen else 5

    raise ValueError(f"unexpected piece count for bucket routing: {p}")