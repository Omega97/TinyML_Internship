"""Perft (performance test) — move-count validation for move generation."""

from __future__ import annotations

from typing import Union

import chess

BoardLike = Union[str, chess.Board]

PERFT_STARTPOS = (
    (1, 20),
    (2, 400),
    (3, 8_902),
    (4, 197_281),
    (5, 4_865_609),
)


def _as_board(board: BoardLike) -> chess.Board:
    if isinstance(board, str):
        return chess.Board(board)
    return board.copy()


def perft(board: BoardLike, depth: int) -> int:
    """
    Count leaf nodes at ``depth`` full moves from ``board``.

    Uses ``python-chess`` legal move generation (v0.1 parity baseline).
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    position = _as_board(board)
    return _perft_recursive(position, depth)


def _perft_recursive(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += _perft_recursive(board, depth - 1)
        board.pop()
    return nodes