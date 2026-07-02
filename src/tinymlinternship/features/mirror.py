"""
Horizontal king mirroring for SARDINE feature encoding.

When the perspective-side king sits on files e–h, the board is flipped horizontally
so the king lands on files a–d before piece-square indices are computed. This matches
the compressed 32-slot king plane in index_map.py.

Incremental NNUE updates must treat a centre-file crossing as a full accumulator
refresh (deferred to build step 5).
"""

from __future__ import annotations

import chess
from typing import Union

BoardLike = Union[str, chess.Board]


def _as_board(board: BoardLike) -> chess.Board:
    if isinstance(board, str):
        return chess.Board(board)
    return board.copy()


def king_square(board: chess.Board, perspective: chess.Color) -> int | None:
    return board.king(perspective)


def needs_horizontal_mirror(board: chess.Board, perspective: chess.Color) -> bool:
    """True when the perspective king is on files e–h (file index ≥ 4)."""
    sq = king_square(board, perspective)
    if sq is None:
        return False
    return chess.square_file(sq) >= 4


def mirror_for_perspective(board: chess.Board, perspective: chess.Color) -> chess.Board:
    """
    Return a board ready for feature indexing from ``perspective``'s point of view.

    Applies ``chess.flip_horizontal`` when the perspective king is on the right half.
    """
    if needs_horizontal_mirror(board, perspective):
        return board.transform(chess.flip_horizontal)
    return board.copy()


def perspective_board(board: BoardLike, perspective: chess.Color) -> chess.Board:
    """Copy ``board`` and apply perspective king mirroring if required."""
    return mirror_for_perspective(_as_board(board), perspective)