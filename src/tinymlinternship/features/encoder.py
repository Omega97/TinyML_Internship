"""
Sparse 716-feature encoder for SARDINE NNUE input.

Dual-perspective contract (NOTES/NNUE.md):
  - White accumulator input: encode from White's POV (king mirroring applied).
  - Black accumulator input: encode from White's POV on ``board.mirror()``.
"""

from __future__ import annotations

import chess
from typing import Union

from tinymlinternship.features.index_map import (
    castling_index,
    ep_file_index,
    is_valid_index,
    piece_square_index,
)
from tinymlinternship.features.mirror import perspective_board

BoardLike = Union[str, chess.Board]


def _as_board(board: BoardLike) -> chess.Board:
    if isinstance(board, str):
        return chess.Board(board)
    return board.copy()


def _append_meta_features(board: chess.Board, out: set[int]) -> None:
    for side in (chess.WHITE, chess.BLACK):
        if board.has_kingside_castling_rights(side):
            out.add(castling_index(side, is_kingside=True))
        if board.has_queenside_castling_rights(side):
            out.add(castling_index(side, is_kingside=False))

    if board.ep_square is not None:
        out.add(ep_file_index(chess.square_file(board.ep_square)))


def encode_perspective(board: BoardLike, perspective: chess.Color) -> list[int]:
    """
    Return sorted active feature indices for one king perspective.

    Applies horizontal king mirroring for ``perspective`` before indexing pieces.
    Castling and en-passant bits follow the mirrored board state.
    """
    base = _as_board(board)
    view = perspective_board(base, perspective)
    active: set[int] = set()

    for square in chess.SQUARES:
        piece = view.piece_at(square)
        if piece is None:
            continue
        idx = piece_square_index(piece.color, piece.piece_type, square)
        if idx is not None:
            active.add(idx)

    # Meta features use the unmirrored board (castling/EP are not king-mirror dependent).
    _append_meta_features(base, active)
    return sorted(active)


def encode_dual(board: BoardLike) -> tuple[list[int], list[int]]:
    """
    Return (white_perspective, black_perspective) active feature lists.

    Black perspective uses ``board.mirror()`` so the network sees the position
    from Black's side of the board.
    """
    base = _as_board(board)
    white_features = encode_perspective(base, chess.WHITE)
    black_features = encode_perspective(base.mirror(), chess.WHITE)
    return white_features, black_features


def validate_features(features: list[int]) -> None:
    """Raise ValueError if indices are out of range or duplicated."""
    if len(features) != len(set(features)):
        raise ValueError("duplicate feature indices")
    for idx in features:
        if not is_valid_index(idx):
            raise ValueError(f"feature index out of range: {idx}")