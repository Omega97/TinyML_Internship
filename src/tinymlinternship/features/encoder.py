"""
Sparse 844-feature encoder for SARDINE NNUE input.

Spec: NOTES/SARDINE Engine Blueprint.md (Input features).

Dual-perspective contract (NOTES/NNUE.md):
  - Own accumulator input: encode from side-to-move POV (king mirroring applied).
  - Opponent accumulator input: encode from opponent POV on ``board.mirror()``.

Castling: rights read from ``base`` board; kingside/queenside labels swapped when
horizontal mirror was applied (``_append_castling_features``). EP from mirrored view.
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
from tinymlinternship.features.mirror import needs_horizontal_mirror, perspective_board
from tinymlinternship.features.tactical import append_tactical_features

BoardLike = Union[str, chess.Board]


def _as_board(board: BoardLike) -> chess.Board:
    if isinstance(board, str):
        return chess.Board(board)
    return board.copy()


def _append_castling_features(
    base: chess.Board,
    out: set[int],
    *,
    horizontally_mirrored: bool,
) -> None:
    """
    Emit castling features in the same coordinate frame as the mirrored piece view.

    ``chess.Board.has_kingside_castling_rights`` is unreliable on a flipped board
    (king/rook geometry changes), so rights are read from ``base`` and kingside/
    queenside labels are swapped when a horizontal mirror was applied.
    """
    for side in (chess.WHITE, chess.BLACK):
        if base.has_kingside_castling_rights(side):
            out.add(castling_index(side, is_kingside=not horizontally_mirrored))
        if base.has_queenside_castling_rights(side):
            out.add(castling_index(side, is_kingside=horizontally_mirrored))


def _append_ep_feature(board: chess.Board, out: set[int]) -> None:
    """EP is file-local — must match the same coordinate frame as piece squares."""
    if board.ep_square is not None:
        out.add(ep_file_index(chess.square_file(board.ep_square)))


def encode_perspective(board: BoardLike, perspective: chess.Color) -> list[int]:
    """
    Return sorted active feature indices for one king perspective.

    Applies horizontal king mirroring for ``perspective`` before indexing pieces.
    Castling and en-passant both follow the mirrored view (same frame as pieces).
    """
    base = _as_board(board)
    view = perspective_board(base, perspective)
    active: set[int] = set()

    for square in chess.SQUARES:
        piece = view.piece_at(square)
        if piece is None:
            continue
        idx = piece_square_index(
            piece.color, piece.piece_type, square, perspective=perspective
        )
        if idx is not None:
            active.add(idx)

    _append_castling_features(
        base, active, horizontally_mirrored=needs_horizontal_mirror(base, perspective)
    )
    _append_ep_feature(view, active)
    append_tactical_features(view, perspective, active)
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