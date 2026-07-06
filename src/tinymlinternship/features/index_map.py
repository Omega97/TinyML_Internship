"""
SARDINE 716-dimensional feature index map.

Layout (see NOTES/SARDINE Engine Blueprint.md):
  768 raw piece-square (6 types × 2 colors × 64 squares)
    − 32  pawn ranks 1/8 removed from the index map (hard-zero at runtime)
    − 32  perspective-king plane compressed 64 → 32 (horizontal mirroring)
  +  4  castling rights (WK, WQ, BK, BQ)
  +  8  en-passant file indicators
  = 716

Piece-square block = 704 indices:
  96 pawns + 512 (N/B/R/Q) + 32 perspective-king + 64 enemy-king = 704.

Only the perspective-side king uses mirrored-file compression; the enemy king
keeps full 64-square resolution (rank, file).
"""

from __future__ import annotations

import chess

FEATURE_DIM = 716

PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

_PIECE_SQUARE_INDEX: dict[tuple[bool, int, int], int] = {}
_KING_SELF_INDEX: dict[tuple[int, int], int] = {}
_KING_ENEMY_INDEX: dict[tuple[int, int], int] = {}
_CASTLING_INDEX: dict[int, int] = {}
_EP_FILE_INDEX: dict[int, int] = {}
_PIECE_SQUARE_COUNT = 0
_META_BASE = 0


def is_pawn_rank_inactive(rank: int) -> bool:
    """Ranks 1 and 8 (0 and 7) have no pawn feature slots in the index map."""
    return rank in (0, 7)


def _mirrored_file(file: int) -> int:
    return min(file, 7 - file)


def _build_maps() -> None:
    global _PIECE_SQUARE_COUNT, _META_BASE

    idx = 0

    for side in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_TYPES:
            if piece_type == chess.KING:
                continue
            for square in chess.SQUARES:
                rank = chess.square_rank(square)
                if piece_type == chess.PAWN and is_pawn_rank_inactive(rank):
                    continue
                _PIECE_SQUARE_INDEX[(side, piece_type, square)] = idx
                idx += 1

    for rank in range(8):
        for mf in range(4):
            _KING_SELF_INDEX[(rank, mf)] = idx
            idx += 1

    for rank in range(8):
        for file in range(8):
            _KING_ENEMY_INDEX[(rank, file)] = idx
            idx += 1

    _PIECE_SQUARE_COUNT = idx
    _META_BASE = idx

    castling_sides = (chess.WHITE, chess.WHITE, chess.BLACK, chess.BLACK)
    castling_flags = (chess.KING, chess.QUEEN, chess.KING, chess.QUEEN)
    for i, (side, flag) in enumerate(zip(castling_sides, castling_flags)):
        key = side << 1 | (0 if flag == chess.KING else 1)
        _CASTLING_INDEX[key] = _META_BASE + i

    for file in range(8):
        _EP_FILE_INDEX[file] = _META_BASE + 4 + file

    if idx + 12 != FEATURE_DIM:
        raise RuntimeError(f"piece-square count {idx} + 12 meta != {FEATURE_DIM}")


def piece_square_count() -> int:
    return _PIECE_SQUARE_COUNT


def meta_base() -> int:
    return _META_BASE


def king_self_index(rank: int, mirrored_file: int) -> int:
    """Perspective-king slot (mirrored_file in 0..3)."""
    return _KING_SELF_INDEX[(rank, mirrored_file)]


def king_enemy_index(rank: int, file: int) -> int:
    """Enemy-king slot (full file resolution)."""
    return _KING_ENEMY_INDEX[(rank, file)]


def piece_square_index(
    side: bool,
    piece_type: int,
    square: int,
    *,
    perspective: bool,
) -> int | None:
    """
    Return the feature index for ``(side, piece, square)`` from ``perspective``'s POV.

    Kings route to the 32-slot perspective plane (``side == perspective``) or the
    64-slot enemy plane (``side != perspective``). Pawns on ranks 1/8 return None.
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    if piece_type == chess.PAWN:
        if is_pawn_rank_inactive(rank):
            return None
        return _PIECE_SQUARE_INDEX[(side, piece_type, square)]

    if piece_type == chess.KING:
        if side == perspective:
            return _KING_SELF_INDEX[(rank, _mirrored_file(file))]
        return _KING_ENEMY_INDEX[(rank, file)]

    return _PIECE_SQUARE_INDEX[(side, piece_type, square)]


def castling_index(side: bool, is_kingside: bool) -> int:
    """Index for one castling-right bit (active when right is still available)."""
    key = side << 1 | (0 if is_kingside else 1)
    return _CASTLING_INDEX[key]


def ep_file_index(file: int) -> int:
    """Index for en-passant indicator on file (0..7)."""
    return _EP_FILE_INDEX[file]


def is_valid_index(index: int) -> bool:
    return 0 <= index < FEATURE_DIM


def all_piece_square_indices() -> frozenset[int]:
    return (
        frozenset(_PIECE_SQUARE_INDEX.values())
        | frozenset(_KING_SELF_INDEX.values())
        | frozenset(_KING_ENEMY_INDEX.values())
    )


_build_maps()