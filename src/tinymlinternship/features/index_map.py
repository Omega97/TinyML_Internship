"""
SARDINE 716-dimensional feature index map.

Layout (see NOTES/SARDINE 🐟.md):
  768 piece-square (6 types × 2 colors × 64 squares)
    − 32  impossible pawn ranks (1st/8th rank, both colors)
    − 32  king-plane compression (64 → 32 slots per color via horizontal mirroring)
  +  4  castling rights (WK, WQ, BK, BQ)
  +  8  en-passant file indicators
  = 716

Piece-square indices are assigned contiguously in (side, piece, square) order.
Pawn slots on ranks 1/8 stay in the map but are never activated (hard-zero at runtime).
King squares alias into 32 mirrored slots per color (−32 per color vs a full 64-plane).
Meta features follow the 704 piece-square block.
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

# Populated at import time.
_PIECE_SQUARE_INDEX: dict[tuple[bool, int, int], int] = {}
_KING_SLOT_INDEX: dict[tuple[bool, int, int], int] = {}
_CASTLING_INDEX: dict[int, int] = {}
_EP_FILE_INDEX: dict[int, int] = {}
_PIECE_SQUARE_COUNT = 0
_META_BASE = 0


def is_pawn_rank_inactive(rank: int) -> bool:
    """Ranks 1 and 8 (0 and 7) never activate pawn features."""
    return rank in (0, 7)


def _build_maps() -> None:
    global _PIECE_SQUARE_COUNT, _META_BASE

    idx = 0

    for side in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_TYPES:
            if piece_type == chess.KING:
                for rank in range(8):
                    for mf in range(4):
                        _KING_SLOT_INDEX[(side, rank, mf)] = idx
                        idx += 1
                for square in chess.SQUARES:
                    rank = chess.square_rank(square)
                    file = chess.square_file(square)
                    mf = min(file, 7 - file)
                    _PIECE_SQUARE_INDEX[(side, piece_type, square)] = _KING_SLOT_INDEX[
                        (side, rank, mf)
                    ]
            else:
                for square in chess.SQUARES:
                    _PIECE_SQUARE_INDEX[(side, piece_type, square)] = idx
                    idx += 1

    _PIECE_SQUARE_COUNT = idx
    _META_BASE = idx

    castling_order = (
        chess.KING,
        chess.QUEEN,
        chess.KING,
        chess.QUEEN,
    )
    castling_sides = (chess.WHITE, chess.WHITE, chess.BLACK, chess.BLACK)
    for i, (side, flag) in enumerate(zip(castling_sides, castling_order)):
        _CASTLING_INDEX[side << 1 | (0 if flag == chess.KING else 1)] = _META_BASE + i

    for file in range(8):
        _EP_FILE_INDEX[file] = _META_BASE + 4 + file

    if idx + 12 != FEATURE_DIM:
        raise RuntimeError(f"piece-square count {idx} + 12 meta != {FEATURE_DIM}")


def piece_square_count() -> int:
    return _PIECE_SQUARE_COUNT


def meta_base() -> int:
    return _META_BASE


def piece_square_index(side: bool, piece_type: int, square: int) -> int | None:
    """
    Return the feature index for an occupied (side, piece, square), or None if inactive.

    For kings, multiple squares alias to the same index (horizontal mirror classes).
    Pawns on ranks 1 and 8 (0 and 7) keep indices but return None (never activated).
    """
    if piece_type == chess.PAWN and is_pawn_rank_inactive(chess.square_rank(square)):
        return None
    if piece_type == chess.KING:
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        mf = min(file, 7 - file)
        return _KING_SLOT_INDEX[(side, rank, mf)]
    return _PIECE_SQUARE_INDEX.get((side, piece_type, square))


def king_slot_index(side: bool, rank: int, mirrored_file: int) -> int:
    """Direct lookup for a canonical king slot (mirrored_file in 0..3)."""
    return _KING_SLOT_INDEX[(side, rank, mirrored_file)]


def castling_index(side: bool, is_kingside: bool) -> int:
    """Index for one castling-right bit (active when right is still available)."""
    key = side << 1 | (0 if is_kingside else 1)
    return _CASTLING_INDEX[key]


def ep_file_index(file: int) -> int:
    """Index for en-passant indicator on file (0..7); active when EP square is on that file."""
    return _EP_FILE_INDEX[file]


def is_valid_index(index: int) -> bool:
    return 0 <= index < FEATURE_DIM


def all_piece_square_indices() -> frozenset[int]:
    """Set of indices used by at least one legal piece-square assignment."""
    return frozenset(_PIECE_SQUARE_INDEX.values()) | frozenset(_KING_SLOT_INDEX.values())


_build_maps()