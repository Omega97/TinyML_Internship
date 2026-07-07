"""
Per-square tactical features for SARDINE NNUE input (844-dim extension).

Two 64-bit planes appended after the base 716 features (perspective frame):
  - pieces under attack: own piece on square s is attacked by opponent
  - king attackers: opponent piece on square s attacks the perspective king
"""

from __future__ import annotations

import chess

from tinymlinternship.features.index_map import king_attacker_index, under_attack_index


def append_tactical_features(
    view: chess.Board,
    perspective: chess.Color,
    active: set[int],
) -> None:
    """Add under-attack and king-attacker square features in ``view`` coordinates."""
    opponent = not perspective
    king_sq = view.king(perspective)

    for square in chess.SQUARES:
        piece = view.piece_at(square)
        if piece is None:
            continue

        if piece.color == perspective and view.is_attacked_by(opponent, square):
            active.add(under_attack_index(square))

        if (
            king_sq is not None
            and piece.color == opponent
            and king_sq in view.attacks(square)
        ):
            active.add(king_attacker_index(square))