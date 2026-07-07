"""Tests for tactical feature planes."""

import chess

from tinymlinternship.features import (
    king_attacker_index,
    tactical_base,
    under_attack_index,
)
from tinymlinternship.features.tactical import append_tactical_features


def test_tactical_index_layout():
    assert tactical_base() == 716
    assert under_attack_index(chess.A1) == 716
    assert under_attack_index(chess.H8) == 779
    assert king_attacker_index(chess.A1) == 780
    assert king_attacker_index(chess.H8) == 843


def test_append_tactical_features_isolated():
    board = chess.Board("8/8/8/8/4q3/8/8/4Q3 w - - 0 1")
    active: set[int] = set()
    append_tactical_features(board, chess.WHITE, active)
    assert under_attack_index(chess.E1) in active
    assert king_attacker_index(chess.E4) not in active