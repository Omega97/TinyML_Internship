"""
SARDINE search v0.1 — 1-ply lookahead (static eval after each legal move).

Side to move picks the move that maximizes its own outcome:
  White maximizes eval (White-positive centipawns).
  Black minimizes eval (equivalently maximizes Black's standing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import chess

from tinymlinternship.engine.eval_hce import MATE_SCORE, evaluate_hce

BoardLike = Union[str, chess.Board]


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move
    score: int
    nodes: int


def _as_board(board: BoardLike) -> chess.Board:
    if isinstance(board, str):
        return chess.Board(board)
    return board.copy()


def _move_order_key(board: chess.Board, move: chess.Move) -> tuple[int, int]:
    """Captures first, then quiet moves (stable tie-break by UCI)."""
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_value = 0 if victim is None else victim.piece_type
        attacker_value = 0 if attacker is None else attacker.piece_type
        return (0, victim_value * 16 - attacker_value)
    return (1, move.uci())


def search_best_move(board: BoardLike) -> SearchResult | None:
    """
    Return the best move at depth 1 and its eval after the move.

    ``score`` is always in centipawns from White's perspective *after* the move.
    Returns ``None`` if there are no legal moves (checkmate/stalemate).
    """
    position = _as_board(board)
    legal = list(position.legal_moves)
    if not legal:
        return None

    maximizing = position.turn == chess.WHITE
    best_move: chess.Move | None = None
    best_score = -MATE_SCORE if maximizing else MATE_SCORE
    nodes = 0

    for move in sorted(legal, key=lambda m: _move_order_key(position, m)):
        position.push(move)
        score = evaluate_hce(position)
        position.pop()
        nodes += 1

        if maximizing:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            if score < best_score:
                best_score = score
                best_move = move

    assert best_move is not None
    return SearchResult(move=best_move, score=best_score, nodes=nodes)