"""
SARDINE search — negamax alpha-beta with pluggable static eval.

v0.3: capture-only quiescence at depth-0 leaves.
v0.2: fixed-depth alpha-beta (``search``).
v0.1: ``search_best_move`` is depth-1 search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import chess

from tinymlinternship.engine.eval_hce import MATE_SCORE, evaluate_hce

BoardLike = Union[str, chess.Board]
EvalFn = Callable[[chess.Board], int]


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move
    score: int
    nodes: int
    depth: int


def _as_board(board: BoardLike) -> chess.Board:
    if isinstance(board, str):
        return chess.Board(board)
    return board.copy()


def _move_order_key(board: chess.Board, move: chess.Move) -> tuple[int, int]:
    """Captures first (MVV-LVA-ish), then quiet moves."""
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_value = 0 if victim is None else victim.piece_type
        attacker_value = 0 if attacker is None else attacker.piece_type
        return (0, victim_value * 16 - attacker_value)
    return (1, move.uci())


def _ordered_moves(board: chess.Board) -> list[chess.Move]:
    return sorted(board.legal_moves, key=lambda m: _move_order_key(board, m))


def _is_noisy(board: chess.Board, move: chess.Move) -> bool:
    """Captures and promotions extend the horizon in quiescence."""
    return board.is_capture(move) or move.promotion is not None


def _noisy_moves(board: chess.Board) -> list[chess.Move]:
    return [m for m in _ordered_moves(board) if _is_noisy(board, m)]


def _eval_stm(board: chess.Board, eval_fn: EvalFn) -> int:
    """Static eval from side-to-move perspective (for negamax)."""
    score = eval_fn(board)
    return score if board.turn == chess.WHITE else -score


def search(
    board: BoardLike,
    depth: int,
    *,
    eval_fn: EvalFn = evaluate_hce,
    quiescence: bool = True,
) -> SearchResult | None:
    """
    Fixed-depth negamax alpha-beta search with optional capture quiescence.

    ``score`` is centipawns from **White's** perspective at the resulting position
    (same convention as ``evaluate_hce``). Returns ``None`` if there are no legal moves.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    position = _as_board(board)
    legal = _ordered_moves(position)
    if not legal:
        return None

    nodes = 0

    def qsearch(node: chess.Board, alpha: int, beta: int) -> int:
        nonlocal nodes
        nodes += 1

        if node.is_game_over():
            return _eval_stm(node, eval_fn)

        stand_pat = _eval_stm(node, eval_fn)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        for move in _noisy_moves(node):
            node.push(move)
            score = -qsearch(node, -beta, -alpha)
            node.pop()

            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return alpha

    def negamax(node: chess.Board, remaining: int, alpha: int, beta: int) -> int:
        nonlocal nodes
        nodes += 1

        if node.is_game_over():
            return _eval_stm(node, eval_fn)

        if remaining == 0:
            if quiescence:
                return qsearch(node, alpha, beta)
            return _eval_stm(node, eval_fn)

        value = -MATE_SCORE
        for move in _ordered_moves(node):
            node.push(move)
            score = -negamax(node, remaining - 1, -beta, -alpha)
            node.pop()

            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value

    best_move: chess.Move | None = None
    best_score = -MATE_SCORE
    alpha = -MATE_SCORE
    beta = MATE_SCORE

    for move in legal:
        position.push(move)
        score = -negamax(position, depth - 1, -beta, -alpha)
        position.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    assert best_move is not None
    position.push(best_move)
    report_score = eval_fn(position)
    position.pop()
    return SearchResult(move=best_move, score=report_score, nodes=nodes, depth=depth)


def search_best_move(board: BoardLike, *, eval_fn: EvalFn = evaluate_hce) -> SearchResult | None:
    """Depth-1 search (v0.1 API)."""
    return search(board, 1, eval_fn=eval_fn)