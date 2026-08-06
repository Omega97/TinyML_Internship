"""
SARDINE search — negamax alpha-beta with pluggable static eval.

v0.3.1: timed iterative deepening (``search_timed`` / movetime).
v0.3: capture-only quiescence at depth-0 leaves.
v0.2: fixed-depth alpha-beta (``search``).
v0.1: ``search_best_move`` is depth-1 search.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import chess

from tinymlinternship.engine.eval_hce import MATE_SCORE, evaluate_hce

BoardLike = Union[str, chess.Board]
EvalFn = Callable[[chess.Board], int]


class _TimeUp(Exception):
    """Internal: search aborted because the wall-clock deadline was hit."""


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
    max_qsearch_depth: int | None = None,
    deadline: float | None = None,
) -> SearchResult | None:
    """
    Fixed-depth negamax alpha-beta search with optional capture quiescence.

    ``max_qsearch_depth`` caps noisy-move extensions at depth-0 leaves (``None`` =
    unlimited). Each capture/promotion in qsearch consumes one ply of this budget.

    ``deadline`` is an optional ``time.perf_counter()`` wall time; if the search
    exceeds it mid-tree, raises ``_TimeUp`` (used by :func:`search_timed`).

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

    def _check_time() -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise _TimeUp()

    def qsearch(
        node: chess.Board,
        alpha: int,
        beta: int,
        qremaining: int | None,
    ) -> int:
        nonlocal nodes
        nodes += 1
        if nodes & 255 == 0:
            _check_time()

        if node.is_game_over():
            return _eval_stm(node, eval_fn)

        stand_pat = _eval_stm(node, eval_fn)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if qremaining is not None and qremaining <= 0:
            return alpha

        for move in _noisy_moves(node):
            node.push(move)
            try:
                next_q = None if qremaining is None else qremaining - 1
                score = -qsearch(node, -beta, -alpha, next_q)
            finally:
                node.pop()

            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return alpha

    def negamax(node: chess.Board, remaining: int, alpha: int, beta: int) -> int:
        nonlocal nodes
        nodes += 1
        if nodes & 255 == 0:
            _check_time()

        if node.is_game_over():
            return _eval_stm(node, eval_fn)

        if remaining == 0:
            if quiescence:
                return qsearch(node, alpha, beta, max_qsearch_depth)
            return _eval_stm(node, eval_fn)

        value = -MATE_SCORE
        for move in _ordered_moves(node):
            node.push(move)
            try:
                score = -negamax(node, remaining - 1, -beta, -alpha)
            finally:
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

    try:
        for move in legal:
            _check_time()
            position.push(move)
            try:
                score = -negamax(position, depth - 1, -beta, -alpha)
            except _TimeUp:
                position.pop()
                raise
            position.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
    except _TimeUp:
        # Keep any fully scored root moves; otherwise let caller fall back.
        if best_move is None:
            raise

    if best_move is None:
        # Should be unreachable (empty legal handled above); keep search robust.
        best_move = legal[0]

    position.push(best_move)
    report_score = eval_fn(position)
    position.pop()
    return SearchResult(move=best_move, score=report_score, nodes=nodes, depth=depth)


def search_timed(
    board: BoardLike,
    movetime_s: float,
    *,
    eval_fn: EvalFn = evaluate_hce,
    quiescence: bool = True,
    max_qsearch_depth: int | None = None,
    max_depth: int = 64,
) -> SearchResult | None:
    """
    Iterative deepening under a **per-move wall-clock budget** (seconds).

    Depth is not fixed: the search runs depth 1, 2, … until ``movetime_s`` is
    exhausted or ``max_depth`` is reached. If a depth is aborted mid-search,
    the last fully completed depth is kept. Depth 1 is always completed even
    if the budget is already exhausted (need a legal move).
    """
    if movetime_s <= 0:
        raise ValueError(f"movetime_s must be > 0, got {movetime_s}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    position = _as_board(board)
    if not any(position.legal_moves):
        return None

    deadline = time.perf_counter() + movetime_s
    best: SearchResult | None = None
    total_nodes = 0

    for depth in range(1, max_depth + 1):
        if time.perf_counter() >= deadline and best is not None:
            break
        try:
            # Always respect the wall clock so deeper ID iterations get budget.
            # If depth-1 is aborted with no scored root move, fall through to fallback.
            result = search(
                position,
                depth,
                eval_fn=eval_fn,
                quiescence=quiescence,
                max_qsearch_depth=max_qsearch_depth,
                deadline=deadline,
            )
        except _TimeUp:
            break
        if result is None:
            break
        total_nodes += result.nodes
        best = SearchResult(
            move=result.move,
            score=result.score,
            nodes=total_nodes,
            depth=result.depth,
        )

    if best is None:
        # Budget exhausted before any root move finished: greedy 1-ply static pick.
        pick: chess.Move | None = None
        pick_score = -MATE_SCORE
        for move in _ordered_moves(position):
            position.push(move)
            # After the move, STM is the opponent; maximise the mover's score.
            sc = -_eval_stm(position, eval_fn)
            position.pop()
            if sc > pick_score:
                pick_score = sc
                pick = move
        assert pick is not None
        position.push(pick)
        report = eval_fn(position)
        position.pop()
        return SearchResult(move=pick, score=report, nodes=total_nodes, depth=0)
    return best


def search_best_move(board: BoardLike, *, eval_fn: EvalFn = evaluate_hce) -> SearchResult | None:
    """Depth-1 search (v0.1 API)."""
    return search(board, 1, eval_fn=eval_fn)