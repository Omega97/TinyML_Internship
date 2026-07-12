"""Play a full game with the SARDINE engine."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import chess
import chess.pgn

from tinymlinternship.engine import ENGINE_VERSION, search
from tinymlinternship.engine.eval_hce import evaluate_hce
from tinymlinternship.engine.search import EvalFn

OnPlyCallback = Callable[[int, int, chess.Move, float], Any]


def play_engine_game(
    *,
    max_plies: int = 200,
    white_name: str = "SARDINE",
    black_name: str = "SARDINE",
    depth: int = 1,
    eval_fn: EvalFn = evaluate_hce,
    quiescence: bool = True,
    max_qsearch_depth: int | None = None,
    annotator: str | None = None,
    on_ply: OnPlyCallback | None = None,
    max_seconds: float | None = None,
) -> chess.pgn.Game:
    """
    Self-play with fixed-depth search until game over, ``max_plies``, or
    ``max_seconds`` elapsed.

    Returns a ``chess.pgn.Game`` with the main line recorded.
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "SARDINE engine self-play"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    if annotator is None:
        annotator = f"SARDINE {ENGINE_VERSION} (HCE, {depth}-ply)"
    game.headers["Annotator"] = annotator

    node = game
    plies = 0
    game_start = time.perf_counter()
    truncated_time = False

    while not board.is_game_over() and plies < max_plies:
        if max_seconds is not None and (time.perf_counter() - game_start) >= max_seconds:
            truncated_time = True
            break
        t0 = time.perf_counter()
        result = search(
            board,
            depth,
            eval_fn=eval_fn,
            quiescence=quiescence,
            max_qsearch_depth=max_qsearch_depth,
        )
        if result is None:
            break
        ply_sec = time.perf_counter() - t0
        node = node.add_variation(result.move)
        board.push(result.move)
        plies += 1
        if on_ply is not None:
            on_ply(plies, max_plies, result.move, ply_sec)

    if truncated_time:
        game.headers["Termination"] = "time limit"

    outcome = board.outcome()
    if outcome is not None:
        game.headers["Result"] = outcome.result()
    else:
        game.headers["Result"] = "*"

    return game