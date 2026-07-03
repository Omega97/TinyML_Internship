"""Play a full game with the SARDINE engine."""

from __future__ import annotations

import chess
import chess.pgn

from tinymlinternship.engine import ENGINE_VERSION, search_best_move


def play_engine_game(
    *,
    max_plies: int = 200,
    white_name: str = "SARDINE",
    black_name: str = "SARDINE",
) -> chess.pgn.Game:
    """
    Self-play with 1-ply HCE search until game over or ``max_plies`` reached.

    Returns a ``chess.pgn.Game`` with the main line recorded.
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "SARDINE engine self-play"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Annotator"] = f"SARDINE {ENGINE_VERSION} (HCE, 1-ply)"

    node = game
    plies = 0

    while not board.is_game_over() and plies < max_plies:
        result = search_best_move(board)
        if result is None:
            break
        node = node.add_variation(result.move)
        board.push(result.move)
        plies += 1

    outcome = board.outcome()
    if outcome is not None:
        game.headers["Result"] = outcome.result()
    else:
        game.headers["Result"] = "*"

    return game