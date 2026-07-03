#!/usr/bin/env python3
"""Play a SARDINE engine game, display it with pygame, and save a GIF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.pgn

from tinymlinternship.visualization import (
    PygameBoardRenderer,
    export_game_gif,
    play_engine_game,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a SARDINE engine game as GIF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "sardine_game.gif",
        help="Output GIF path (default: project root / sardine_game.gif)",
    )
    parser.add_argument("--max-plies", type=int, default=200, help="Max half-moves")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No pygame window (still builds frames for optional pygame GIF)",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=450,
        help="Milliseconds per GIF frame",
    )
    parser.add_argument(
        "--exporter",
        choices=("gifpgn", "chess_gif", "pygame"),
        default="gifpgn",
        help="GIF backend (gifpgn recommended on Windows)",
    )
    args = parser.parse_args(argv)

    print("Playing engine self-play...")
    game = play_engine_game(max_plies=args.max_plies)
    moves = max(0, game.end().ply() - game.ply())
    print(f"Game finished: {moves} moves, result {game.headers.get('Result', '*')}")

    renderer = PygameBoardRenderer(headless=args.headless)
    pygame_frames: list = []

    try:
        board = game.board()
        pygame_frames.append(renderer.show(board, caption="Start", delay_ms=200))
        last_move = None
        for node in game.mainline():
            if node.move is None:
                continue
            last_move = node.move
            board.push(last_move)
            pygame_frames.append(
                renderer.show(
                    board,
                    last_move=last_move,
                    caption=f"Move {board.fullmove_number}{'.' if board.turn == chess.BLACK else '...'} {last_move.uci()}",
                    delay_ms=120 if not args.headless else 0,
                )
            )
    finally:
        renderer.quit()

    frame_duration = args.frame_ms / 1000.0
    output = args.output.resolve()
    export_game_gif(
        game,
        output,
        exporter=args.exporter,
        frame_duration=frame_duration,
        board_size=480,
        pygame_frames=pygame_frames if args.exporter == "pygame" else None,
    )
    print(f"Saved GIF: {output} ({output.stat().st_size:,} bytes)")

    # Also dump PGN next to GIF for inspection.
    pgn_path = output.with_suffix(".pgn")
    with pgn_path.open("w", encoding="utf-8") as handle:
        print(game, file=handle)
    print(f"Saved PGN: {pgn_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())