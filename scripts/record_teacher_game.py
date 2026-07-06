#!/usr/bin/env python3
"""Self-play with Lc0 teacher eval (1-ply) and export GIF to images/."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.pgn

from tinymlinternship.config.settings import LC0_NETWORK_PRESETS
from tinymlinternship.engine import ENGINE_VERSION, search
from tinymlinternship.engine.eval_lc0 import Lc0Teacher
from tinymlinternship.visualization import PygameBoardRenderer, export_game_gif


class LiveMovePrinter:
    """Print moves as the game progresses: ``1. e4 e5`` then ``2. Nc3`` …"""

    def print_move(self, board: chess.Board, move: chess.Move) -> None:
        san = board.san(move)
        if board.turn == chess.WHITE:
            print(f"{board.fullmove_number}. {san}", end="", flush=True)
        else:
            print(f" {san}", flush=True)
        board.push(move)

    def finish(self, board: chess.Board) -> None:
        if board.turn == chess.BLACK:
            print(flush=True)


def resolve_network(name: str) -> Path:
    key = name.lower()
    if key in LC0_NETWORK_PRESETS:
        path = LC0_NETWORK_PRESETS[key]
    else:
        path = Path(name)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / path
    if not path.exists():
        raise SystemExit(f"Network not found: {path}")
    return path.resolve()


def play_teacher_game(
    teacher: Lc0Teacher,
    *,
    max_plies: int,
    depth: int,
    white_name: str,
    black_name: str,
    network_label: str,
    live: LiveMovePrinter,
) -> chess.pgn.Game:
    def eval_fn(board: chess.Board) -> int:
        return teacher.evaluate_cp(board)

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "SARDINE teacher self-play"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Annotator"] = (
        f"SARDINE {ENGINE_VERSION} (Lc0 {network_label}, {depth}-ply, no qsearch)"
    )

    node = game
    plies = 0

    while not board.is_game_over() and plies < max_plies:
        result = search(board, depth, eval_fn=eval_fn, quiescence=True)
        if result is None:
            break
        live.print_move(board, result.move)
        node = node.add_variation(result.move)
        plies += 1

    live.finish(board)

    outcome = board.outcome()
    game.headers["Result"] = outcome.result() if outcome else "*"
    return game


def main(argv: list[str] | None = None) -> int:
    presets = ", ".join(sorted(LC0_NETWORK_PRESETS))
    parser = argparse.ArgumentParser(description="Record Lc0 teacher 1-ply self-play GIF")
    parser.add_argument(
        "--network",
        default="fast",
        help=f"Preset ({presets}) or path to .pb.gz (default: fast = 791556)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,  # Will be auto-generated if not provided
        help="Output GIF path (optional; auto-generated from model + timestamp if omitted)",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=400,
        help="Max half-moves (default 400)",
    )
    parser.add_argument("--depth", type=int, default=1, help="Search depth (default 1)")
    parser.add_argument("--headless", action="store_true", help="No pygame window")
    parser.add_argument("--frame-ms", type=int, default=500, help="GIF frame duration")
    parser.add_argument(
        "--exporter",
        choices=("gifpgn", "chess_gif", "pygame"),
        default="gifpgn",
    )
    args = parser.parse_args(argv)

    network_path = resolve_network(args.network)
    network_label = args.network if args.network.lower() in LC0_NETWORK_PRESETS else network_path.name

    # --- Auto-generate output path if not specified ---
    if args.output is None:
        # Sanitize the model label so it's safe for filenames
        safe_label = network_label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"teacher_{safe_label}_depth{args.depth}_{timestamp}.gif"
        args.output = Path(__file__).parent.parent / "images" / "games" / filename

    print(f"Network: {network_path.name} ({network_path.stat().st_size // (1024 * 1024)} MiB)")
    teacher = Lc0Teacher(weights=str(network_path))
    teacher.start()

    live = LiveMovePrinter()
    print(f"Playing {network_label} {args.depth}-ply self-play (max {args.max_plies} plies) …")
    print()

    try:
        game = play_teacher_game(
            teacher,
            max_plies=args.max_plies,
            depth=args.depth,
            white_name=f"Lc0-{network_label}",
            black_name=f"Lc0-{network_label}",
            network_label=network_label,
            live=live,
        )
    finally:
        teacher.close()

    print()
    moves = max(0, game.end().ply() - game.ply())
    print(f"Game finished: {moves} plies, result {game.headers.get('Result', '*')}")

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
                    caption=(
                        f"{board.fullmove_number}{'.' if board.turn == chess.BLACK else '...'} "
                        f"{last_move.uci()}"
                    ),
                    delay_ms=0,
                )
            )
    finally:
        renderer.quit()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    export_game_gif(
        game,
        output,
        exporter=args.exporter,
        frame_duration=args.frame_ms / 1000.0,
        board_size=480,
        pygame_frames=pygame_frames if args.exporter == "pygame" else None,
    )
    print(f"Saved GIF: {output} ({output.stat().st_size:,} bytes)")

    pgn_path = output.with_suffix(".pgn")
    with pgn_path.open("w", encoding="utf-8") as handle:
        print(game, file=handle)
    print(f"Saved PGN: {pgn_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
