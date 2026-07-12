#!/usr/bin/env python3
"""Play a SARDINE engine game, display it with pygame, and save a GIF."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.pgn

from tinymlinternship.config.settings import NNUE_CHECKPOINT_DEFAULT
from tinymlinternship.engine import EVAL_CHOICES, ENGINE_VERSION, make_eval_fn
from tinymlinternship.visualization import (
    PygameBoardRenderer,
    export_game_gif,
    play_engine_game,
)


class _TerminalProgress:
    """Single-line progress bar (no extra dependencies)."""

    def __init__(self, total: int, label: str, *, width: int = 32) -> None:
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self._started = time.perf_counter()
        self._last_extra = ""

    def update(self, current: int, extra: str = "") -> None:
        ratio = min(current / self.total, 1.0)
        filled = int(self.width * ratio)
        if filled >= self.width:
            bar = "=" * self.width
        else:
            bar = "=" * filled + ">" + " " * (self.width - filled - 1)
        elapsed = time.perf_counter() - self._started
        if extra:
            self._last_extra = extra
        suffix = self._last_extra
        line = f"\r{self.label} [{bar}] {current}/{self.total}  {elapsed:5.1f}s"
        if suffix:
            line += f"  {suffix}"
        sys.stdout.write(line[:120].ljust(120))
        sys.stdout.flush()

    def finish(self, message: str = "") -> None:
        elapsed = time.perf_counter() - self._started
        sys.stdout.write(f"\r{self.label} done in {elapsed:.1f}s")
        if message:
            sys.stdout.write(f" — {message}")
        sys.stdout.write("\n")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a SARDINE engine game as GIF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "images" / "sardine_game.gif",
        help="Output GIF path (default: images/sardine_game.gif)",
    )
    parser.add_argument("--max-plies", type=int, default=200, help="Max half-moves")
    parser.add_argument("--depth", type=int, default=1, help="Search depth in full moves")
    parser.add_argument(
        "--eval",
        choices=EVAL_CHOICES,
        default="hce",
        help="Static eval backend (default: hce)",
    )
    parser.add_argument(
        "--nnue-checkpoint",
        type=Path,
        default=NNUE_CHECKPOINT_DEFAULT,
        help="NNUE checkpoint path (--eval nnue)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No pygame window (still builds frames for optional pygame GIF)",
    )
    parser.add_argument(
        "--quiescence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture quiescence at leaves (default: on)",
    )
    parser.add_argument(
        "--max-qsearch-depth",
        type=int,
        default=None,
        metavar="PLIES",
        help="Cap quiescence depth (default: unlimited; try 6 with --depth 2)",
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars",
    )
    args = parser.parse_args(argv)

    eval_fn = make_eval_fn(
        args.eval,
        nnue_checkpoint=args.nnue_checkpoint if args.eval == "nnue" else None,
    )
    annotator = f"SARDINE {ENGINE_VERSION} ({args.eval}, depth {args.depth})"
    qsearch = "on" if args.quiescence else "off"
    if args.quiescence and args.max_qsearch_depth is not None:
        qsearch = f"on,max{args.max_qsearch_depth}"
    print(f"Playing engine self-play ({annotator}, qsearch={qsearch})...")

    play_progress = None if args.no_progress else _TerminalProgress(args.max_plies, "Self-play")

    def _on_ply(ply: int, max_plies: int, move: chess.Move, ply_sec: float) -> None:
        if play_progress is not None:
            play_progress.update(ply, f"ply {ply} {move.uci()} {ply_sec:.1f}s")

    game = play_engine_game(
        max_plies=args.max_plies,
        depth=args.depth,
        eval_fn=eval_fn,
        quiescence=args.quiescence,
        max_qsearch_depth=args.max_qsearch_depth,
        annotator=annotator,
        on_ply=None if args.no_progress else _on_ply,
    )
    moves = max(0, game.end().ply() - game.ply())
    if play_progress is not None:
        play_progress.finish(f"{moves} plies, result {game.headers.get('Result', '*')}")
    else:
        print(f"Game finished: {moves} moves, result {game.headers.get('Result', '*')}")

    renderer = PygameBoardRenderer(headless=args.headless)
    pygame_frames: list = []

    mainline = list(game.mainline())
    frame_total = sum(1 for node in mainline if node.move is not None) + 1
    frame_progress = None if args.no_progress else _TerminalProgress(frame_total, "Frames")

    try:
        board = game.board()
        pygame_frames.append(renderer.show(board, caption="Start", delay_ms=200))
        if frame_progress is not None:
            frame_progress.update(1)
        last_move = None
        frame_idx = 1
        for node in mainline:
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
            frame_idx += 1
            if frame_progress is not None:
                frame_progress.update(frame_idx, last_move.uci())
    finally:
        renderer.quit()

    if frame_progress is not None:
        frame_progress.finish()

    frame_duration = args.frame_ms / 1000.0
    output = args.output.resolve()
    if not args.no_progress:
        print("Exporting GIF...", flush=True)
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