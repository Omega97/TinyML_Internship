"""Export a ``chess.pgn.Game`` to GIF (gifpgn / chess_gif)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import chess.pgn
import imageio.v2 as imageio
import numpy as np
import pygame

Exporter = Literal["gifpgn", "chess_gif", "pygame"]


def export_game_gif(
    game: chess.pgn.Game,
    output_path: str | Path,
    *,
    exporter: Exporter = "gifpgn",
    frame_duration: float = 0.45,
    board_size: int = 480,
    pygame_frames: Iterable[pygame.Surface] | None = None,
) -> Path:
    """
    Write ``game`` to an animated GIF at ``output_path``.

    Default backend is **gifpgn** (works on Windows without libvips).
    **chess_gif** is attempted when requested but needs libvips + gifsicle.
    **pygame** encodes pre-rendered pygame surfaces via imageio.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if exporter == "pygame":
        if pygame_frames is None:
            raise ValueError("pygame exporter requires pygame_frames")
        _export_pygame_frames(pygame_frames, path, frame_duration)
        return path

    if exporter == "chess_gif":
        try:
            _export_chess_gif(game, path, frame_duration, board_size)
            return path
        except (ImportError, OSError) as exc:
            # libvips/gifsicle often missing on Windows — fall back.
            _export_gifpgn(game, path, frame_duration, board_size)
            return path

    _export_gifpgn(game, path, frame_duration, board_size)
    return path


def _export_gifpgn(
    game: chess.pgn.Game,
    path: Path,
    frame_duration: float,
    board_size: int,
) -> None:
    from gifpgn import CreateGifFromPGN

    if game.end().ply() - game.ply() < 1:
        raise ValueError("Cannot export GIF: game has no moves")

    maker = CreateGifFromPGN(game)
    maker.board_size = board_size
    maker.frame_duration = frame_duration
    maker.generate(str(path))


def _export_chess_gif(
    game: chess.pgn.Game,
    path: Path,
    frame_duration: float,
    board_size: int,
) -> None:
    import io

    from chess_gif.gif_maker import GIFMaker

    buffer = io.StringIO()
    print(game, file=buffer)
    maker = GIFMaker(side=max(32, board_size // 8), delay=int(frame_duration * 1000))
    maker.make_gif_from_pgn_string(buffer.getvalue(), str(path))


def _export_pygame_frames(
    frames: Iterable[pygame.Surface],
    path: Path,
    frame_duration: float,
) -> None:
    arrays = []
    for frame in frames:
        rgb = pygame.surfarray.array3d(frame).swapaxes(0, 1)
        arrays.append(rgb)

    if not arrays:
        raise ValueError("No frames to encode")

    imageio.mimsave(path, arrays, duration=frame_duration, loop=0)