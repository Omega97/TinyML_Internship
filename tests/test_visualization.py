"""Tests for game playback and GIF export."""

from pathlib import Path

import chess.pgn

from tinymlinternship.visualization.gif_export import export_game_gif
from tinymlinternship.visualization.playback import play_engine_game


def test_play_engine_game_produces_moves(tmp_path: Path):
    game = play_engine_game(max_plies=10)
    assert game.end().ply() - game.ply() >= 1
    assert game.headers["Result"] in {"1-0", "0-1", "1/2-1/2", "*"}


def test_export_game_gif_gifpgn(tmp_path: Path):
    game = play_engine_game(max_plies=6)
    out = tmp_path / "mini.gif"
    export_game_gif(game, out, exporter="gifpgn", frame_duration=0.2, board_size=320)
    assert out.exists()
    assert out.stat().st_size > 1_000