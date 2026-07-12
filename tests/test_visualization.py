"""Tests for game playback and GIF export."""

from pathlib import Path

import chess.pgn

from datetime import date

from tinymlinternship.visualization.game_paths import (
    artifact_paths,
    engine_player_label,
    game_artifact_stem,
)
from tinymlinternship.visualization.gif_export import export_game_gif
from tinymlinternship.visualization.playback import play_engine_game


def test_play_engine_game_produces_moves(tmp_path: Path):
    game = play_engine_game(max_plies=10)
    assert game.end().ply() - game.ply() >= 1
    assert game.headers["Result"] in {"1-0", "0-1", "1/2-1/2", "*"}


def test_play_engine_game_max_seconds():
    game = play_engine_game(max_plies=200, depth=1, max_seconds=0.0)
    assert game.end().ply() - game.ply() == 0
    assert game.headers.get("Termination") == "time limit"


def test_game_artifact_stem():
    stem = game_artifact_stem("nnue-w128-844-d1", "nnue-w128-844-d1", on=date(2026, 7, 10))
    assert stem == "nnue_w128_844_d1_vs_nnue_w128_844_d1_2026-07-10"


def test_engine_player_label_nnue():
    label = engine_player_label(
        "nnue",
        depth=1,
        quiescence=False,
        nnue_checkpoint="models/checkpoints/nnue/pilot_W128_844/best.pt",
    )
    assert label == "nnue-w128-844-d1"


def test_artifact_paths(tmp_path: Path):
    pgn, gif = artifact_paths("hce-d1", "hce-d1", tmp_path, on=date(2026, 7, 10))
    assert pgn.name == "hce_d1_vs_hce_d1_2026-07-10.pgn"
    assert gif.name == "hce_d1_vs_hce_d1_2026-07-10.gif"


def test_export_game_gif_gifpgn(tmp_path: Path):
    game = play_engine_game(max_plies=6)
    out = tmp_path / "mini.gif"
    export_game_gif(game, out, exporter="gifpgn", frame_duration=0.2, board_size=320)
    assert out.exists()
    assert out.stat().st_size > 1_000