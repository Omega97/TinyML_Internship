"""Naming and export helpers for self-play PGN + GIF artifacts."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import chess.pgn

from tinymlinternship.visualization.gif_export import Exporter, export_game_gif


def slug_name(name: str) -> str:
    """Filesystem-safe player slug (lowercase, underscores)."""
    s = name.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-+", "-", s).strip("-")
    return s.replace("-", "_")


def game_artifact_stem(white: str, black: str, *, on: date | None = None) -> str:
    """``[bianco]_vs_[nero]_[data]`` without extension."""
    day = on or date.today()
    return f"{slug_name(white)}_vs_{slug_name(black)}_{day.isoformat()}"


def engine_player_label(
    eval_backend: str,
    *,
    depth: int,
    quiescence: bool,
    nnue_checkpoint: Path | str | None = None,
) -> str:
    """Human-readable player name for SARDINE self-play headers and filenames."""
    if eval_backend == "hce":
        if depth == 1 and not quiescence:
            return "hce-d1"
        q = "qsearch" if quiescence else "no-qsearch"
        return f"hce-d{depth}-{q}"
    if eval_backend == "nnue":
        stem = "w128-844"
        if nnue_checkpoint is not None:
            run = Path(nnue_checkpoint).parent.name.lower().replace("_", "-")
            if run.startswith("pilot-"):
                run = run[len("pilot-") :]
            stem = run
        return f"nnue-{stem}-d{depth}"
    if eval_backend == "lc0":
        return f"lc0-d{depth}"
    return slug_name(eval_backend)


def artifact_paths(
    white: str,
    black: str,
    output_dir: Path | str,
    *,
    on: date | None = None,
) -> tuple[Path, Path]:
    """Return ``(pgn_path, gif_path)`` under ``output_dir``."""
    stem = game_artifact_stem(white, black, on=on)
    base = Path(output_dir) / stem
    return base.with_suffix(".pgn"), base.with_suffix(".gif")


def write_game_pgn(game: chess.pgn.Game, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        print(game, file=handle, end="\n\n")
    return out


def write_game_gif(
    game: chess.pgn.Game,
    path: Path | str,
    *,
    frame_ms: int = 450,
    exporter: Exporter = "gifpgn",
    board_size: int = 480,
) -> Path:
    out = Path(path)
    export_game_gif(
        game,
        out,
        exporter=exporter,
        frame_duration=frame_ms / 1000.0,
        board_size=board_size,
    )
    return out