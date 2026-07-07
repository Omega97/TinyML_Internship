"""Locate Stockfish binary for bot evaluation scripts."""

from __future__ import annotations

import os
from pathlib import Path

from tinymlinternship.config.settings import PROJECT_ROOT


def resolve_stockfish(path: str | None = None) -> str:
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Stockfish binary not found: {p}")
        return str(p.resolve())
    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    for candidate in (
        PROJECT_ROOT / "models" / "teacher" / "stockfish" / "stockfish.exe",
        Path(r"C:\Program Files\Stockfish\stockfish.exe"),
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        "Stockfish not found — pass --stockfish PATH, set STOCKFISH_PATH, "
        "or install to models/teacher/stockfish/stockfish.exe"
    )