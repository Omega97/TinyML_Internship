"""Locate Stockfish binary for bot evaluation scripts (PATH / env only)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_stockfish(path: str | None = None) -> str:
    """
    Resolve Stockfish executable.

    Order: explicit ``path`` → ``STOCKFISH_PATH`` env → ``stockfish`` / ``stockfish.exe`` on PATH
    → common Windows install dir. No in-repo binary is shipped.
    """
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Stockfish binary not found: {p}")
        return str(p.resolve())

    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).is_file():
        return str(Path(env).resolve())

    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found:
            return str(Path(found).resolve())

    for candidate in (
        Path(r"C:\Program Files\Stockfish\stockfish.exe"),
        Path(r"C:\Program Files (x86)\Stockfish\stockfish.exe"),
    ):
        if candidate.is_file():
            return str(candidate.resolve())

    raise FileNotFoundError(
        "Stockfish not found — pass --stockfish PATH, set STOCKFISH_PATH, "
        "or install Stockfish on PATH (https://stockfishchess.org/download/)."
    )
