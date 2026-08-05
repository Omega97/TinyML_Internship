"""Minimal Cfish UCI smoke via python-chess.

Resolves ``src/cfish/cfish.exe`` from the repo root (or this script's parent)
and sets cwd to ``src/cfish`` so ``EvalFile`` finds the stock NNUE next to the
binary. Run from repo root::

    py -3.12 scripts/cfish.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.engine

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CFISH_DIR = _REPO_ROOT / "src" / "cfish"
_CFISH_EXE = _CFISH_DIR / ("cfish.exe" if sys.platform == "win32" else "cfish")


def main() -> None:
    if not _CFISH_EXE.is_file():
        raise SystemExit(f"Cfish binary not found: {_CFISH_EXE}")

    # cwd must be src/cfish so DefaultEvalFile (nn-….nnue) resolves.
    engine = chess.engine.SimpleEngine.popen_uci(
        str(_CFISH_EXE),
        cwd=str(_CFISH_DIR),
    )
    try:
        # Large pages often fail on Windows; avoid noisy TT allocation errors.
        try:
            engine.configure({"LargePages": False})
        except chess.engine.EngineError:
            pass

        board = chess.Board()
        result = engine.play(board, chess.engine.Limit(depth=5))
        print(f"Best Move: {result.move}")

        info = engine.analyse(board, chess.engine.Limit(depth=5))
        print(f"Score: {info['score'].relative}")
        if "nps" in info:
            print(f"nps: {info['nps']}")
        if "nodes" in info:
            print(f"nodes: {info['nodes']}")
        if "depth" in info:
            print(f"depth: {info['depth']}")
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
