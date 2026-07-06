"""
Lc0 BT4 teacher evaluation for 1-ply search baseline.

Uses a persistent ``lc0`` UCI process (``go nodes 1``) and maps WDL permille to
centipawns from White's perspective for the existing negamax search stack.
"""

from __future__ import annotations

import atexit
import re
import subprocess
from pathlib import Path

import chess

from tinymlinternship.config.settings import LC0_BINARY, LC0_NETWORK_BT4, LC0_NETWORK_DEFAULT
from tinymlinternship.engine.eval_hce import MATE_SCORE

WDL_PERMILLE_RE = re.compile(r"\bwdl\s+(\d+)\s+(\d+)\s+(\d+)\b", re.IGNORECASE)
CP_SCALE = 1000  # map expected reward in [-1, 1] to centipawn-like search scores


def parse_wdl_permille(line: str) -> tuple[int, int, int] | None:
    match = WDL_PERMILLE_RE.search(line)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def wdl_to_expected_reward_white(board: chess.Board, win: int, draw: int, loss: int) -> float:
    """WDL permille (side to move) → expected reward from White's perspective."""
    _ = draw
    stm = (win - loss) / 1000.0
    return stm if board.turn == chess.WHITE else -stm


def expected_reward_to_cp(reward: float) -> int:
    return int(round(reward * CP_SCALE))


class Lc0Teacher:
    """Persistent Lc0 UCI session for position evaluation."""

    def __init__(
        self,
        *,
        binary: str | None = None,
        weights: str | None = None,
        backend: str = "blas",
    ) -> None:
        self.binary = str(binary or LC0_BINARY)
        self.weights = str(weights or LC0_NETWORK_DEFAULT)
        self.backend = backend
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        if not Path(self.binary).exists():
            raise FileNotFoundError(
                f"lc0 binary not found: {self.binary} — run scripts/download_teacher.py"
            )
        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"BT4 weights not found: {self.weights} — run scripts/download_teacher.py"
            )

        cmd = [self.binary, f"--weights={self.weights}", f"--backend={self.backend}"]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self._send("uci")
        self._read_until(b"uciok")
        self._send("setoption name UCI_ShowWDL value true")
        self._send("isready")
        self._read_until(b"readyok")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._send("quit")
            self._proc.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
            self._proc.kill()
        finally:
            self._proc = None

    def __enter__(self) -> Lc0Teacher:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _send(self, line: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((line + "\n").encode())
        self._proc.stdin.flush()

    def _read_until(self, token: bytes, *, limit: int = 500) -> list[str]:
        assert self._proc is not None and self._proc.stdout is not None
        lines: list[str] = []
        while len(lines) < limit:
            raw = self._proc.stdout.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                lines.append(text)
            if token in raw:
                break
        return lines

    def evaluate_wdl(self, board: chess.Board) -> tuple[int, int, int]:
        self.start()
        fen = board.fen()
        self._send(f"position fen {fen}")
        self._send("go nodes 1")
        lines = self._read_until(b"bestmove")
        for line in reversed(lines):
            parsed = parse_wdl_permille(line)
            if parsed is not None:
                return parsed
        raise RuntimeError(f"lc0 returned no WDL for fen={fen!r}")

    def evaluate_expected_reward(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0 if board.turn == chess.WHITE else 1.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            return 0.0

        win, draw, loss = self.evaluate_wdl(board)
        return wdl_to_expected_reward_white(board, win, draw, loss)

    def evaluate_cp(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        return expected_reward_to_cp(self.evaluate_expected_reward(board))


_teacher_singleton: Lc0Teacher | None = None


def get_lc0_teacher() -> Lc0Teacher:
    global _teacher_singleton
    if _teacher_singleton is None:
        _teacher_singleton = Lc0Teacher()
        _teacher_singleton.start()
        atexit.register(_close_teacher_singleton)
    return _teacher_singleton


def _close_teacher_singleton() -> None:
    global _teacher_singleton
    if _teacher_singleton is not None:
        _teacher_singleton.close()
        _teacher_singleton = None


def evaluate_lc0_teacher(board: chess.Board) -> int:
    """Static eval in centipawn-like units (White = positive) via Lc0 BT4 WDL."""
    return get_lc0_teacher().evaluate_cp(board)