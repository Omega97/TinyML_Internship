"""
Lc0 teacher evaluation.

Uses a persistent ``lc0`` UCI process. Default is ``go nodes 1`` (value head
only). ``go depth N`` is Lc0's MCTS depth stopper (average tree depth), not
Stockfish-style αβ depth. Maps WDL permille to White-POV expected reward.
"""

from __future__ import annotations

import atexit
import re
import subprocess
from pathlib import Path

import chess

from tinymlinternship.config.settings import LC0_BINARY, LC0_NETWORK_DEFAULT
from tinymlinternship.data.wdl import (
    permille_to_stm_wdl,
    terminal_stm_wdl,
    white_value_from_stm,
)

MATE_SCORE = 32_000

WDL_PERMILLE_RE = re.compile(r"\bwdl\s+(\d+)\s+(\d+)\s+(\d+)\b", re.IGNORECASE)
MATE_RE = re.compile(r"\bscore mate (-?\d+)\b", re.IGNORECASE)
CP_SCALE = 1000  # map expected reward in [-1, 1] to centipawn-like search scores


def parse_wdl_permille(line: str) -> tuple[int, int, int] | None:
    match = WDL_PERMILLE_RE.search(line)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def mate_to_wdl_permille(mate_ply: int) -> tuple[int, int, int]:
    """UCI ``score mate N`` is from side to move."""
    if mate_ply > 0:
        return 1000, 0, 0
    if mate_ply < 0:
        return 0, 0, 1000
    return 0, 1000, 0


def wdl_from_uci_lines(lines: list[str]) -> tuple[int, int, int] | None:
    """Last WDL on the search, else WDL synthesized from the last mate score."""
    last_wdl: tuple[int, int, int] | None = None
    last_mate: int | None = None
    for line in lines:
        parsed = parse_wdl_permille(line)
        if parsed is not None:
            last_wdl = parsed
        mate = MATE_RE.search(line)
        if mate is not None:
            last_mate = int(mate.group(1))
    if last_wdl is not None:
        return last_wdl
    if last_mate is not None:
        return mate_to_wdl_permille(last_mate)
    return None


def go_command(*, depth: int | None = None, nodes: int | None = None) -> str:
    """Build a UCI ``go`` line. ``nodes 1`` is the value-head-only baseline."""
    parts = ["go"]
    if depth is not None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        parts.extend(["depth", str(int(depth))])
    if nodes is not None:
        if nodes < 1:
            raise ValueError("nodes must be >= 1")
        parts.extend(["nodes", str(int(nodes))])
    if len(parts) == 1:
        parts.extend(["nodes", "1"])
    return " ".join(parts)


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
        go: str = "go nodes 1",
        smart_pruning_factor: float | None = None,
    ) -> None:
        self.binary = str(binary or LC0_BINARY)
        self.weights = str(weights or LC0_NETWORK_DEFAULT)
        self.backend = backend
        self.go = go
        self.smart_pruning_factor = smart_pruning_factor
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
        self._read_until(b"uciok", limit=2000)
        self._send("setoption name UCI_ShowWDL value true")
        if self.smart_pruning_factor is not None:
            self._send(
                f"setoption name SmartPruningFactor value {self.smart_pruning_factor}"
            )
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

    def evaluate_wdl(
        self, board: chess.Board, *, go: str | None = None
    ) -> tuple[int, int, int]:
        self.start()
        fen = board.fen()
        cmd = go or self.go
        self._send(f"position fen {fen}")
        self._send(cmd)
        lines = self._read_until(b"bestmove", limit=2000)
        parsed = wdl_from_uci_lines(lines)
        if parsed is not None:
            return parsed
        raise RuntimeError(f"lc0 returned no WDL for fen={fen!r} go={cmd!r}")

    def evaluate_stm_wdl(self, board: chess.Board, *, go: str | None = None) -> tuple[float, float, float]:
        """STM ``(W, D, L)`` probabilities (teacher, or a terminal assignment)."""
        terminal = terminal_stm_wdl(board)
        if terminal is not None:
            return terminal
        win, draw, loss = self.evaluate_wdl(board, go=go)
        return permille_to_stm_wdl(win, draw, loss)

    def evaluate_expected_reward(self, board: chess.Board) -> float:
        w, d, l = self.evaluate_stm_wdl(board)
        return white_value_from_stm(board, w, d, l)

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