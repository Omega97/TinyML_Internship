"""Batched Lc0 value-head WDL via ``lc0 fenbatch`` (keeps the GPU busy).

UCI ``go nodes 1`` evaluates one position per kernel launch. ``fenbatch`` feeds
a minibatch into Lc0's CUDA ``EvaluateBatch`` so the GB10 stays occupied.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tinymlinternship.config.settings import LC0_BINARY, LC0_NETWORK_DEFAULT
from tinymlinternship.data.wdl import normalize_wdl


def wdl_from_q_d(q: float, d: float) -> tuple[float, float, float]:
    """Value-head ``q = W-L``, ``d = D`` → STM ``(W, D, L)`` on the simplex."""
    return normalize_wdl(0.5 * (1.0 + q - d), d, 0.5 * (1.0 - q - d))


def parse_fenbatch_line(line: str) -> tuple[float, float, float]:
    """Parse one ``ok w d l`` result line from ``lc0 fenbatch``."""
    text = line.strip()
    if not text:
        raise RuntimeError("lc0 fenbatch returned an empty result line")
    if text.startswith("err "):
        raise RuntimeError(f"lc0 fenbatch: {text[4:]}")
    if not text.startswith("ok "):
        raise RuntimeError(f"lc0 fenbatch: unexpected line {text!r}")
    parts = text.split()
    if len(parts) < 4:
        raise RuntimeError(f"lc0 fenbatch: bad result {text!r}")
    return normalize_wdl(float(parts[1]), float(parts[2]), float(parts[3]))


class Lc0FenBatch:
    """Persistent ``lc0 fenbatch`` process: stdin FENs, stdout STM WDL."""

    def __init__(
        self,
        *,
        binary: str | None = None,
        weights: str | None = None,
        backend: str = "cuda-auto",
        batch_size: int = 256,
        backend_opts: str | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.binary = str(binary or LC0_BINARY)
        self.weights = str(weights or LC0_NETWORK_DEFAULT)
        self.backend = backend
        self.batch_size = int(batch_size)
        self.backend_opts = backend_opts
        self.extra_args = list(extra_args or [])
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        if not Path(self.binary).exists():
            raise FileNotFoundError(
                f"lc0 binary not found: {self.binary} — run scripts/download_teacher.py"
            )
        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"Lc0 weights not found: {self.weights} — run scripts/download_teacher.py"
            )
        cmd = [
            self.binary,
            "fenbatch",
            f"--weights={self.weights}",
            f"--backend={self.backend}",
            f"--batch-size={self.batch_size}",
            "--nncache=0",
        ]
        if self.backend_opts:
            cmd.append(f"--backend-opts={self.backend_opts}")
        cmd.extend(self.extra_args)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        ready = self._readline()
        if ready != "READY":
            self.close()
            raise RuntimeError(
                f"lc0 fenbatch did not become READY (got {ready!r}). "
                "Rebuild lc0 with the fenbatch mode, or check --backend."
            )

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and proc.poll() is None:
                try:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                except BrokenPipeError:
                    pass
            proc.wait(timeout=15)
        except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
            proc.kill()
            proc.wait(timeout=5)
        finally:
            self._proc = None

    def __enter__(self) -> Lc0FenBatch:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _readline(self) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        raw = self._proc.stdout.readline()
        if raw == "" and self._proc.poll() is not None:
            raise RuntimeError(
                f"lc0 fenbatch exited with code {self._proc.returncode}"
            )
        return raw.rstrip("\n\r")

    def _send(self, line: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")

    def evaluate_batch(self, fens: list[str]) -> list[tuple[float, float, float]]:
        """STM WDL for each FEN, one GPU (or CPU-backend) minibatch."""
        if not fens:
            return []
        self.start()
        for fen in fens:
            self._send(fen.replace("\n", " ").strip())
        n = len(fens)
        if n % self.batch_size != 0:
            self._send("EVAL")
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.flush()
        return [parse_fenbatch_line(self._readline()) for _ in range(n)]
