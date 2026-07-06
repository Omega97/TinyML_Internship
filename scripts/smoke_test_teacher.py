#!/usr/bin/env python3
"""Smoke test: Lc0 BT4 teacher returns WDL on startpos."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import LC0_BINARY, LC0_NETWORK_BT4

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def read_until(proc: subprocess.Popen[bytes], token: bytes, limit: int = 200) -> list[str]:
    lines: list[str] = []
    while len(lines) < limit:
        raw = proc.stdout.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            lines.append(line)
        if token in raw:
            break
    return lines


def main() -> None:
    if not LC0_BINARY.exists():
        raise SystemExit(f"lc0 binary missing — run: py -3.12 scripts/download_teacher.py\n  {LC0_BINARY}")
    if not LC0_NETWORK_BT4.exists():
        raise SystemExit(f"BT4 network missing — run: py -3.12 scripts/download_teacher.py\n  {LC0_NETWORK_BT4}")

    cmd = [
        str(LC0_BINARY),
        f"--weights={LC0_NETWORK_BT4}",
        "--backend=blas",
    ]
    print("cmd:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert proc.stdin and proc.stdout

    def send(line: str) -> None:
        proc.stdin.write((line + "\n").encode())
        proc.stdin.flush()

    send("uci")
    read_until(proc, b"uciok")
    send("setoption name UCI_ShowWDL value true")
    send("isready")
    read_until(proc, b"readyok")
    send(f"position fen {STARTPOS}")
    send("go nodes 1")
    lines = read_until(proc, b"bestmove", limit=50)
    send("quit")
    proc.wait(timeout=30)

    wdl_line = next(
        (ln for ln in lines if "info" in ln and (" wdl " in ln.lower() or "score wdl" in ln.lower())),
        None,
    )
    print("--- UCI output (tail) ---")
    for ln in lines[-8:]:
        print(ln)
    if wdl_line is None:
        raise SystemExit("FAIL: no WDL in search output")
    print("OK: teacher responds with WDL")


if __name__ == "__main__":
    main()