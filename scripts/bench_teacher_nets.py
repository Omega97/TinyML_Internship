#!/usr/bin/env python3
"""Benchmark lc0 networks: single eval + 1-ply search on startpos."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from tinymlinternship.config.settings import LC0_BINARY, LC0_NETWORK_BT4, TEACHER_DIR
from tinymlinternship.engine import search
from tinymlinternship.engine.eval_lc0 import Lc0Teacher

T1_256 = TEACHER_DIR / "networks" / "t1-256x10-distilled-swa-2432500.pb.gz"
BUNDLED = TEACHER_DIR / "lc0" / "791556.pb.gz"
T1_256_URL = (
    "https://storage.lczero.org/files/networks-contrib/"
    "t1-256x10-distilled-swa-2432500.pb.gz"
)


def download_t1_256() -> None:
    if T1_256.exists():
        return
    from urllib.request import Request, urlopen

    T1_256.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {T1_256.name} …")
    req = Request(T1_256_URL, headers={"User-Agent": "SARDINE-bench/0.1"})
    with urlopen(req, timeout=120) as resp, T1_256.open("wb") as out:
        while chunk := resp.read(1024 * 1024):
            out.write(chunk)
    print(f"  saved {T1_256.stat().st_size // (1024*1024)} MiB")


def bench_net(label: str, weights: Path) -> None:
    teacher = Lc0Teacher(weights=str(weights))
    teacher.start()
    board = chess.Board()

    # warm-up
    teacher.evaluate_cp(board)

    t0 = time.perf_counter()
    for _ in range(5):
        teacher.evaluate_cp(board)
    eval_ms = (time.perf_counter() - t0) / 5 * 1000

    def eval_fn(b: chess.Board) -> int:
        return teacher.evaluate_cp(b)

    t0 = time.perf_counter()
    r = search(board, 1, eval_fn=eval_fn, quiescence=False)
    move_s = time.perf_counter() - t0

    mb = weights.stat().st_size / (1024 * 1024)
    print(
        f"{label:12} {mb:5.1f} MB  eval={eval_ms:6.0f} ms  "
        f"1-ply={move_s:5.2f} s  move={r.move.uci() if r else '?'}  nodes={r.nodes if r else 0}"
    )
    teacher.close()


def main() -> None:
    download_t1_256()
    nets = [
        ("791556", BUNDLED),
        ("T1-256", T1_256),
        ("BT4", LC0_NETWORK_BT4),
    ]
    print(f"lc0: {LC0_BINARY}\n")
    for label, path in nets:
        if not path.exists():
            print(f"{label}: missing {path}")
            continue
        bench_net(label, path)


if __name__ == "__main__":
    main()