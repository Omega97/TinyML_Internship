#!/usr/bin/env python3
"""Inspect ChessBench .bag records (Research format) and summarize value semantics."""

from __future__ import annotations

import argparse
import math
import mmap
import os
import struct
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from apache_beam import coders

from tinymlinternship.config.settings import CHESSBENCH_RAW_DIR

STATE_CODER = coders.TupleCoder((
    coders.StrUtf8Coder(),
    coders.FloatCoder(),
))
ACTION_CODER = coders.TupleCoder((
    coders.StrUtf8Coder(),
    coders.StrUtf8Coder(),
    coders.FloatCoder(),
))


class BagFileReader:
    def __init__(self, filename: str) -> None:
        fd = os.open(filename, os.O_RDONLY)
        try:
            self._records = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            file_size = self._records.size()
        except ValueError:
            self._records = b""
            file_size = 0
        finally:
            os.close(fd)
        if 0 < file_size < 8:
            raise ValueError("bag file too small")
        self._limits = self._records
        if file_size:
            (index_start,) = struct.unpack("<Q", self._records[-8:])
        else:
            index_start = 0
        index_size = file_size - index_start
        self._num_records = index_size // 8
        self._limits_start = index_start

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, index: int) -> bytes:
        if not 0 <= index < self._num_records:
            raise IndexError("bag index out of range")
        end = index * 8 + self._limits_start
        if index:
            rec_range = struct.unpack("<2q", self._limits[end - 8 : end + 8])
        else:
            rec_range = (0, *struct.unpack("<q", self._limits[end : end + 8]))
        return self._records[slice(*rec_range)]


def win_prob_to_cp(win_prob: float) -> float:
    """Inverse of Lichess/DeepMind formula (centipawns, side to move)."""
    win_prob = min(max(win_prob, 1e-9), 1.0 - 1e-9)
    return math.log(win_prob / (1.0 - win_prob)) / 0.00368208


def study_bag(path: Path, *, sample: int, kind: str) -> None:
    reader = BagFileReader(str(path))
    coder = STATE_CODER if kind == "state_value" else ACTION_CODER
    n = len(reader)
    print(f"\n=== {path.name} ({kind}) ===")
    print(f"records: {n:,}")

    values: list[float] = []
    step = max(1, n // sample)
    indices = list(range(0, n, step))[:sample]

    for i in indices:
        record = coder.decode(reader[i])
        values.append(record[-1])
        if len(values) <= 3:
            board = chess.Board(record[0])
            stm = "W" if board.turn == chess.WHITE else "B"
            print(f"  sample: {record}")
            print(f"    win_prob={record[-1]:.6f} · cp(STM)≈{win_prob_to_cp(record[-1]):+.0f} · STM={stm}")

    print(
        f"win_prob: min={min(values):.6f} max={max(values):.6f} "
        f"mean={statistics.mean(values):.6f} median={statistics.median(values):.6f}"
    )
    print(f"at 0.0: {sum(1 for v in values if v == 0.0)} · at 1.0: {sum(1 for v in values if v == 1.0)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Study ChessBench value semantics")
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args(argv)

    action_path = CHESSBENCH_RAW_DIR / "test" / "action_value_data.bag"
    state_path = CHESSBENCH_RAW_DIR / "test" / "state_value_data.bag"

    if not state_path.exists() and not action_path.exists():
        print("No ChessBench files — run: py -3.12 scripts/download_chessbench.py")
        return 1

    print("ChessBench value type (DeepMind searchless_chess, paper §2.1):")
    print("  field: win_prob (float64)")
    print("  type: WIN PROBABILITY in [0, 1] — NOT centipawns, NOT WDL (W−L)")
    print("  engine: Stockfish 16, 50 ms/position (state) or /move (action)")
    print("  cp → win_prob: 1 / (1 + exp(-0.00368208 * cp)); mate → 1.0")
    print("  POV: score.relative → side to move")
    print("  SARDINE approx map: expected_reward ≈ 2*win_prob - 1")

    if state_path.exists():
        study_bag(state_path, sample=args.sample, kind="state_value")
    if action_path.exists():
        study_bag(action_path, sample=args.sample, kind="action_value")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())