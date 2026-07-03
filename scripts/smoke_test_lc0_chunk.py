#!/usr/bin/env python3
"""Smoke test: parse one Lc0 training .gz chunk and validate positions."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.data.lc0_parser import V6_RECORD_SIZE, iter_positions, read_chunk
from tinymlinternship.features.bucket import bucket_id, piece_count


def default_chunk() -> Path:
    root = Path(__file__).parent.parent
    chunks = sorted(
        (root / "data/raw/lc0/chunks").rglob("*.gz"),
        key=lambda p: p.stat().st_size,
    )
    if not chunks:
        raise FileNotFoundError("No .gz chunks under data/raw/lc0/chunks/")
    return chunks[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test Lc0 chunk parser")
    parser.add_argument(
        "--chunk",
        type=Path,
        default=None,
        help="Path to one training .gz chunk (default: smallest chunk found)",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max positions to decode")
    args = parser.parse_args(argv)

    chunk = args.chunk or default_chunk()
    if not chunk.exists():
        print(f"Chunk not found: {chunk}", file=sys.stderr)
        return 1

    raw = read_chunk(chunk)
    record_count = len(raw) // V6_RECORD_SIZE
    print(f"Chunk:        {chunk}")
    print(f"Compressed:   {chunk.stat().st_size:,} bytes")
    print(f"Decompressed: {len(raw):,} bytes")
    print(f"Records:      {record_count} (V6 @ {V6_RECORD_SIZE} B)")
    print()

    valid = 0
    illegal = 0
    buckets: Counter[int] = Counter()
    samples: list[str] = []

    for idx, pos in enumerate(iter_positions(chunk, limit=args.limit)):
        try:
            board = chess.Board(pos.fen)
            ok = board.is_valid()
        except ValueError:
            ok = False

        if ok:
            valid += 1
            buckets[bucket_id(board)] += 1
            if len(samples) < 5:
                samples.append(
                    f"  [{idx}] fen={pos.fen!r}  pieces={piece_count(board)}  "
                    f"bucket={bucket_id(board)}  best_q={pos.best_q:+.3f}  "
                    f"plies_left={pos.plies_left:.0f}  visits={pos.visits}"
                )
        else:
            illegal += 1
            if illegal <= 3:
                print(f"  INVALID [{idx}] {pos.fen}")

    print(f"Decoded:  {valid + illegal} positions (limit={args.limit})")
    print(f"Valid:    {valid}")
    print(f"Invalid:  {illegal}")
    print(f"Buckets:  {dict(sorted(buckets.items()))}")
    print()
    print("Samples:")
    for line in samples:
        print(line)

    if valid == 0:
        print("\nSmoke test FAILED — no valid positions.", file=sys.stderr)
        return 1

    print("\nSmoke test OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())