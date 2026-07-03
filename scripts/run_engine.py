#!/usr/bin/env python3
"""Run SARDINE engine v0.1 (HCE + 1-ply search) from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.engine import ENGINE_VERSION, search_best_move


def _parse_moves(move_str: str) -> list[chess.Move]:
    moves: list[chess.Move] = []
    for token in move_str.split():
        moves.append(chess.Move.from_uci(token))
    return moves


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SARDINE engine v0.1 — HCE + 1-ply search")
    parser.add_argument(
        "--fen",
        default=chess.STARTING_FEN,
        help="FEN of the position (default: startpos)",
    )
    parser.add_argument(
        "--moves",
        default="",
        help="Space-separated UCI moves to apply before searching",
    )
    parser.add_argument("--version", action="store_true", help="Print engine version and exit")
    args = parser.parse_args(argv)

    if args.version:
        print(f"SARDINE {ENGINE_VERSION}")
        return 0

    board = chess.Board(args.fen)
    if args.moves.strip():
        for move in _parse_moves(args.moves.strip()):
            board.push(move)

    result = search_best_move(board)
    if result is None:
        print("nomove")
        return 1

    side = "White" if board.turn == chess.WHITE else "Black"
    print(f"bestmove {result.move.uci()}")
    print(f"info side {side} score cp {result.score} nodes {result.nodes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())