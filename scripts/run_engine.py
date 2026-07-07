#!/usr/bin/env python3
"""Run SARDINE engine (HCE + alpha-beta search) from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import NNUE_CHECKPOINT_DEFAULT
from tinymlinternship.engine import ENGINE_VERSION, EVAL_CHOICES, make_eval_fn, search


def _parse_moves(move_str: str) -> list[chess.Move]:
    moves: list[chess.Move] = []
    for token in move_str.split():
        moves.append(chess.Move.from_uci(token))
    return moves


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SARDINE engine — alpha-beta search")
    parser.add_argument(
        "--eval",
        choices=EVAL_CHOICES,
        default="hce",
        help="Static eval backend (default: hce)",
    )
    parser.add_argument(
        "--nnue-checkpoint",
        type=Path,
        default=NNUE_CHECKPOINT_DEFAULT,
        help="NNUE checkpoint path (--eval nnue)",
    )
    parser.add_argument("--depth", type=int, default=1, help="Search depth in full moves (default: 1)")
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

    eval_fn = make_eval_fn(
        args.eval,
        nnue_checkpoint=args.nnue_checkpoint if args.eval == "nnue" else None,
    )
    result = search(board, args.depth, eval_fn=eval_fn)
    if result is None:
        print("nomove")
        return 1

    side = "White" if board.turn == chess.WHITE else "Black"
    print(f"bestmove {result.move.uci()}")
    print(
        f"info side {side} depth {result.depth} score cp {result.score} nodes {result.nodes}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())