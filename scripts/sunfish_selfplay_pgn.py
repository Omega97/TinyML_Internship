#!/usr/bin/env python3
"""Sunfish depth-1 self-play → PGN for ACPL gate calibration (blueprint B1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.pgn

PROJECT_ROOT = Path(__file__).parent.parent
SUNFISH_DIR = PROJECT_ROOT / "models" / "teacher" / "sunfish"


def _load_sunfish():
    """Load sunfish core without starting the UCI REPL."""
    text = (SUNFISH_DIR / "sunfish.py").read_text(encoding="utf-8")
    cut = text.find("hist = [Position(initial")
    if cut < 0:
        raise RuntimeError("Unexpected sunfish.py layout — cannot strip UCI launcher")
    text = text[:cut]
    ns: dict = {"__name__": "sunfish_core"}
    exec(compile(text, str(SUNFISH_DIR / "sunfish.py"), "exec"), ns)
    return ns


def _sunfish_move_to_uci(move, *, white_to_move: bool) -> str:
    render = move["render"]
    i, j = move["i"], move["j"]
    if not white_to_move:
        i, j = 119 - i, 119 - j
    prom = move["prom"].lower() if move["prom"] else ""
    return render(i) + render(j) + prom


def _pick_move(searcher, position_cls, history: list, *, depth: int):
    pos = history[-1]
    move = None
    for d, _gamma, _score, cand in searcher.search(history):
        if d >= depth and cand is not None:
            move = cand
            break
    if move is None:
        moves = list(pos.gen_moves())
        if not moves:
            return None
        move = moves[0]
    return move


def play_sunfish_game(
    *,
    max_plies: int,
    depth: int,
    sunfish_ns: dict,
) -> chess.pgn.Game:
    Position = sunfish_ns["Position"]
    Searcher = sunfish_ns["Searcher"]
    initial = sunfish_ns["initial"]

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Sunfish gate"
    game.headers["Site"] = "SARDINE calibration"
    game.headers["White"] = "Sunfish"
    game.headers["Black"] = "Sunfish"
    game.headers["Annotator"] = f"sunfish depth {depth}"

    hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]
    node = game

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        searcher = Searcher()
        move = _pick_move(searcher, Position, hist, depth=depth)
        if move is None:
            break
        uci = _sunfish_move_to_uci(
            {"i": move.i, "j": move.j, "prom": move.prom, "render": sunfish_ns["render"]},
            white_to_move=board.turn == chess.WHITE,
        )
        chess_move = chess.Move.from_uci(uci)
        if chess_move not in board.legal_moves:
            raise RuntimeError(f"Sunfish illegal move at ply {ply + 1}: {uci}")
        board.push(chess_move)
        node = node.add_variation(chess_move)
        hist.append(hist[-1].move(move))

    game.headers["Result"] = board.result(claim_draw=True)
    return game


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sunfish self-play PGN for ACPL gate")
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "plots" / "sunfish_d1_gate.pgn",
    )
    args = parser.parse_args(argv)

    if not (SUNFISH_DIR / "sunfish.py").is_file():
        raise SystemExit(
            f"Sunfish not found at {SUNFISH_DIR}. "
            "Clone: git clone https://github.com/thomasahle/sunfish.git models/teacher/sunfish"
        )

    sunfish_ns = _load_sunfish()
    game = play_sunfish_game(
        max_plies=args.max_plies,
        depth=args.depth,
        sunfish_ns=sunfish_ns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")
    plies = max(0, game.end().ply() - game.ply())
    print(f"Sunfish self-play: {plies} half-moves → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())