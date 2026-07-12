#!/usr/bin/env python3
"""Sunfish depth-1 self-play → PGN for ACPL gate calibration (blueprint B1)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.pgn

from tinymlinternship.visualization import artifact_paths, write_game_gif, write_game_pgn

PROJECT_ROOT = Path(__file__).parent.parent
def _sunfish_player(depth: int) -> str:
    return f"sunfish-d{depth}"
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
    player: str | None = None,
    max_seconds: float | None = None,
) -> chess.pgn.Game:
    Position = sunfish_ns["Position"]
    Searcher = sunfish_ns["Searcher"]
    initial = sunfish_ns["initial"]

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Sunfish gate"
    game.headers["Site"] = "SARDINE calibration"
    name = player or _sunfish_player(depth)
    game.headers["White"] = name
    game.headers["Black"] = name
    game.headers["Annotator"] = f"sunfish depth {depth}"

    hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]
    node = game
    game_start = time.perf_counter()
    truncated_time = False

    for ply in range(max_plies):
        if max_seconds is not None and (time.perf_counter() - game_start) >= max_seconds:
            truncated_time = True
            break
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

    if truncated_time:
        game.headers["Termination"] = "time limit"
    game.headers["Result"] = board.result(claim_draw=True)
    return game


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sunfish self-play PGN for ACPL gate")
    parser.add_argument("--games", type=int, default=1, help="Self-play games to generate")
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "images" / "games",
        help="Directory for auto-named PGN + GIF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override PGN path (default: [white]_vs_[black]_[date].pgn)",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF export",
    )
    parser.add_argument("--frame-ms", type=int, default=450, help="GIF frame duration")
    parser.add_argument(
        "--max-game-seconds",
        type=float,
        default=None,
        help="Stop self-play after this many seconds per game",
    )
    args = parser.parse_args(argv)

    if not (SUNFISH_DIR / "sunfish.py").is_file():
        raise SystemExit(
            f"Sunfish not found at {SUNFISH_DIR}. "
            "Clone: git clone https://github.com/thomasahle/sunfish.git models/teacher/sunfish"
        )

    sunfish_ns = _load_sunfish()
    player = _sunfish_player(args.depth)
    games: list[chess.pgn.Game] = []
    for i in range(args.games):
        games.append(
            play_sunfish_game(
                max_plies=args.max_plies,
                depth=args.depth,
                sunfish_ns=sunfish_ns,
                player=player,
                max_seconds=args.max_game_seconds,
            )
        )
        plies = max(0, games[-1].end().ply() - games[-1].ply())
        print(f"Sunfish self-play {i + 1}/{args.games}: {plies} half-moves")

    pgn_path, gif_path = artifact_paths(player, player, args.output_dir)
    if args.output is not None:
        pgn_path = args.output
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    with pgn_path.open("w", encoding="utf-8") as handle:
        for game in games:
            print(game, file=handle, end="\n\n")
    print(f"PGN written: {pgn_path} ({len(games)} game(s))")
    if not args.no_gif and len(games) == 1:
        write_game_gif(games[0], gif_path, frame_ms=args.frame_ms)
        print(f"GIF written: {gif_path} ({gif_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())