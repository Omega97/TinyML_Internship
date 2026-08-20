#!/usr/bin/env python3
"""Cfish vs Cfish self-play → PGN for Stockfish ACPL gate."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CFISH_DIR = _REPO_ROOT / "src" / "cfish"
_CFISH_EXE = _CFISH_DIR / ("cfish.exe" if sys.platform == "win32" else "cfish")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cfish hybrid self-play → PGN")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "images" / "plots" / "PGN_and_JSON" / "cfish_hybrid_d5_gate.pgn",
    )
    parser.add_argument(
        "--also-games-dir",
        type=Path,
        default=_REPO_ROOT / "images" / "games",
        help="Also write dated PGN under images/games (empty to skip)",
    )
    args = parser.parse_args(argv)

    if not _CFISH_EXE.is_file():
        raise SystemExit(f"Cfish binary not found: {_CFISH_EXE}")

    engine = chess.engine.SimpleEngine.popen_uci(str(_CFISH_EXE), cwd=str(_CFISH_DIR))
    games: list[chess.pgn.Game] = []
    nps_samples: list[int] = []
    try:
        try:
            engine.configure(
                {
                    "LargePages": False,
                    "Threads": 1,
                    "Hash": args.hash_mb,
                    "Use NNUE": "Hybrid",
                }
            )
        except chess.engine.EngineError as exc:
            print(f"configure warn: {exc}", file=sys.stderr)

        limit = chess.engine.Limit(depth=args.depth)
        label = f"Cfish-hybrid-d{args.depth}"
        for g in range(args.games):
            # Mild diversity across games
            try:
                engine.configure({"Contempt": 24 + g * 12})
            except chess.engine.EngineError:
                pass

            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Cfish self-play gate"
            game.headers["Site"] = "SARDINE"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["White"] = label
            game.headers["Black"] = label
            game.headers["Annotator"] = (
                f"Cfish Hybrid NNUE nn-62ef826d1a6d depth {args.depth}"
            )
            game.headers["Round"] = str(g + 1)
            node = game

            for _ply in range(args.max_plies):
                if board.is_game_over(claim_draw=True):
                    break
                result = engine.play(board, limit, info=chess.engine.INFO_ALL)
                if result.info and "nps" in result.info:
                    nps_samples.append(int(result.info["nps"]))
                move = result.move
                if move is None:
                    break
                board.push(move)
                node = node.add_variation(move)

            game.headers["Result"] = board.result(claim_draw=True)
            games.append(game)
            print(
                f"game {g + 1}/{args.games}: plies={board.ply()} "
                f"result={game.headers['Result']}"
            )
    finally:
        engine.quit()

    text = "\n\n".join(str(g) for g in games) + "\n\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"PGN written: {args.output}")

    if args.also_games_dir is not None:
        args.also_games_dir.mkdir(parents=True, exist_ok=True)
        dated = (
            args.also_games_dir
            / f"{label}_vs_{label}_{datetime.now().strftime('%Y-%m-%d')}.pgn"
        )
        dated.write_text(text, encoding="utf-8")
        print(f"PGN copy: {dated}")

    if nps_samples:
        mean_nps = sum(nps_samples) // len(nps_samples)
        print(
            f"nps during play: mean={mean_nps} "
            f"min={min(nps_samples)} max={max(nps_samples)} n={len(nps_samples)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
