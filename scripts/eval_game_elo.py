#!/usr/bin/env python3
"""
Estimate per-player Elo from a game record via Stockfish ACPL (blueprint §Bot Evaluation).

Accepts PGN file or UCI move list; reports White and Black separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess.engine
import chess.pgn

from tinymlinternship.bot_eval import (
    analyse_game_acpl,
    game_from_moves,
    resolve_stockfish,
    split_report_by_side,
)
from tinymlinternship.bot_eval.acpl import iter_games


def _load_game(
    *,
    pgn: Path | None,
    moves: str | None,
    fen: str | None,
    white: str,
    black: str,
) -> chess.pgn.Game:
    if pgn is not None:
        text = pgn.read_text(encoding="utf-8")
        games = list(iter_games(text))
        if not games:
            raise SystemExit(f"No games in {pgn}")
        if len(games) > 1:
            print(f"Warning: {len(games)} games in PGN — analysing first only", file=sys.stderr)
        return games[0]
    if moves:
        tokens = moves.split()
        if not tokens:
            raise SystemExit("Empty --moves list")
        return game_from_moves(tokens, fen=fen, white=white, black=black)
    raise SystemExit("Provide --pgn PATH or --moves 'e2e4 e7e5 ...'")


def _print_side(label: str, report, *, verbose: bool) -> None:
    print(f"\n=== {label} ===")
    print(f"Moves:  {report.moves_analysed}")
    if report.moves_analysed == 0:
        print("  (no moves for this side)")
        return
    print(f"ACPL:   {report.acpl:.1f} cp  (σ={report.acpl_std:.1f})")
    print(f"Elo:    {report.elo_estimate}  (range {report.elo_range_str})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-player Elo estimate from a game record (Stockfish ACPL)"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pgn", type=Path, help="PGN file (first game used)")
    src.add_argument(
        "--moves",
        type=str,
        help="UCI move list, space-separated (e.g. 'e2e4 e7e5 g1f3')",
    )
    parser.add_argument("--fen", type=str, default=None, help="Start FEN (with --moves)")
    parser.add_argument("--white", type=str, default="?", help="White player name")
    parser.add_argument("--black", type=str, default="?", help="Black player name")
    parser.add_argument("--stockfish", type=str, default=None, help="Stockfish binary path")
    parser.add_argument("--sf-depth", type=int, default=16, help="Analysis depth")
    parser.add_argument(
        "--sf-movetime-ms",
        type=int,
        default=None,
        help="Analysis movetime per position (overrides --sf-depth)",
    )
    parser.add_argument("--max-plies", type=int, default=None, help="Cap half-moves analysed")
    parser.add_argument("--json", type=Path, default=None, help="Write JSON report")
    parser.add_argument("--verbose", action="store_true", help="Print worst moves per side")
    args = parser.parse_args(argv)

    try:
        sf_path = resolve_stockfish(args.stockfish)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    game = _load_game(
        pgn=args.pgn,
        moves=args.moves,
        fen=args.fen,
        white=args.white,
        black=args.black,
    )
    white_name = game.headers.get("White", args.white)
    black_name = game.headers.get("Black", args.black)
    print(f"Game: {white_name} vs {black_name}  result {game.headers.get('Result', '*')}")

    limit = (
        chess.engine.Limit(time=args.sf_movetime_ms / 1000)
        if args.sf_movetime_ms
        else chess.engine.Limit(depth=args.sf_depth)
    )

    with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
        engine.configure({"Threads": 1})
        full = analyse_game_acpl(
            game, engine, limit=limit, max_plies=args.max_plies
        )
    sides = split_report_by_side(full)

    _print_side(f"White ({white_name})", sides["white"], verbose=args.verbose)
    _print_side(f"Black ({black_name})", sides["black"], verbose=args.verbose)
    print(f"\nFormula: Elo ≈ 2855 - ACPL × 10")

    if args.verbose:
        for color, rep in sides.items():
            if not rep.move_results:
                continue
            print(f"\nWorst {color} moves:")
            for r in sorted(rep.move_results, key=lambda x: x.cpl, reverse=True)[:3]:
                if r.cpl:
                    print(f"  ply {r.ply}: {r.move_uci} CPL={r.cpl}")

    if args.json:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "stockfish": sf_path,
            "sf_limit": {"depth": args.sf_depth, "movetime_ms": args.sf_movetime_ms},
            "white": white_name,
            "black": black_name,
            "result": game.headers.get("Result", "*"),
            "players": {
                "white": _side_json(sides["white"]),
                "black": _side_json(sides["black"]),
            },
            "elo_formula": "2855 - 10 * ACPL",
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json}")

    return 0


def _side_json(report) -> dict:
    return {
        "moves_analysed": report.moves_analysed,
        "acpl": round(report.acpl, 2),
        "acpl_std": round(report.acpl_std, 2),
        "elo_estimate": report.elo_estimate,
        "elo_low": report.elo_low,
        "elo_high": report.elo_high,
    }


if __name__ == "__main__":
    raise SystemExit(main())