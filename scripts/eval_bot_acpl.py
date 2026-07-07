#!/usr/bin/env python3
"""
Depth-1 gate: self-play → Stockfish ACPL → heuristic Elo (blueprint §Bot Evaluation).

Not a win-rate match vs Sunfish — Stockfish judges move quality in our own games.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess.engine
import chess.pgn

from tinymlinternship.bot_eval import (
    analyse_game_acpl,
    elo_estimate_from_acpl,
    merge_reports,
)
from tinymlinternship.config.settings import NNUE_CHECKPOINT_DEFAULT, PROJECT_ROOT
from tinymlinternship.engine import EVAL_CHOICES, ENGINE_VERSION, make_eval_fn
from tinymlinternship.visualization import play_engine_game


def _resolve_stockfish(path: str | None) -> str:
    if path:
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"Stockfish binary not found: {p}")
        return str(p.resolve())
    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    for candidate in (
        PROJECT_ROOT / "models" / "teacher" / "stockfish" / "stockfish.exe",
        Path(r"C:\Program Files\Stockfish\stockfish.exe"),
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    raise SystemExit(
        "Stockfish not found — install Stockfish and pass --stockfish PATH, "
        "or set STOCKFISH_PATH (see https://stockfishchess.org/download/)."
    )


def _print_report(
    label: str,
    report,
    *,
    verbose: bool,
) -> None:
    print(f"\n=== {label} ===")
    print(f"Moves analysed: {report.moves_analysed}")
    print(f"ACPL:           {report.acpl:.1f} cp  (σ={report.acpl_std:.1f})")
    print(f"Elo heuristic:  {report.elo_estimate}  (range {report.elo_range_str})")
    print(f"  formula: Elo ≈ {2855} - ACPL × {10}")
    if verbose and report.move_results:
        print("\nWorst moves (top 5 CPL):")
        worst = sorted(report.move_results, key=lambda r: r.cpl, reverse=True)[:5]
        for r in worst:
            if r.cpl == 0:
                continue
            print(
                f"  ply {r.ply} {r.side}: {r.move_uci} "
                f"(best {r.best_move_uci}) CPL={r.cpl} "
                f"[{r.played_cp:+d} vs {r.best_cp:+d} cp]"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SARDINE depth-1 gate: self-play + Stockfish ACPL → Elo heuristic"
    )
    parser.add_argument("--games", type=int, default=1, help="Self-play games to generate")
    parser.add_argument("--max-plies", type=int, default=80, help="Max half-moves per game")
    parser.add_argument("--depth", type=int, default=1, help="SARDINE search depth")
    parser.add_argument(
        "--eval",
        choices=EVAL_CHOICES,
        default="nnue",
        help="Eval backend for self-play (default: nnue)",
    )
    parser.add_argument(
        "--nnue-checkpoint",
        type=Path,
        default=NNUE_CHECKPOINT_DEFAULT,
    )
    parser.add_argument(
        "--quiescence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture quiescence at leaves (default: off for depth-1 gate)",
    )
    parser.add_argument(
        "--pgn",
        type=Path,
        default=None,
        help="Analyse existing PGN instead of generating self-play",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to stockfish binary (or STOCKFISH_PATH env)",
    )
    parser.add_argument(
        "--sf-depth",
        type=int,
        default=16,
        help="Stockfish analysis depth (default: 16)",
    )
    parser.add_argument(
        "--sf-movetime-ms",
        type=int,
        default=None,
        help="Stockfish movetime per position (overrides --sf-depth if set)",
    )
    parser.add_argument(
        "--output-pgn",
        type=Path,
        default=None,
        help="Write self-play PGN to this path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write JSON report",
    )
    parser.add_argument("--verbose", action="store_true", help="Print worst moves")
    args = parser.parse_args(argv)

    sf_path = _resolve_stockfish(args.stockfish)
    limit = (
        chess.engine.Limit(time=args.sf_movetime_ms / 1000)
        if args.sf_movetime_ms
        else chess.engine.Limit(depth=args.sf_depth)
    )

    eval_fn = make_eval_fn(
        args.eval,
        nnue_checkpoint=args.nnue_checkpoint if args.eval == "nnue" else None,
    )
    annotator = (
        f"SARDINE {ENGINE_VERSION} ({args.eval}, depth {args.depth}, "
        f"qsearch={'on' if args.quiescence else 'off'})"
    )

    games: list[chess.pgn.Game] = []
    if args.pgn is not None:
        text = args.pgn.read_text(encoding="utf-8")
        from tinymlinternship.bot_eval.acpl import iter_games

        games = list(iter_games(text))
        if not games:
            raise SystemExit(f"No games in {args.pgn}")
        print(f"Loaded {len(games)} game(s) from {args.pgn}")
    else:
        for i in range(args.games):
            print(f"Self-play game {i + 1}/{args.games} ({annotator})...")
            game = play_engine_game(
                max_plies=args.max_plies,
                depth=args.depth,
                eval_fn=eval_fn,
                quiescence=args.quiescence,
                annotator=annotator,
            )
            games.append(game)
            plies = max(0, game.end().ply() - game.ply())
            print(f"  {plies} half-moves, result {game.headers.get('Result', '*')}")

    if args.output_pgn and not args.pgn:
        args.output_pgn.parent.mkdir(parents=True, exist_ok=True)
        with args.output_pgn.open("w", encoding="utf-8") as f:
            for g in games:
                print(g, file=f, end="\n\n")
        print(f"PGN written: {args.output_pgn}")

    reports = []
    with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
        engine.configure({"Threads": 1})
        for i, game in enumerate(games):
            rep = analyse_game_acpl(game, engine, limit=limit, max_plies=args.max_plies)
            reports.append(rep)
            if len(games) > 1:
                _print_report(f"Game {i + 1}", rep, verbose=args.verbose)

    combined = merge_reports(reports) if len(reports) > 1 else reports[0]
    _print_report("Combined" if len(reports) > 1 else "Game", combined, verbose=args.verbose)

    if args.json:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "annotator": annotator,
            "stockfish": sf_path,
            "sf_limit": {"depth": args.sf_depth, "movetime_ms": args.sf_movetime_ms},
            "games": len(games),
            "acpl": round(combined.acpl, 2),
            "acpl_std": round(combined.acpl_std, 2),
            "moves_analysed": combined.moves_analysed,
            "elo_estimate": combined.elo_estimate,
            "elo_low": combined.elo_low,
            "elo_high": combined.elo_high,
            "elo_formula": "2855 - 10 * ACPL",
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON report: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())