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
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.engine
import chess.pgn

from tinymlinternship.bot_eval import (
    analyse_game_acpl,
    elo_estimate_from_acpl,
    merge_reports,
)
from tinymlinternship.config.settings import NNUE_CHECKPOINT_DEFAULT, PROJECT_ROOT
from tinymlinternship.engine import EVAL_CHOICES, ENGINE_VERSION, make_eval_fn
from tinymlinternship.visualization import (
    artifact_paths,
    engine_player_label,
    play_engine_game,
    write_game_gif,
    write_game_pgn,
)


class _TerminalProgress:
    """Single-line progress bar (no extra dependencies)."""

    def __init__(self, total: int, label: str, *, width: int = 32) -> None:
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self._started = time.perf_counter()
        self._last_extra = ""

    def update(self, current: int, extra: str = "") -> None:
        ratio = min(current / self.total, 1.0)
        filled = int(self.width * ratio)
        if filled >= self.width:
            bar = "=" * self.width
        else:
            bar = "=" * filled + ">" + " " * (self.width - filled - 1)
        elapsed = time.perf_counter() - self._started
        if extra:
            self._last_extra = extra
        line = f"\r{self.label} [{bar}] {current}/{self.total}  {elapsed:5.1f}s"
        if self._last_extra:
            line += f"  {self._last_extra}"
        sys.stdout.write(line[:120].ljust(120))
        sys.stdout.flush()

    def finish(self, message: str = "") -> None:
        elapsed = time.perf_counter() - self._started
        sys.stdout.write(f"\r{self.label} done in {elapsed:.1f}s")
        if message:
            sys.stdout.write(f" — {message}")
        sys.stdout.write("\n")
        sys.stdout.flush()


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
    parser.add_argument(
        "--max-game-seconds",
        type=float,
        default=None,
        help="Stop self-play after this many seconds per game (e.g. 300 for 5 min)",
    )
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
        "--max-qsearch-depth",
        type=int,
        default=None,
        metavar="PLIES",
        help=(
            "Cap capture/promotion extensions in quiescence (default: unlimited). "
            "Suggested 6 when --depth 2 with qsearch on."
        ),
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "images" / "games",
        help="Directory for auto-named PGN + GIF (default: images/games)",
    )
    parser.add_argument(
        "--output-pgn",
        type=Path,
        default=None,
        help="Override PGN path (default: [white]_vs_[black]_[date].pgn in --output-dir)",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=None,
        help="Override GIF path (default: [white]_vs_[black]_[date].gif in --output-dir)",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF export",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=450,
        help="Milliseconds per GIF frame",
    )
    parser.add_argument(
        "--white",
        type=str,
        default=None,
        help="White player name for PGN headers and filenames (default: from --eval)",
    )
    parser.add_argument(
        "--black",
        type=str,
        default=None,
        help="Black player name (default: same as white for self-play)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write JSON report",
    )
    parser.add_argument("--verbose", action="store_true", help="Print worst moves")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars",
    )
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
    qsearch_note = "on" if args.quiescence else "off"
    if args.quiescence and args.max_qsearch_depth is not None:
        qsearch_note = f"on,max{args.max_qsearch_depth}"
    annotator = (
        f"SARDINE {ENGINE_VERSION} ({args.eval}, depth {args.depth}, "
        f"qsearch={qsearch_note})"
    )
    default_player = engine_player_label(
        args.eval,
        depth=args.depth,
        quiescence=args.quiescence,
        nnue_checkpoint=args.nnue_checkpoint if args.eval == "nnue" else None,
    )
    white_name = args.white or default_player
    black_name = args.black or default_player

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
            label = f"Self-play {i + 1}/{args.games}"
            if not args.no_progress:
                print(f"{label} ({annotator})...")
                play_bar = _TerminalProgress(args.max_plies, label)

                def _on_ply(
                    ply: int,
                    _max_plies: int,
                    move: chess.Move,
                    ply_sec: float,
                    _bar: _TerminalProgress = play_bar,
                ) -> None:
                    _bar.update(ply, f"ply {ply} {move.uci()} {ply_sec:.1f}s")

                on_ply = _on_ply
            else:
                play_bar = None
                print(f"{label} ({annotator})...")
                on_ply = None

            game = play_engine_game(
                max_plies=args.max_plies,
                depth=args.depth,
                eval_fn=eval_fn,
                quiescence=args.quiescence,
                max_qsearch_depth=args.max_qsearch_depth,
                white_name=white_name,
                black_name=black_name,
                annotator=annotator,
                on_ply=on_ply,
                max_seconds=args.max_game_seconds,
            )
            games.append(game)
            plies = max(0, game.end().ply() - game.ply())
            suffix = game.headers.get("Result", "*")
            if game.headers.get("Termination") == "time limit":
                suffix += " (time limit)"
            if play_bar is not None:
                play_bar.finish(f"{plies} plies, result {suffix}")
            else:
                print(f"  {plies} half-moves, result {suffix}")

    if not args.pgn:
        pgn_path, gif_path = artifact_paths(white_name, black_name, args.output_dir)
        if args.output_pgn is not None:
            pgn_path = args.output_pgn
        if args.output_gif is not None:
            gif_path = args.output_gif

        with pgn_path.open("w", encoding="utf-8") as handle:
            for game in games:
                print(game, file=handle, end="\n\n")
        print(f"PGN written: {pgn_path}")

        if not args.no_gif:
            for i, game in enumerate(games):
                target = gif_path if len(games) == 1 else gif_path.with_stem(
                    f"{gif_path.stem}_game{i + 1}"
                )
                write_game_gif(game, target, frame_ms=args.frame_ms)
                print(f"GIF written: {target} ({target.stat().st_size:,} bytes)")

    reports = []
    with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
        engine.configure({"Threads": 1})
        for i, game in enumerate(games):
            analyse_moves = min(
                args.max_plies,
                sum(1 for _ in game.mainline_moves()),
            )
            sf_label = f"Stockfish {i + 1}/{len(games)}"
            sf_bar = (
                None
                if args.no_progress
                else _TerminalProgress(analyse_moves, sf_label)
            )
            if sf_bar is not None:
                print(f"{sf_label} analysing {analyse_moves} moves...")

            def _on_analyse(
                ply: int,
                total: int,
                move: chess.Move,
                _bar: _TerminalProgress | None = sf_bar,
            ) -> None:
                if _bar is not None:
                    _bar.update(ply, move.uci())

            rep = analyse_game_acpl(
                game,
                engine,
                limit=limit,
                max_plies=args.max_plies,
                on_ply=None if sf_bar is None else _on_analyse,
            )
            if sf_bar is not None:
                sf_bar.finish()
            reports.append(rep)
            if len(games) > 1:
                _print_report(f"Game {i + 1}", rep, verbose=args.verbose)

    combined = merge_reports(reports) if len(reports) > 1 else reports[0]
    _print_report("Combined" if len(reports) > 1 else "Game", combined, verbose=args.verbose)

    report_annotator = annotator
    if args.pgn and games:
        report_annotator = games[0].headers.get("Annotator", annotator)

    if args.json:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "annotator": report_annotator,
            "stockfish": sf_path,
            "sf_limit": {"depth": args.sf_depth, "movetime_ms": args.sf_movetime_ms},
            "max_plies": args.max_plies,
            "max_game_seconds": args.max_game_seconds,
            "max_qsearch_depth": args.max_qsearch_depth,
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