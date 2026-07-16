#!/usr/bin/env python3
"""Stream PGN games → FEN positions + SARDINE bucket metadata.

Reads standard PGN (Lichess export, multi-game files, engine self-play) without
loading the whole file into memory. Emits parquet/CSV with natural bucket
distribution for production labeling (see blueprint §Training data).

Example::

    py -3.12 scripts/lichess_pgn_to_fen.py \\
        --input images/games/hce_d1_vs_hce_d1_2026-07-10.pgn \\
        --max-games 16 --output data/processed/lichess/smoke_fens.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.pgn
import pandas as pd

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.features.bucket import (
    NUM_BUCKETS,
    bucket_id,
    has_queen,
    piece_count,
)

DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "lichess" / "positions.parquet"


def iter_games(pgn_path: Path) -> Iterator[chess.pgn.Game]:
    """Yield games from a PGN file one at a time (streaming)."""
    with pgn_path.open(encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def positions_from_game(
    game: chess.pgn.Game,
    *,
    game_index: int,
    min_ply: int,
    max_plies_per_game: int | None,
    sample_every: int,
    include_startpos: bool,
) -> list[dict[str, Any]]:
    """Walk mainline moves and collect FEN + bucket fields."""
    rows: list[dict[str, Any]] = []
    board = game.board()
    result = game.headers.get("Result", "")
    white = game.headers.get("White", "")
    black = game.headers.get("Black", "")
    site = game.headers.get("Site", "")
    game_id = game.headers.get("GameId") or game.headers.get("LichessId") or f"game_{game_index}"

    def maybe_record(ply: int) -> None:
        if ply < min_ply:
            return
        if sample_every > 1 and (ply - min_ply) % sample_every != 0:
            return
        rows.append(
            {
                "fen": board.fen(),
                "bucket_id": int(bucket_id(board)),
                "piece_count": int(piece_count(board)),
                "has_queen": bool(has_queen(board)),
                "stm_white": board.turn == chess.WHITE,
                "ply": ply,
                "source": "lichess",
                "game_id": str(game_id),
                # optional debug headers (not training targets)
                "game_index": game_index,
                "result": result,
                "white": white,
                "black": black,
                "site": site,
            }
        )

    ply = 0
    if include_startpos:
        maybe_record(ply)

    for move in game.mainline_moves():
        board.push(move)
        ply += 1
        if max_plies_per_game is not None and ply > max_plies_per_game:
            break
        maybe_record(ply)

    return rows


def convert_pgn(
    pgn_path: Path,
    *,
    max_games: int | None = None,
    max_positions: int | None = None,
    min_ply: int = 0,
    max_plies_per_game: int | None = None,
    sample_every: int = 1,
    include_startpos: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert a PGN path to position rows + summary stats."""
    all_rows: list[dict[str, Any]] = []
    games_read = 0
    games_empty = 0
    illegal_or_broken = 0
    bucket_counts: Counter[int] = Counter()

    for game in iter_games(pgn_path):
        if max_games is not None and games_read >= max_games:
            break
        games_read += 1
        try:
            rows = positions_from_game(
                game,
                game_index=games_read - 1,
                min_ply=min_ply,
                max_plies_per_game=max_plies_per_game,
                sample_every=sample_every,
                include_startpos=include_startpos,
            )
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError) as exc:
            illegal_or_broken += 1
            print(f"  skip game {games_read - 1}: {exc}", file=sys.stderr)
            continue

        if not rows:
            games_empty += 1
            continue

        for row in rows:
            bucket_counts[row["bucket_id"]] += 1
            all_rows.append(row)
            if max_positions is not None and len(all_rows) >= max_positions:
                break
        if max_positions is not None and len(all_rows) >= max_positions:
            break

    stats = {
        "source": str(pgn_path),
        "games_read": games_read,
        "games_empty": games_empty,
        "games_broken": illegal_or_broken,
        "positions": len(all_rows),
        "min_ply": min_ply,
        "sample_every": sample_every,
        "include_startpos": include_startpos,
        "bucket_counts": {str(b): int(bucket_counts.get(b, 0)) for b in range(NUM_BUCKETS)},
        "buckets_present": sorted(int(b) for b in bucket_counts if bucket_counts[b] > 0),
        "buckets_missing": sorted(
            b for b in range(NUM_BUCKETS) if bucket_counts.get(b, 0) == 0
        ),
    }
    return all_rows, stats


def write_output(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output, index=False)
    elif suffix in (".parquet", ".pq"):
        df.to_parquet(output, index=False)
    else:
        # Default to parquet if extension is odd
        df.to_parquet(output, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream PGN → FEN + bucket_id (SARDINE production data path)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="PGN file (Lichess export or multi-game)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output parquet/CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Stop after N games")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Stop after N positions total",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=0,
        help="Skip positions with ply < N (half-moves from start; default 0)",
    )
    parser.add_argument(
        "--max-plies-per-game",
        type=int,
        default=None,
        help="Cap mainline length per game",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Keep every N-th position after min-ply (default 1 = all)",
    )
    parser.add_argument(
        "--no-startpos",
        action="store_true",
        help="Do not record the initial FEN of each game",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional path for stats JSON (default: next to output as *.stats.json)",
    )
    args = parser.parse_args(argv)

    pgn_path = args.input
    if not pgn_path.is_file():
        print(f"PGN not found: {pgn_path}", file=sys.stderr)
        return 1
    if args.sample_every < 1:
        print("--sample-every must be >= 1", file=sys.stderr)
        return 1

    print(f"Reading {pgn_path} ...")
    rows, stats = convert_pgn(
        pgn_path.resolve(),
        max_games=args.max_games,
        max_positions=args.max_positions,
        min_ply=args.min_ply,
        max_plies_per_game=args.max_plies_per_game,
        sample_every=args.sample_every,
        include_startpos=not args.no_startpos,
    )

    if not rows:
        print("No positions extracted.", file=sys.stderr)
        return 1

    out = args.output
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve() if not out.exists() else out.resolve()
    else:
        out = out.resolve()

    write_output(rows, out)

    stats_path = args.stats_json
    if stats_path is None:
        stats_path = out.with_suffix(out.suffix + ".stats.json")
        if out.suffix.lower() in (".parquet", ".pq", ".csv"):
            stats_path = out.with_name(out.stem + ".stats.json")
    else:
        stats_path = stats_path.resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel_src = str(pgn_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        rel_src = str(pgn_path.resolve())
    try:
        rel_out = str(out.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_out = str(out)
    stats["source"] = rel_src
    stats["output"] = rel_out
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows):,} positions → {rel_out}")
    print(f"Games read: {stats['games_read']}  (empty={stats['games_empty']}, broken={stats['games_broken']})")
    print("Bucket counts:")
    for b in range(NUM_BUCKETS):
        n = stats["bucket_counts"][str(b)]
        bar = "#" * min(40, n // max(1, len(rows) // 40)) if n else ""
        print(f"  bucket {b}: {n:6d}  {bar}")
    missing = stats["buckets_missing"]
    if missing:
        print(f"Buckets missing (skew / short games): {missing}")
    else:
        print("All 8 buckets present.")
    print(f"Stats: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
