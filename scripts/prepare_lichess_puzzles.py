#!/usr/bin/env python3
"""Sample Lichess puzzles → pre-label parquet (FEN after opponent setup move).

Source: https://huggingface.co/datasets/Lichess/chess-puzzles
(ai-feed.md §1 / P1 tactical boost). The published FEN is *before* the
opponent's first move; we push that move so ``fen`` is the player-to-move
position.

Example::

    py -3.12 scripts/prepare_lichess_puzzles.py --limit 4000
    py -3.12 scripts/label_positions.py --input data/raw/lichess/puzzles_sample.parquet \\
        --output data/processed/labeled/lichess_puzzles.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import pandas as pd

from tinymlinternship.config.settings import LICHESS_RAW_DIR, PROJECT_ROOT
from tinymlinternship.data.schema import ensure_prelabel_columns

HF_DATASET = "Lichess/chess-puzzles"
SOURCE_NAME = "lichess_puzzles"
DEFAULT_OUTPUT = LICHESS_RAW_DIR / "puzzles_sample.parquet"


def puzzle_player_fen(fen: str, moves: str) -> str:
    """FEN after the opponent's setup move (first UCI in ``Moves``)."""
    board = chess.Board(fen)
    tokens = str(moves).split()
    if not tokens:
        raise ValueError("empty Moves")
    board.push_uci(tokens[0])
    return board.fen()


def sample_puzzles(*, limit: int, skip: int) -> pd.DataFrame:
    from datasets import load_dataset

    stream = load_dataset(HF_DATASET, split="train", streaming=True)
    rows: list[dict] = []
    seen = 0
    for rec in stream:
        if seen < skip:
            seen += 1
            continue
        fen_raw = rec.get("FEN") or rec.get("fen")
        moves = rec.get("Moves") or rec.get("moves") or ""
        if not fen_raw:
            continue
        try:
            fen = puzzle_player_fen(str(fen_raw), str(moves))
        except (ValueError, chess.IllegalMoveError, chess.InvalidFenError):
            continue
        puzzle_id = rec.get("PuzzleId") or rec.get("puzzleid") or f"row_{seen}"
        rows.append(
            {
                "fen": fen,
                "source": SOURCE_NAME,
                "game_id": f"puzzle:{puzzle_id}",
                "ply": -1,
                "puzzle_id": str(puzzle_id),
                "puzzle_rating": rec.get("Rating"),
                "themes": rec.get("Themes"),
            }
        )
        if len(rows) >= limit:
            break
        seen += 1
    if not rows:
        raise RuntimeError(f"no puzzles sampled from {HF_DATASET}")
    return ensure_prelabel_columns(pd.DataFrame(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample Lichess puzzles → FEN parquet")
    parser.add_argument("--limit", type=int, default=4000, help="Puzzles to keep")
    parser.add_argument("--skip", type=int, default=0, help="Streaming rows to skip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    output = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output).resolve()
    df = sample_puzzles(limit=args.limit, skip=args.skip)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hf_dataset": HF_DATASET,
        "source": SOURCE_NAME,
        "rows": int(len(df)),
        "output": str(output.relative_to(PROJECT_ROOT)),
        "fen": "after first UCI in Moves (player to move)",
    }
    output.with_suffix(".stats.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(df):,} puzzle FENs → {output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
