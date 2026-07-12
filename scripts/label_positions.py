#!/usr/bin/env python3
"""Label chess positions with Lc0 UCI WDL → expected_reward = W − L (White POV).

Input: parquet/CSV with a ``fen`` column, or built-in startpos smoke positions.
Output: parquet with ``fen``, ``expected_reward``, and raw WDL permille columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import pandas as pd

from tinymlinternship.config.settings import (
    LC0_NETWORK_DEFAULT,
    LC0_NETWORK_PRESETS,
    PROJECT_ROOT,
)
from tinymlinternship.engine.eval_lc0 import Lc0Teacher


def smoke_fens(*, moves: int = 2) -> list[str]:
    """Startpos plus a short self-play line for quick teacher checks."""
    board = chess.Board()
    fens = [board.fen()]
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5")[: moves]:
        board.push_uci(uci)
        fens.append(board.fen())
    return fens


def load_fens(path: Path, *, limit: int | None, fen_column: str) -> list[str]:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if fen_column not in df.columns:
        raise ValueError(f"column {fen_column!r} not in {path} (have {list(df.columns)})")
    fens = df[fen_column].astype(str).tolist()
    if limit is not None:
        fens = fens[:limit]
    return fens


def label_fens(
    fens: list[str],
    teacher: Lc0Teacher,
    *,
    verbose: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for i, fen in enumerate(fens, start=1):
        board = chess.Board(fen)
        win, draw, loss = teacher.evaluate_wdl(board)
        reward = teacher.evaluate_expected_reward(board)
        row = {
            "fen": fen,
            "expected_reward": reward,
            "wdl_win": win,
            "wdl_draw": draw,
            "wdl_loss": loss,
        }
        rows.append(row)
        if verbose:
            print(
                f"[{i}/{len(fens)}] reward={reward:+.4f} wdl={win}/{draw}/{loss}  {fen[:48]}..."
            )
    return rows


def validate_rewards(rows: list[dict]) -> None:
    bad = [r for r in rows if not (-1.0 <= r["expected_reward"] <= 1.0)]
    if bad:
        sample = bad[0]
        raise ValueError(
            f"{len(bad)} label(s) outside [-1, +1]; first: reward={sample['expected_reward']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label FEN positions via Lc0 UCI WDL (expected_reward = W − L, White POV)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input parquet/CSV with a fen column (default: smoke startpos line)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet (default: data/processed/labeled/smoke_labeled.parquet)",
    )
    parser.add_argument("--fen-column", default="fen")
    parser.add_argument("--limit", type=int, default=None, help="Max positions to label")
    parser.add_argument(
        "--network",
        choices=sorted(LC0_NETWORK_PRESETS),
        default=None,
        help=f"Lc0 weights preset (default: {LC0_NETWORK_DEFAULT.name})",
    )
    parser.add_argument("--backend", default="blas")
    parser.add_argument(
        "--smoke-moves",
        type=int,
        default=2,
        help="When --input is omitted, number of plies after startpos (default 2)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.input is None:
        fens = smoke_fens(moves=args.smoke_moves)
        if args.limit is not None:
            fens = fens[: args.limit]
        source = "smoke:startpos"
    else:
        if not args.input.exists():
            print(f"Input not found: {args.input}", file=sys.stderr)
            return 1
        input_path = args.input.resolve()
        fens = load_fens(input_path, limit=args.limit, fen_column=args.fen_column)
        try:
            source = str(input_path.relative_to(PROJECT_ROOT))
        except ValueError:
            source = str(input_path)

    if not fens:
        print("No positions to label.", file=sys.stderr)
        return 1

    weights = LC0_NETWORK_PRESETS[args.network] if args.network else LC0_NETWORK_DEFAULT
    output = args.output.resolve() if args.output else None
    if output is None:
        output = PROJECT_ROOT / "data" / "processed" / "labeled" / "smoke_labeled.parquet"

    with Lc0Teacher(weights=str(weights), backend=args.backend) as teacher:
        rows = label_fens(fens, teacher, verbose=args.verbose)

    validate_rewards(rows)
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    summary = {
        "source": source,
        "count": len(rows),
        "network": weights.name,
        "expected_reward_min": float(df["expected_reward"].min()),
        "expected_reward_max": float(df["expected_reward"].max()),
        "expected_reward_mean": float(df["expected_reward"].mean()),
        "output": str(output.relative_to(PROJECT_ROOT.resolve())),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())