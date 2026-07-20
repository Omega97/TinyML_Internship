#!/usr/bin/env python3
"""Label chess positions with Lc0 UCI WDL → expected_reward (White POV).

Input: parquet/CSV with a ``fen`` column (ideally full ASSETS pre-label schema).
Output: same metadata columns preserved + ``expected_reward``, WDL, ``teacher_network``.

See ASSETS.md §Ideal final training set.
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
from tinymlinternship.data.schema import (
    LABEL_FORMULA,
    ensure_prelabel_columns,
    sha256_file,
    stm_white_from_fen,
    validate_rewards_series,
)
from tinymlinternship.engine.eval_lc0 import (
    Lc0Teacher,
    wdl_to_expected_reward_white,
)


def _terminal_wdl_and_reward(board: chess.Board) -> tuple[tuple[int, int, int], float] | None:
    """Synthetic STM WDL + White-POV reward for positions where Lc0 may omit WDL."""
    if board.is_checkmate():
        # Side to move is mated.
        return (0, 0, 1000), (-1.0 if board.turn == chess.WHITE else 1.0)
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_threefold_repetition()
        or board.can_claim_fifty_moves()
    ):
        return (0, 1000, 0), 0.0
    return None


def smoke_frame(*, moves: int = 2) -> pd.DataFrame:
    """Startpos plus a short line for quick teacher checks."""
    board = chess.Board()
    rows: list[dict] = []
    fens = [board.fen()]
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5")[:moves]:
        board.push_uci(uci)
        fens.append(board.fen())
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        from tinymlinternship.features.bucket import bucket_id, has_queen, piece_count

        rows.append(
            {
                "fen": fen,
                "bucket_id": int(bucket_id(b)),
                "piece_count": int(piece_count(b)),
                "has_queen": bool(has_queen(b)),
                "stm_white": b.turn == chess.WHITE,
                "ply": i,
                "source": "lichess",
                "game_id": "smoke:startpos",
            }
        )
    return pd.DataFrame(rows)


def load_input_frame(path: Path, *, limit: int | None, fen_column: str) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if fen_column not in df.columns:
        raise ValueError(f"column {fen_column!r} not in {path} (have {list(df.columns)})")
    if fen_column != "fen":
        df = df.rename(columns={fen_column: "fen"})
    if limit is not None:
        df = df.iloc[:limit].copy()
    return ensure_prelabel_columns(df)


def label_frame(
    df: pd.DataFrame,
    teacher: Lc0Teacher,
    *,
    teacher_network: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Label each row; preserve all input columns; add reward + WDL + teacher id."""
    rewards: list[float] = []
    wins: list[int] = []
    draws: list[int] = []
    losses: list[int] = []
    n = len(df)
    for i, fen in enumerate(df["fen"].astype(str).tolist(), start=1):
        board = chess.Board(fen)
        terminal = _terminal_wdl_and_reward(board)
        if terminal is not None:
            (win, draw, loss), reward = terminal
        else:
            win, draw, loss = teacher.evaluate_wdl(board)
            reward = wdl_to_expected_reward_white(board, win, draw, loss)
        rewards.append(float(reward))
        wins.append(int(win))
        draws.append(int(draw))
        losses.append(int(loss))
        if verbose:
            print(
                f"[{i}/{n}] reward={reward:+.4f} wdl={win}/{draw}/{loss}  {fen[:48]}..."
            )
    out = df.copy()
    out["expected_reward"] = rewards
    out["wdl_win"] = wins
    out["wdl_draw"] = draws
    out["wdl_loss"] = losses
    out["teacher_network"] = teacher_network
    if "stm_white" not in out.columns:
        out["stm_white"] = out["fen"].map(stm_white_from_fen)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label FEN positions via Lc0 UCI WDL (expected_reward White POV)"
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
        df = smoke_frame(moves=args.smoke_moves)
        if args.limit is not None:
            df = df.iloc[: args.limit].copy()
        source = "smoke:startpos"
    else:
        if not args.input.exists():
            print(f"Input not found: {args.input}", file=sys.stderr)
            return 1
        input_path = args.input.resolve()
        df = load_input_frame(input_path, limit=args.limit, fen_column=args.fen_column)
        try:
            source = str(input_path.relative_to(PROJECT_ROOT))
        except ValueError:
            source = str(input_path)

    if df.empty:
        print("No positions to label.", file=sys.stderr)
        return 1

    weights = LC0_NETWORK_PRESETS[args.network] if args.network else LC0_NETWORK_DEFAULT
    weights = weights.resolve() if weights.exists() else weights
    teacher_name = weights.name
    output = args.output.resolve() if args.output else None
    if output is None:
        output = PROJECT_ROOT / "data" / "processed" / "labeled" / "smoke_labeled.parquet"

    with Lc0Teacher(weights=str(weights), backend=args.backend) as teacher:
        out = label_frame(df, teacher, teacher_network=teacher_name, verbose=args.verbose)

    validate_rewards_series(out["expected_reward"])
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    net_sha = sha256_file(weights) if Path(weights).is_file() else None
    try:
        out_rel = str(output.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        out_rel = str(output)

    summary = {
        "source": source,
        "count": len(out),
        "network": teacher_name,
        "network_sha256": net_sha,
        "label_formula": LABEL_FORMULA,
        "expected_reward_min": float(out["expected_reward"].min()),
        "expected_reward_max": float(out["expected_reward"].max()),
        "expected_reward_mean": float(out["expected_reward"].mean()),
        "columns": list(out.columns),
        "output": out_rel,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
