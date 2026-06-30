"""Plot the distribution of piece counts across positions in a sample of games."""

import argparse
import csv
from pathlib import Path

import chess
import matplotlib.pyplot as plt
import pandas as pd

from tinymlinternship.config.settings import LICHESS_CSV, PROJECT_ROOT


def collect_piece_counts(
    csv_path: Path,
    max_games: int,
    min_moves: int = 0,
) -> tuple[list[int], int]:
    counts: list[int] = []
    games_used = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if games_used >= max_games:
                break

            moves_str = row.get("moves", "").strip()
            if not moves_str:
                continue

            move_list = moves_str.split()
            if len(move_list) < min_moves:
                continue

            board = chess.Board()
            counts.append(len(board.piece_map()))

            for san in move_list:
                try:
                    board.push_san(san)
                except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
                    break
                counts.append(len(board.piece_map()))

            games_used += 1

    return counts, games_used


def plot_distribution(
    counts: list[int],
    output_path: Path,
    games_used: int,
    min_moves: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = range(min(counts), max(counts) + 2)
    ax.hist(counts, bins=bins, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.set_xlabel("Number of pieces on the board")
    ax.set_ylabel("Number of positions")
    filter_note = f", moves ≥ {min_moves}" if min_moves else ""
    ax.set_title(
        f"Piece-count distribution ({games_used:,} games{filter_note}, {len(counts):,} positions)"
    )
    ax.set_xticks(list(bins))
    ax.grid(axis="y", alpha=0.3)

    mean_count = sum(counts) / len(counts)
    ax.axvline(mean_count, color="#C44E52", linestyle="--", linewidth=1.5, label=f"mean = {mean_count:.1f}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_to_excel(counts: list[int], output_path: Path) -> None:
    frequency = pd.Series(counts).value_counts()
    distribution = pd.DataFrame(
        {
            "piece_count": range(2, 33),
            "position_count": [int(frequency.get(n, 0)) for n in range(2, 33)],
        }
    )
    distribution.to_excel(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-games", type=int, default=1000)
    parser.add_argument(
        "--min-moves",
        type=int,
        default=0,
        help="Skip games with fewer than this many moves (plies in the moves column).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "plots" / "piece_count_distribution.png",
    )
    parser.add_argument(
        "--excel-output",
        type=Path,
        default=PROJECT_ROOT / "excel" / "piece_count_distribution.xlsx",
    )
    args = parser.parse_args()

    if not LICHESS_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {LICHESS_CSV}. Run scripts/download_data.py first.")

    counts, games_used = collect_piece_counts(
        LICHESS_CSV, args.max_games, min_moves=args.min_moves
    )
    if not counts:
        raise RuntimeError("No positions collected — check the dataset format or min-moves filter.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.excel_output.parent.mkdir(parents=True, exist_ok=True)

    plot_distribution(counts, args.output, games_used, args.min_moves)
    export_to_excel(counts, args.excel_output)
    print(f"Collected {len(counts):,} positions from {games_used:,} games (min_moves={args.min_moves})")
    print(f"Piece count range: {min(counts)} – {max(counts)}")
    print(f"Saved plot to: {args.output}")
    print(f"Saved Excel to: {args.excel_output}")


if __name__ == "__main__":
    main()