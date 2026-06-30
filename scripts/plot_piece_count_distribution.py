"""Plot the distribution of piece counts across positions in a sample of games."""

import argparse
import csv
from pathlib import Path

import chess
import matplotlib.pyplot as plt
import pandas as pd

from tinymlinternship.config.settings import LICHESS_CSV, PROJECT_ROOT


def collect_piece_counts(csv_path: Path, max_games: int) -> list[int]:
    counts: list[int] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_games:
                break

            moves_str = row.get("moves", "").strip()
            if not moves_str:
                continue

            board = chess.Board()
            counts.append(len(board.piece_map()))

            for san in moves_str.split():
                try:
                    board.push_san(san)
                except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
                    break
                counts.append(len(board.piece_map()))

    return counts


def plot_distribution(counts: list[int], output_path: Path, max_games: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = range(min(counts), max(counts) + 2)
    ax.hist(counts, bins=bins, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.set_xlabel("Number of pieces on the board")
    ax.set_ylabel("Number of positions")
    ax.set_title(f"Piece-count distribution ({max_games} games, {len(counts):,} positions)")
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "piece_count_distribution.png",
    )
    parser.add_argument(
        "--excel-output",
        type=Path,
        default=PROJECT_ROOT / "piece_count_distribution.xlsx",
    )
    args = parser.parse_args()

    if not LICHESS_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {LICHESS_CSV}. Run scripts/download_data.py first.")

    counts = collect_piece_counts(LICHESS_CSV, args.max_games)
    if not counts:
        raise RuntimeError("No positions collected — check the dataset format.")

    plot_distribution(counts, args.output, args.max_games)
    export_to_excel(counts, args.excel_output)
    print(f"Collected {len(counts):,} positions from {args.max_games} games")
    print(f"Piece count range: {min(counts)} – {max(counts)}")
    print(f"Saved plot to: {args.output}")
    print(f"Saved Excel to: {args.excel_output}")


if __name__ == "__main__":
    main()