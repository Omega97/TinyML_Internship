"""
Minimal script to display moves from a specific game in the Lichess dataset.
"""
import pandas as pd
from src.tinymlinternship.config.settings import LICHESS_CSV


def display_game(game_id: int = 0):
    """Display moves for a game by index."""
    if not LICHESS_CSV.exists():
        print("❌ Dataset not found. Please run download_data.py first.")
        return

    df = pd.read_csv(LICHESS_CSV)

    if game_id >= len(df):
        print(f"❌ Game index {game_id} out of range. Total games: {len(df)}")
        game_id = 0

    game = df.iloc[game_id]

    print("=" * 80)
    print(f"GAME #{game_id}")
    print("=" * 80)
    print(f"White: {game['white_id']} ({game['white_rating']})")
    print(f"Black: {game['black_id']} ({game['black_rating']})")
    print(f"Winner: {game['winner'].upper()}")
    print(f"Opening: {game.get('opening_name', 'Unknown')}")
    print("-" * 80)
    print("MOVES:")
    print(game['moves'])
    print("=" * 80)


if __name__ == "__main__":
    display_game(game_id=0)
