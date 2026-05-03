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

    # Load data
    df = pd.read_csv(LICHESS_CSV)

    # Fetch game
    if game_id >= len(df):
        print(f"❌ Game index {game_id} out of range. Total games: {len(df)}")
        game_id = 0
    game = df.iloc[game_id]

    print(f"GAME #{game_id}")
    for key in df.columns:
        print(f"{key:_<15}  {game.get(key, 'Unknown')}")


def main():
    """
    GAME #1
    id_____________  l1NXvwaE
    rated__________  True
    created_at_____  1504130000000.0
    last_move_at___  1504130000000.0
    turns__________  16
    victory_status_  resign
    winner_________  black
    increment_code_  5+10
    white_id_______  a-00
    white_rating___  1322
    black_id_______  skinnerua
    black_rating___  1261
    moves__________  d4 Nc6 e4 e5 f4 f6 dxe5 fxe5 fxe5 Nxe5 Qd4 Nc6 Qe5+ Nxe5 c4 Bb4+
    opening_eco____  B00
    opening_name___  Nimzowitsch Defense: Kennedy Variation
    opening_ply____  4
    """
    display_game(game_id=1)


if __name__ == "__main__":
    main()
