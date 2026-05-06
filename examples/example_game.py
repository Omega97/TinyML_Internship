"""
Minimal script to display moves from a specific game in the Lichess dataset.
"""
import chess
import pandas as pd
from src.tinymlinternship.config.settings import LICHESS_CSV
from src.tinymlinternship.datasets.featurizer import fen_to_tensor, get_legal_mask


TEST_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


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


# ===== Examples =====


def example_make_move():
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


def example_fen_to_tensor():
    """
     torch.Size([12, 8, 8])
    """
    board = chess.Board(TEST_FEN)
    tensor_board = fen_to_tensor(board)

    print(tensor_board.shape)


def example_legal_moves():
    """
    tensor([False, False, False,  ..., False, False, False])
    torch.Size([4096])
    """
    board = chess.Board(TEST_FEN)
    mask = get_legal_mask(board)
    print(mask)
    print(mask.shape)


if __name__ == "__main__":
    # example_make_move()
    # example_fen_to_tensor()
    example_legal_moves()
