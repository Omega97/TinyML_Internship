"""
scripts/play_random_game.py
Run a chess game where both sides play completely random legal moves.
Displays the board state after every move. Useful for baseline testing,
data generation, and verifying your featurizer pipeline.
"""
import chess
import random
import time
from typing import Optional


def random_move(board: chess.Board) -> chess.Move:
    """Return a uniformly random legal move."""
    return random.choice(list(board.legal_moves))


def play_random_game(max_moves: int = 100, delay: float = 0.) -> Optional[str]:
    """
    Play a game with random moves, printing the board after each ply.
    Returns the final FEN, or None if game ends before max_moves.
    """
    board = chess.Board()
    move_count = 0

    print("🎲 Starting Random Chess Game 🎲")
    print(board)
    print("=" * 45)

    while not board.is_game_over() and move_count < max_moves:
        # 1. Pick move
        move = random_move(board)
        board.push(move)
        move_count += 1

        # 2. Display state
        turn_str = "White" if board.turn == chess.WHITE else "Black"
        print(f"\n🔹 Move {move_count} ({turn_str}): {move.uci()}")
        print(board)
        print("=" * 45)

        # Optional delay for console readability
        if delay > 0:
            time.sleep(delay)

    # 3. Game termination
    print("\n🏁 GAME OVER")
    if board.is_game_over():
        result = board.result()
        print(f"Result: {result}")
        if result == "1-0":
            print("🥇 White wins!")
        elif result == "0-1":
            print("🥇 Black wins!")
        else:
            print("🤝 Draw!")
    else:
        print(f"⏹️ Stopped after {max_moves} moves (limit reached).")

    print(f"📜 Final FEN: {board.fen()}")
    return board.fen()


if __name__ == "__main__":
    play_random_game(max_moves=200, delay=0.)
