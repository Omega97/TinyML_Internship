import chess


# todo find better character for "."
CHARACTERS = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',  # White
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',  # Black
        '.': '·',
    }


def print_board(board: chess.Board) -> str:
    """Print board with emoji instead of letters."""
    trans_table = str.maketrans(CHARACTERS)
    return str(board).translate(trans_table)
