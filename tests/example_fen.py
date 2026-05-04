import chess


TEST_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def apply_move_to_fen(fen: str, move_uci: str) -> str:
    """Apply a UCI move to a FEN and return the updated FEN."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {move_uci} for position {fen}")

    board.push(move)
    return board.fen()


# ===== Examples =====


def example_make_moves():
    """
     rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
    r n b q k b n r
    p p p p p p p p
    . . . . . . . .
    . . . . . . . .
    . . . . P . . .
    . . . . . . . .
    P P P P . P P P
    R N B Q K B N R

     rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2
    r n b q k b n r
    p p p p . p p p
    . . . . . . . .
    . . . . p . . .
    . . . . P . . .
    . . . . . . . .
    P P P P . P P P
    R N B Q K B N R
    """

    new_fen = apply_move_to_fen(TEST_FEN, "e7e5")

    for fen in TEST_FEN, new_fen:
        print(f"\n {fen}")
        print(chess.Board(fen))


def example_legal_moves():
    """
    20 legal moves:
    g8h6, g8f6, b8c6, b8a6, h7h6, g7g6, f7f6, e7e6, d7d6, c7c6, b7b6, a7a6, h7h5, g7g5, f7f5, e7e5, d7d5, c7c5, b7b5, a7a5
    """
    board = chess.Board(TEST_FEN)
    legal_moves = list(board.legal_moves)

    print(f"\n{len(legal_moves)} legal moves:")
    print(", ".join([str(move) for move in legal_moves]))


if __name__ == '__main__':
    example_make_moves()
    # example_legal_moves()
