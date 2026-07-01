"""
Feature extraction for chess positions.
Converts FEN/Board states into PyTorch tensors suitable for policy/value networks.
"""
import chess
import torch
from typing import Union


# Piece-to-channel mapping:
# Channels 0-5: White p, n, b, r, k
# Channels 6-11: Black P, N, B, R, Q, K
_PIECE_CHANNELS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def fen_to_tensor(fen_or_board: Union[str, chess.Board], flatten: bool = False) -> torch.Tensor:
    """
    Convert a chess position to an 8x8x12 tensor (or flat 768-vector).
    Shape: (C, H, W) where C=12, H=8 (ranks), W=8 (files)
    """
    if isinstance(fen_or_board, str):
        board = chess.Board(fen_or_board)
    else:
        board = fen_or_board

    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            # Map to channel: white=0-5, black=6-11
            ch = _PIECE_CHANNELS[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            tensor[ch, rank, file] = 1.0

    if flatten:
        return tensor.view(768)
    return tensor


def get_legal_mask(fen_or_board: Union[str, chess.Board], policy_size: int = 4096) -> torch.Tensor:
    """
    Create a boolean mask for legal moves.
    Default: 64x64 = 4096 policy head (index = from_sq * 64 + to_sq).
    Note: Promotions & castling require extending to 4672 (73x64) or handling separately.
    """
    if isinstance(fen_or_board, str):
        board = chess.Board(fen_or_board)
    else:
        board = fen_or_board

    mask = torch.zeros(policy_size, dtype=torch.bool)
    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        idx = from_sq * 64 + to_sq
        if idx < policy_size:
            mask[idx] = True
    return mask
