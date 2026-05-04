# tests/test_policy_inference.py
"""
This program creates a minimal random policy
"""
import torch
import chess
from src.tinymlinternship.datasets.featurizer import fen_to_tensor, get_legal_mask


TEST_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"


class TinyPolicy(torch.nn.Module):
    """
    Dummy Policy Network for I/O Validation
    """
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(16 * 8 * 8, 4096)  # all possible moves

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(x.view(x.size(0), -1))


def test_fen_to_tensor_shape_and_values():
    """Verify featurizer outputs correct shape, dtype, and piece count."""
    tensor = fen_to_tensor(TEST_FEN)

    assert tensor.shape == (12, 8, 8), f"Expected (12, 8, 8), got {tensor.shape}"
    assert tensor.dtype == torch.float32
    assert tensor.ndim == 3

    # Verify active channels match pieces on board
    board = chess.Board(TEST_FEN)
    active_pieces = tensor.sum().item()
    assert active_pieces == len(board.piece_map()), "Tensor active count != pieces on board"


def test_get_legal_mask_shape_and_validity():
    """Verify legal mask matches board's legal move count and shape."""
    mask = get_legal_mask(TEST_FEN)

    assert mask.shape == (4096,), f"Expected (4096,), got {mask.shape}"
    assert mask.dtype == torch.bool

    board = chess.Board(TEST_FEN)
    legal_count = len(list(board.legal_moves))
    assert mask.sum().item() == legal_count, f"Mask active bits ({mask.sum()}) != legal moves ({legal_count})"


def test_full_inference_pipeline():
    """End-to-end test: FEN → tensor → model → masked softmax → valid move output."""
    model = TinyPolicy().eval()

    x = fen_to_tensor(TEST_FEN).unsqueeze(0)  # (1, 12, 8, 8)
    mask = get_legal_mask(TEST_FEN)  # (4096,)

    with torch.no_grad():
        logits = model(x).squeeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)

    top_idx = probs.argmax().item()
    from_sq, to_sq = divmod(top_idx, 64)

    # 1. Top move must be legal
    assert mask[top_idx].item() is True, "Argmax move is not in legal mask"
    # 2. Probabilities must sum to ~1.0
    assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities do not sum to 1"
    # 3. Squares must be valid UCI indices
    assert 0 <= from_sq < 64 and 0 <= to_sq < 64, "Invalid square indices in top move"
    # 4. Verify move string matches chess library expectation
    predicted_move = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq]
    board = chess.Board(TEST_FEN)
    legal_uci = [move.uci() for move in board.legal_moves]
    assert predicted_move in legal_uci, f"Predicted UCI {predicted_move} not in legal moves"
