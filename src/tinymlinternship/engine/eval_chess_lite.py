"""chess_lite (HF) value evaluation for shallow search."""

from __future__ import annotations

from pathlib import Path

import chess
import torch
import torch.nn as nn

from tinymlinternship.config.settings import CHESS_LITE_WEIGHTS
from tinymlinternship.engine.eval_lc0 import expected_reward_to_cp
from tinymlinternship.engine.eval_hce import MATE_SCORE

_PIECE_CHANNELS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """15-channel encoding: 12 piece planes + STM + last-move from/to."""
    tensor = torch.zeros(15, 8, 8, dtype=torch.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        ch = _PIECE_CHANNELS[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
        tensor[ch, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.move_stack:
        last = board.move_stack[-1]
        tensor[13, chess.square_rank(last.from_square), chess.square_file(last.from_square)] = 1.0
        tensor[14, chess.square_rank(last.to_square), chess.square_file(last.to_square)] = 1.0

    return tensor


class BossChessNet(nn.Module):
    """Architecture inferred from satana123/chess_lite checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(15, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.policy_head(features), self.value_head(features)


def _remap_state_dict(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint omits Flatten layers; shift Linear indices in policy/value heads."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key.startswith("policy_head.2"):
            remapped[key.replace("policy_head.2", "policy_head.3", 1)] = value
        elif key.startswith("policy_head.4"):
            remapped[key.replace("policy_head.4", "policy_head.5", 1)] = value
        elif key.startswith("value_head.2"):
            remapped[key.replace("value_head.2", "value_head.3", 1)] = value
        elif key.startswith("value_head.4"):
            remapped[key.replace("value_head.4", "value_head.5", 1)] = value
        else:
            remapped[key] = value
    return remapped


def expected_reward_white(board: chess.Board, value_stm: float) -> float:
    return value_stm if board.turn == chess.WHITE else -value_stm


class ChessLiteEvaluator:
    """Loads chess_lite.pth and exposes centipawn-like eval for negamax."""

    def __init__(self, weights: str | Path | None = None) -> None:
        path = Path(weights or CHESS_LITE_WEIGHTS)
        if not path.exists():
            raise FileNotFoundError(
                f"chess_lite weights not found: {path} — run scripts/download_hf_teacher.py"
            )
        self.weights = path
        self._model = BossChessNet()
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(_remap_state_dict(state))
        self._model.eval()

    def evaluate_expected_reward(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0 if board.turn == chess.WHITE else 1.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            return 0.0

        with torch.no_grad():
            x = board_to_tensor(board).unsqueeze(0)
            _, value = self._model(x)
        return expected_reward_white(board, float(value.item()))

    def evaluate_cp(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        return expected_reward_to_cp(self.evaluate_expected_reward(board))


_evaluator_singleton: ChessLiteEvaluator | None = None


def get_chess_lite_evaluator() -> ChessLiteEvaluator:
    global _evaluator_singleton
    if _evaluator_singleton is None:
        _evaluator_singleton = ChessLiteEvaluator()
    return _evaluator_singleton


def evaluate_chess_lite(board: chess.Board) -> int:
    """Static eval in centipawn-like units (White = positive) via chess_lite value head."""
    return get_chess_lite_evaluator().evaluate_cp(board)