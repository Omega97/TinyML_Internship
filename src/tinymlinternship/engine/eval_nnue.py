"""
Bucketed NNUE static evaluation for the SARDINE search stack.

Loads a PyTorch checkpoint (``BucketedNNUE``) and maps tanh expected-reward
output to centipawn-like scores from White's perspective (same scale as Lc0).
"""

from __future__ import annotations

import atexit
from pathlib import Path

import chess
import torch

from tinymlinternship.config.settings import NNUE_CHECKPOINT_DEFAULT
from tinymlinternship.engine.eval_hce import MATE_SCORE
from tinymlinternship.engine.eval_lc0 import CP_SCALE, expected_reward_to_cp
from tinymlinternship.features import FEATURE_DIM, bucket_id, encode_dual
from tinymlinternship.nnue import BucketedNNUE, indices_to_binary


def stm_reward_to_white(board: chess.Board, reward_stm: float) -> float:
    """Expected reward from side to move → White's perspective."""
    return reward_stm if board.turn == chess.WHITE else -reward_stm


class NnueEvaluator:
    """Loads and runs a trained ``BucketedNNUE`` checkpoint."""

    def __init__(
        self,
        checkpoint: Path | str = NNUE_CHECKPOINT_DEFAULT,
        *,
        device: str = "cpu",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self._model: BucketedNNUE | None = None
        self.hidden_dim = 128

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"NNUE checkpoint not found: {self.checkpoint} — run scripts/train_nnue.py"
            )

        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        self.hidden_dim = int(payload.get("hidden_dim", 128))
        model = BucketedNNUE(hidden_dim=self.hidden_dim)
        l1_in = payload["model_state_dict"]["l1.weight"].shape[1]
        if l1_in != FEATURE_DIM:
            raise ValueError(
                f"checkpoint L1 input {l1_in} != encoder FEATURE_DIM {FEATURE_DIM}; "
                "retrain with scripts/train_nnue.py"
            )
        model.load_state_dict(payload["model_state_dict"])
        model.to(self.device)
        model.eval()
        self._model = model

    @property
    def model(self) -> BucketedNNUE:
        self.load()
        assert self._model is not None
        return self._model

    @torch.inference_mode()
    def evaluate_expected_reward_stm(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0 if board.turn == chess.WHITE else 1.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            return 0.0

        white_idx, black_idx = encode_dual(board)
        white = indices_to_binary(white_idx).unsqueeze(0).to(self.device)
        black = indices_to_binary(black_idx).unsqueeze(0).to(self.device)
        bucket = torch.tensor([bucket_id(board)], dtype=torch.long, device=self.device)
        stm_white = torch.tensor([board.turn == chess.WHITE], dtype=torch.bool, device=self.device)
        return float(self.model(white, black, bucket, stm_white).item())

    def evaluate_expected_reward_white(self, board: chess.Board) -> float:
        return stm_reward_to_white(board, self.evaluate_expected_reward_stm(board))

    def evaluate_cp(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        return expected_reward_to_cp(self.evaluate_expected_reward_white(board))


_evaluator_singleton: NnueEvaluator | None = None
_evaluator_checkpoint: Path | None = None


def get_nnue_evaluator(checkpoint: Path | str | None = None) -> NnueEvaluator:
    global _evaluator_singleton, _evaluator_checkpoint
    path = Path(checkpoint or NNUE_CHECKPOINT_DEFAULT)
    if _evaluator_singleton is None or _evaluator_checkpoint != path:
        if _evaluator_singleton is not None:
            _evaluator_singleton = None
        _evaluator_singleton = NnueEvaluator(path)
        _evaluator_singleton.load()
        _evaluator_checkpoint = path
        atexit.register(_close_nnue_singleton)
    return _evaluator_singleton


def _close_nnue_singleton() -> None:
    global _evaluator_singleton, _evaluator_checkpoint
    _evaluator_singleton = None
    _evaluator_checkpoint = None


def evaluate_nnue(board: chess.Board, *, checkpoint: Path | str | None = None) -> int:
    """Static eval in centipawn-like units (White = positive) via trained NNUE."""
    return get_nnue_evaluator(checkpoint).evaluate_cp(board)