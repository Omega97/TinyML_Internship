"""
NNUE static evaluation for the SARDINE search stack.

Loads a PyTorch checkpoint (F3 ``SingleHeadNNUE`` or legacy ``BucketedNNUE``)
and maps tanh expected-reward output to centipawn-like scores from White's
perspective (same scale as Lc0).
"""

from __future__ import annotations

import atexit
from pathlib import Path

import chess
import torch
import torch.nn as nn

from tinymlinternship.config.settings import NNUE_CHECKPOINT_DEFAULT
from tinymlinternship.engine.eval_hce import MATE_SCORE
from tinymlinternship.engine.eval_lc0 import expected_reward_to_cp
from tinymlinternship.features import FEATURE_DIM, bucket_id, encode_dual
from tinymlinternship.nnue import (
    Architecture,
    build_nnue,
    indices_to_binary,
    infer_architecture,
)


def stm_reward_to_white(board: chess.Board, reward_stm: float) -> float:
    """Expected reward from side to move → White's perspective."""
    return reward_stm if board.turn == chess.WHITE else -reward_stm


class NnueEvaluator:
    """Loads and runs a trained single-head or bucketed NNUE checkpoint."""

    def __init__(
        self,
        checkpoint: Path | str = NNUE_CHECKPOINT_DEFAULT,
        *,
        device: str = "cpu",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self._model: nn.Module | None = None
        self.hidden_dim = 128
        self.hidden2_dim = 256
        self.architecture: Architecture = "single_head"

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"NNUE checkpoint not found: {self.checkpoint} — run scripts/train_nnue.py"
            )

        # weights_only=False: payload includes architecture strings / scalars.
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        # Support both full payload (best.pt) and raw state_dict (old last.pt).
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state = payload["model_state_dict"]
            self.hidden_dim = int(payload.get("hidden_dim", 128))
            self.hidden2_dim = int(payload.get("hidden2_dim", 256))
            arch_raw = payload.get("architecture")
            inferred = infer_architecture(state)
            # Prefer keys in the file when present; fall back to state_dict shape.
            if arch_raw in ("single_head", "bucketed", "dual_hidden"):
                self.architecture = arch_raw  # type: ignore[assignment]
            else:
                self.architecture = inferred
            # If metadata disagrees with weights (e.g. l2 present), trust weights.
            if inferred == "dual_hidden" and self.architecture != "dual_hidden":
                self.architecture = "dual_hidden"
            num_buckets = int(payload.get("num_buckets", 8))
            if "l2.weight" in state:
                self.hidden2_dim = int(state["l2.weight"].shape[0])
            if "l1.weight" in state:
                self.hidden_dim = int(state["l1.weight"].shape[0])
        else:
            state = payload
            self.hidden_dim = 128
            self.hidden2_dim = 256
            self.architecture = infer_architecture(state)
            num_buckets = 8
            if "l1.weight" in state:
                self.hidden_dim = int(state["l1.weight"].shape[0])
            if "l2.weight" in state:
                self.hidden2_dim = int(state["l2.weight"].shape[0])

        l1_in = state["l1.weight"].shape[1]
        if l1_in != FEATURE_DIM:
            raise ValueError(
                f"checkpoint L1 input {l1_in} != encoder FEATURE_DIM {FEATURE_DIM}; "
                "retrain with scripts/train_nnue.py"
            )
        model = build_nnue(
            self.architecture,
            hidden_dim=self.hidden_dim,
            hidden2_dim=self.hidden2_dim,
            num_buckets=num_buckets,
        )
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self._model = model

    @property
    def model(self) -> nn.Module:
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