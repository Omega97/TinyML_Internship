"""Resolve engine eval backends by name."""

from __future__ import annotations

from pathlib import Path

import chess

from tinymlinternship.engine.eval_hce import evaluate_hce
from tinymlinternship.engine.eval_lc0 import evaluate_lc0_teacher
from tinymlinternship.engine.eval_nnue import evaluate_nnue
from tinymlinternship.engine.search import EvalFn

EVAL_CHOICES = ("hce", "nnue", "lc0")


def make_eval_fn(
    name: str,
    *,
    nnue_checkpoint: Path | str | None = None,
) -> EvalFn:
    if name == "hce":
        return evaluate_hce
    if name == "nnue":
        checkpoint = nnue_checkpoint

        def _nnue_eval(board: chess.Board) -> int:
            return evaluate_nnue(board, checkpoint=checkpoint)

        return _nnue_eval
    if name == "lc0":
        return evaluate_lc0_teacher
    raise ValueError(f"unknown eval backend: {name!r} (choices: {', '.join(EVAL_CHOICES)})")