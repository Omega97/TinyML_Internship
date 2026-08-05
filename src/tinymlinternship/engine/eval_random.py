"""
Untrained random value function — ACPL floor / null baseline.

Maps each position to a deterministic pseudo-random centipawn score from
White's perspective (same FEN → same score). No learned weights.
"""

from __future__ import annotations

import hashlib
import struct

import chess

from tinymlinternship.engine.eval_hce import MATE_SCORE

# Half-range of uniform integer scores in centipawns (White POV).
DEFAULT_CP_RANGE = 500
DEFAULT_SEED = 0x53415244  # "SARD"


def _digest_to_unit(digest: bytes) -> float:
    """Map first 8 bytes of a hash to [0, 1)."""
    (u64,) = struct.unpack(">Q", digest[:8])
    return u64 / 2**64


def evaluate_random(
    board: chess.Board,
    *,
    seed: int = DEFAULT_SEED,
    cp_range: int = DEFAULT_CP_RANGE,
) -> int:
    """
    Deterministic random eval in ``[-cp_range, +cp_range]`` (White POV).

    Terminal positions use the same mate / draw convention as HCE so search
    still prefers forced mates when it sees them at leaves.
    """
    if board.is_checkmate():
        # Side to move is mated → bad for STM → good for White if Black to move.
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_fifty_moves()
        or board.is_repetition()
    ):
        return 0

    # FEN without halfmove/fullmove clocks so pure position identity dominates.
    key = f"{seed}:{board.board_fen()} {board.turn} {board.castling_xfen()} {board.ep_square}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    u = _digest_to_unit(digest)
    # Symmetric integer in [-cp_range, +cp_range]
    score = int(round((2.0 * u - 1.0) * cp_range))
    return max(-cp_range, min(cp_range, score))
