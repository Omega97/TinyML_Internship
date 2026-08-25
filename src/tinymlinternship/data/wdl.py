"""STM Win/Draw/Loss probabilities for teacher labels and the student head.

Lc0 emits WDL permille from the **side to move**. Dual-POV concat is
``[STM ‖ opp]``, so the training target is the same STM triple
``(W, D, L)`` with ``W + D + L = 1``. White-POV expected reward is
``v = W - L`` if White to move, else ``L - W``.

Existing slices that only store White-POV ``value`` are inverted with the
maximum-entropy WDL that has ``W - L = v_stm`` (closed form on the simplex).
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

import chess

WDL_DIGITS = 3
WDL_EPS = 1e-12


def normalize_wdl(win: float, draw: float, loss: float) -> tuple[float, float, float]:
    w = max(float(win), 0.0)
    d = max(float(draw), 0.0)
    l = max(float(loss), 0.0)
    total = w + d + l
    if total <= WDL_EPS:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (w / total, d / total, l / total)


def round_wdl(win: float, draw: float, loss: float, *, digits: int = WDL_DIGITS) -> tuple[float, float, float]:
    """Round ``(W, D, L)`` to ``digits`` places so they still sum to 1 exactly at that scale."""
    w, d, l = normalize_wdl(win, draw, loss)
    scale = 10**digits
    parts = [int(round(w * scale)), int(round(d * scale)), int(round(l * scale))]
    parts[max(range(3), key=lambda i: (parts[i], -i))] += scale - sum(parts)
    for i, part in enumerate(parts):
        if part < 0:
            j = max(range(3), key=lambda k: parts[k])
            parts[j] += part
            parts[i] = 0
    return tuple(part / scale for part in parts)


def permille_to_stm_wdl(win: int, draw: int, loss: int) -> tuple[float, float, float]:
    return normalize_wdl(win / 1000.0, draw / 1000.0, loss / 1000.0)


def stm_value(win: float, draw: float, loss: float) -> float:
    _ = draw
    w, d, l = normalize_wdl(win, draw, loss)
    return w - l


def white_value_from_stm(
    board: chess.Board, win: float, draw: float, loss: float
) -> float:
    v = stm_value(win, draw, loss)
    return v if board.turn == chess.WHITE else -v


def stm_white_from_fen(fen: str) -> bool:
    parts = str(fen).split()
    return len(parts) < 2 or parts[1] != "b"


def value_to_wdl(v: float) -> tuple[float, float, float]:
    """Max-entropy ``(W, D, L)`` on the simplex with ``W - L = clip(v, -1, 1)``."""
    x = max(-1.0, min(1.0, float(v)))
    if x >= 1.0 - WDL_EPS:
        return (1.0, 0.0, 0.0)
    if x <= -1.0 + WDL_EPS:
        return (0.0, 0.0, 1.0)
    disc = 4.0 - 3.0 * x * x
    exp_l = (x + disc**0.5) / (2.0 * (1.0 - x))
    z = exp_l + 1.0 + 1.0 / exp_l
    return (exp_l / z, 1.0 / z, (1.0 / exp_l) / z)


def stm_wdl_from_white_value(fen: str, value: float) -> tuple[float, float, float]:
    v = float(value)
    v_stm = v if stm_white_from_fen(fen) else -v
    return value_to_wdl(v_stm)


def terminal_stm_wdl(board: chess.Board) -> tuple[float, float, float] | None:
    if board.is_checkmate():
        return (0.0, 0.0, 1.0)
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_threefold_repetition()
        or board.can_claim_fifty_moves()
    ):
        return (0.0, 1.0, 0.0)
    return None


def parse_wdl_cell(cell: Any) -> tuple[float, float, float] | None:
    if cell is None:
        return None
    if isinstance(cell, dict):
        if {"w", "d", "l"} <= set(cell) or {"win", "draw", "loss"} <= set(cell):
            win = cell.get("w", cell.get("win"))
            draw = cell.get("d", cell.get("draw"))
            loss = cell.get("l", cell.get("loss"))
            return normalize_wdl(float(win), float(draw), float(loss))
        return None
    if isinstance(cell, str):
        return None
    # Parquet/pandas stores wdl as numpy.ndarray, which is not a Sequence.
    if hasattr(cell, "tolist") and not isinstance(cell, (bytes, bytearray)):
        try:
            cell = cell.tolist()
        except (TypeError, ValueError):
            return None
    if isinstance(cell, Sequence) and not isinstance(cell, (bytes, bytearray)):
        if len(cell) < 3:
            return None
        try:
            return normalize_wdl(float(cell[0]), float(cell[1]), float(cell[2]))
        except (TypeError, ValueError):
            return None
    return None


def wdl_from_row(row: dict[str, Any]) -> tuple[float, float, float]:
    """Prefer an explicit STM ``wdl``; else invert White-POV ``value`` via the FEN."""
    parsed = parse_wdl_cell(row.get("wdl"))
    if parsed is not None:
        return parsed
    if all(key in row for key in ("w", "d", "l")):
        return normalize_wdl(row["w"], row["d"], row["l"])
    value = row.get("value")
    if value is None:
        raise ValueError("row has neither wdl nor value")
    return stm_wdl_from_white_value(str(row["fen"]), float(value))


def labeled_payload(fen: str, wdl: tuple[float, float, float], visits: int) -> dict[str, Any]:
    w, d, l = round_wdl(*wdl)
    v_stm = stm_value(w, d, l)
    value = v_stm if stm_white_from_fen(fen) else -v_stm
    return {
        "fen": fen,
        "wdl": [w, d, l],
        "value": round(float(value), WDL_DIGITS),
        "visits": int(visits),
    }


_JSON_FLOAT_RE = re.compile(r"-?(?:0|[1-9]\d*)\.\d+(?:[eE][+-]?\d+)?")


def dumps_labeled_json(payload: Any, *, digits: int = WDL_DIGITS) -> str:
    """``json.dumps`` with every float literal printed to ``digits`` decimal places."""
    text = json.dumps(payload, indent=2)
    spec = f".{int(digits)}f"

    def _fmt(match: re.Match[str]) -> str:
        return format(float(match.group(0)), spec)

    return _JSON_FLOAT_RE.sub(_fmt, text) + ("\n" if not text.endswith("\n") else "")
