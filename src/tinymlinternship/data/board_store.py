"""Unified board+eval store: hash → {fen, stm_white, count, source, ply, rewards, wdl}."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import chess

BOARD_EVAL_DIR_NAME = "board_eval"
DATASET_JSON_NAME = "dataset.json"
# Per-source slim tables live in this subfolder; the join is the parquet in BOARD_EVAL_DIR_NAME.
FEN_VALUE_VISITS_DIR_NAME = "fen_value_visits"
FEN_VALUE_VISITS_JOINED_NAME = "fen_value_visits.parquet"

# Canonical per-hash fields (2026-08-06)
CANONICAL_TAGS: tuple[str, ...] = (
    "fen",
    "stm_white",
    "count",
    "source",
    "ply",
    "expected_reward",
    "wdl_win",
    "wdl_draw",
    "wdl_loss",
)


def board_hash(fen: str) -> str:
    """SHA-256 of EPD (board + STM + castling + EP); halfmove/fullmove excluded."""
    board = chess.Board(fen)
    return hashlib.sha256(board.epd().encode("utf-8")).hexdigest()


def _as_prob(x: float) -> float:
    """Lc0 WDL may be permille (0–1000) or already probability (0–1)."""
    v = float(x)
    if v > 1.0 + 1e-6:
        return v / 1000.0
    return v


def empty_record(fen: str, *, source: str | None = None, ply: int | None = None) -> dict[str, Any]:
    board = chess.Board(fen)
    rec: dict[str, Any] = {
        "fen": board.fen(),
        "stm_white": board.turn == chess.WHITE,
        "count": 0,
        "source": source or "unknown",
        "ply": int(ply) if ply is not None else 0,
    }
    return rec


def bump_count(
    store: dict[str, dict[str, Any]],
    fen: str,
    *,
    source: str | None = None,
    ply: int | None = None,
) -> str:
    """Increment observation count for a position (creates shell record if needed)."""
    key = board_hash(fen)
    if key not in store:
        store[key] = empty_record(fen, source=source, ply=ply)
    store[key]["count"] = int(store[key].get("count", 0)) + 1
    if source and store[key].get("source") in (None, "unknown"):
        store[key]["source"] = source
    if ply is not None and store[key].get("ply", 0) == 0:
        store[key]["ply"] = int(ply)
    return key


def merge_label(
    store: dict[str, dict[str, Any]],
    fen: str,
    *,
    expected_reward: float,
    wdl_win: float,
    wdl_draw: float,
    wdl_loss: float,
    source: str | None = None,
    ply: int | None = None,
) -> str:
    """
    Attach / average teacher labels on a hash.

    Running mean of expected_reward and WDL when the same hash is labeled more than once.
    Does **not** increment count (use :func:`bump_count` for multiplicity from extracts).
    """
    key = board_hash(fen)
    if key not in store:
        # Label-only observation (never seen in raw extracts): count once.
        store[key] = empty_record(fen, source=source, ply=ply)
        store[key]["count"] = 1
    elif int(store[key].get("count", 0)) <= 0:
        store[key]["count"] = 1

    rec = store[key]
    r = float(expected_reward)
    ww, wd, wl = _as_prob(wdl_win), _as_prob(wdl_draw), _as_prob(wdl_loss)
    n = int(rec.get("_n_labels", 0))
    if n == 0:
        rec["expected_reward"] = r
        rec["wdl_win"] = ww
        rec["wdl_draw"] = wd
        rec["wdl_loss"] = wl
        rec["_n_labels"] = 1
        rec["_reward_sum"] = r
        rec["_w_sum"] = ww
        rec["_d_sum"] = wd
        rec["_l_sum"] = wl
    else:
        n2 = n + 1
        rec["_reward_sum"] = float(rec["_reward_sum"]) + r
        rec["_w_sum"] = float(rec["_w_sum"]) + ww
        rec["_d_sum"] = float(rec["_d_sum"]) + wd
        rec["_l_sum"] = float(rec["_l_sum"]) + wl
        rec["_n_labels"] = n2
        rec["expected_reward"] = rec["_reward_sum"] / n2
        rec["wdl_win"] = rec["_w_sum"] / n2
        rec["wdl_draw"] = rec["_d_sum"] / n2
        rec["wdl_loss"] = rec["_l_sum"] / n2

    if source:
        rec["source"] = source
    if ply is not None:
        rec["ply"] = int(ply)
    # Keep fen/stm fresh from this fen
    board = chess.Board(fen)
    rec["fen"] = board.fen()
    rec["stm_white"] = board.turn == chess.WHITE
    return key


def finalize_record(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Drop helpers; keep only CANONICAL_TAGS (require label fields)."""
    out = {k: rec[k] for k in CANONICAL_TAGS if k in rec}
    # Ensure types
    out["fen"] = str(out["fen"])
    out["stm_white"] = bool(out["stm_white"])
    out["count"] = int(out.get("count", 1))
    out["source"] = str(out.get("source", "unknown"))
    out["ply"] = int(out.get("ply", 0))
    out["expected_reward"] = float(out["expected_reward"])
    out["wdl_win"] = float(out["wdl_win"])
    out["wdl_draw"] = float(out["wdl_draw"])
    out["wdl_loss"] = float(out["wdl_loss"])
    return out


def save_dataset_json(
    store: Mapping[str, Mapping[str, Any]],
    path: Path | str,
    *,
    meta: Mapping[str, Any] | None = None,
    labeled_only: bool = True,
) -> Path:
    """
    Write unified dataset::

        { "meta": {...}, "data": { "<hash>": { canonical tags }, ... } }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, Any]] = {}
    for h, rec in store.items():
        if labeled_only and "expected_reward" not in rec:
            continue
        data[h] = finalize_record(rec)

    payload = {
        "meta": dict(meta or {}),
        "data": data,
    }
    payload["meta"]["n_positions"] = len(data)
    payload["meta"]["canonical_tags"] = list(CANONICAL_TAGS)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_dataset_json(path: Path | str) -> dict[str, dict[str, Any]]:
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "data" in doc and isinstance(doc["data"], dict):
        return doc["data"]
    return doc


def _slim_shell(fen: str) -> dict[str, Any]:
    board = chess.Board(fen)
    return {
        "fen": board.fen(),
        "visits": 0,
        "_value_sum": 0.0,
        "_value_n": 0,
    }


def bump_visits(store: dict[str, dict[str, Any]], fen: str) -> str:
    """Count one observation of ``fen`` (EPD hash; halfmove/fullmove ignored)."""
    key = board_hash(fen)
    if key not in store:
        store[key] = _slim_shell(fen)
    store[key]["visits"] = int(store[key]["visits"]) + 1
    return key


def add_teacher_value(store: dict[str, dict[str, Any]], fen: str, value: float) -> str:
    """Attach / average a teacher scalar on ``fen`` (does not increment visits)."""
    key = board_hash(fen)
    if key not in store:
        store[key] = _slim_shell(fen)
    rec = store[key]
    rec["_value_sum"] = float(rec["_value_sum"]) + float(value)
    rec["_value_n"] = int(rec["_value_n"]) + 1
    rec["fen"] = chess.Board(fen).fen()
    return key


def fen_value_visits_source_filename(source: str) -> str:
    """``fen_value_visits_<source>.parquet`` (lowercase slug)."""
    slug_chars: list[str] = []
    for char in source.strip().lower():
        if char.isalnum() or char in "-_":
            slug_chars.append(char)
        else:
            slug_chars.append("_")
    slug = "".join(slug_chars).strip("_") or "unknown"
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"fen_value_visits_{slug}.parquet"


def slim_fen_value_visits(
    store: Mapping[str, Mapping[str, Any]],
    *,
    labeled_only: bool = True,
) -> list[dict[str, Any]]:
    """Rows with columns ``fen``, ``value`` (teacher), ``visits`` (observation count)."""
    rows: list[dict[str, Any]] = []
    for rec in store.values():
        n_lab = int(rec.get("_value_n", 0))
        if labeled_only and n_lab <= 0:
            continue
        visits = int(rec.get("visits", 0))
        if visits <= 0:
            visits = max(n_lab, 1)
        rows.append(
            {
                "fen": str(rec["fen"]),
                "value": float(rec["_value_sum"]) / n_lab,
                "visits": visits,
            }
        )
    return rows
