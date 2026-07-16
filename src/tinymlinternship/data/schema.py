"""Uniform training-set schema (see ASSETS.md §Ideal final training set)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import chess
import pandas as pd

from tinymlinternship.features.bucket import NUM_BUCKETS

# Pre-label (after PGN / Lc0 extract)
PRELABEL_COLUMNS: tuple[str, ...] = (
    "fen",
    "bucket_id",
    "piece_count",
    "has_queen",
    "stm_white",
    "ply",
    "source",
    "game_id",
)

# Production labeled rows (merge / train)
LABELED_REQUIRED_COLUMNS: tuple[str, ...] = PRELABEL_COLUMNS + (
    "expected_reward",
)

LABELED_RECOMMENDED_COLUMNS: tuple[str, ...] = (
    "wdl_win",
    "wdl_draw",
    "wdl_loss",
    "teacher_network",
)

VALID_SOURCES = frozenset({"lichess", "lc0"})

LABEL_FORMULA = "expected_reward = (W - L) / 1000 from STM WDL, flipped to White POV"


def stm_white_from_fen(fen: str) -> bool:
    return chess.Board(fen).turn == chess.WHITE


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ensure_prelabel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a pre-label frame to ASSETS columns (best-effort renames)."""
    out = df.copy()
    if "bucket_id" not in out.columns and "bucket" in out.columns:
        out = out.rename(columns={"bucket": "bucket_id"})
    if "stm_white" not in out.columns and "fen" in out.columns:
        out["stm_white"] = out["fen"].map(stm_white_from_fen)
    if "has_queen" not in out.columns and "fen" in out.columns:
        from tinymlinternship.features.bucket import has_queen

        out["has_queen"] = out["fen"].map(lambda f: bool(has_queen(f)))
    if "piece_count" not in out.columns and "fen" in out.columns:
        from tinymlinternship.features.bucket import piece_count

        out["piece_count"] = out["fen"].map(lambda f: int(piece_count(f)))
    if "bucket_id" not in out.columns and "fen" in out.columns:
        from tinymlinternship.features.bucket import bucket_id

        out["bucket_id"] = out["fen"].map(lambda f: int(bucket_id(f)))
    if "ply" not in out.columns:
        out["ply"] = -1
    if "source" not in out.columns:
        out["source"] = "unknown"
    if "game_id" not in out.columns:
        out["game_id"] = [f"row_{i}" for i in range(len(out))]
    missing = [c for c in PRELABEL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"cannot ensure prelabel columns; missing {missing}")
    return out


def validate_rewards_series(series: pd.Series) -> None:
    bad = series[(series < -1.0) | (series > 1.0)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} label(s) outside [-1, +1]; first={float(bad.iloc[0])}"
        )


def validate_labeled_frame(df: pd.DataFrame, *, require_source: bool = True) -> None:
    missing = [c for c in LABELED_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"labeled frame missing required columns: {missing}")
    validate_rewards_series(df["expected_reward"])
    if require_source:
        bad_src = set(df["source"].astype(str).unique()) - VALID_SOURCES
        if bad_src:
            raise ValueError(f"invalid source values: {sorted(bad_src)}")


def bucket_histogram(df: pd.DataFrame, column: str = "bucket_id") -> dict[str, int]:
    counts = Counter(int(x) for x in df[column].tolist())
    return {str(b): int(counts.get(b, 0)) for b in range(NUM_BUCKETS)}


def split_by_game(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.05,
    seed: int = 42,
    game_col: str = "game_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/val split by game id (no position leakage across games)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if game_col not in df.columns:
        raise ValueError(f"missing {game_col!r} for game-level split")
    games = df[game_col].astype(str).unique().tolist()
    rng = __import__("random").Random(seed)
    rng.shuffle(games)
    n_val = max(1, int(round(len(games) * val_fraction))) if len(games) > 1 else 0
    if len(games) == 1:
        n_val = 0
    val_games = set(games[:n_val])
    train_games = set(games[n_val:])
    if not train_games and games:
        # Keep at least one game in train
        train_games = {games[0]}
        val_games = set(games[1:])
    train = df[df[game_col].astype(str).isin(train_games)].reset_index(drop=True)
    val = df[df[game_col].astype(str).isin(val_games)].reset_index(drop=True)
    return train, val


def build_manifest(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    sources: Iterable[str],
    teacher_network: str,
    teacher_network_sha256: str | None = None,
    teacher_binary: str | None = None,
    filters: dict[str, Any] | None = None,
    sample_rate: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_df = pd.concat([train, val], ignore_index=True) if len(val) else train
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_formula": LABEL_FORMULA,
        "teacher_network": teacher_network,
        "teacher_network_sha256": teacher_network_sha256,
        "teacher_binary": teacher_binary,
        "sources": list(sources),
        "filters": filters or {},
        "sample_rate": sample_rate or {},
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "total_rows": int(len(all_df)),
        "train_games": int(train["game_id"].nunique()) if len(train) else 0,
        "val_games": int(val["game_id"].nunique()) if len(val) else 0,
        "bucket_counts_train": bucket_histogram(train) if len(train) else {},
        "bucket_counts_val": bucket_histogram(val) if len(val) else {},
        "bucket_counts_all": bucket_histogram(all_df) if len(all_df) else {},
        "expected_reward_min": float(all_df["expected_reward"].min()) if len(all_df) else None,
        "expected_reward_max": float(all_df["expected_reward"].max()) if len(all_df) else None,
        "expected_reward_mean": float(all_df["expected_reward"].mean()) if len(all_df) else None,
        "schema_required": list(LABELED_REQUIRED_COLUMNS),
        "schema_recommended": list(LABELED_RECOMMENDED_COLUMNS),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
