"""Lc0 chunk → filtered, bucket-aware position tables (labels added separately via Lc0 UCI)."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chess
import pandas as pd

from tinymlinternship.data.lc0_parser import (
    V6_RECORD_SIZE,
    Lc0Position,
    _iter_records,
    decode_v6_record,
    read_chunk,
)
from tinymlinternship.features.bucket import NUM_BUCKETS, bucket_id, has_queen, piece_count

# Blueprint «games ≥ 16 moves» in plot_piece_count_distribution uses min_moves on
# the Kaggle moves column (= 16 full moves per side pair = 32 half-moves / ply).
DEFAULT_MIN_PLY = 32

# Fase 4 (diversity): skip redundant opening plies. Fase 5 (balance): quota N/8
# per bucket. Complementary — see NOTES/Lc0 preprocessing pipeline.md §Fase 4–5.
# Bucket 7 (p=32): first capture often at ply 16–30 — relax opening skip only here.
DEFAULT_BUCKET_MIN_PLY: dict[int, int] = {7: 8}

INVARIANCE_DELETE_BIT = 0x40
INVARIANCE_ADJUDICATED_BIT = 0x20


@dataclass(frozen=True)
class ParsedPosition:
    """One training candidate after parse + legality check."""

    fen: str
    bucket: int  # internal; exported as bucket_id in positions_to_dataframe
    ply: int
    game_id: str
    chunk_path: str
    piece_count: int
    has_queen: bool
    stm_white: bool
    source: str
    best_q: float
    root_q: float
    plies_left: float
    visits: int
    adjudicated: bool


@dataclass
class FilterConfig:
    min_ply: int = DEFAULT_MIN_PLY
    bucket_min_ply: dict[int, int] = field(default_factory=lambda: dict(DEFAULT_BUCKET_MIN_PLY))
    min_visits: int = 0
    skip_delete_flag: bool = True

    def min_ply_for_bucket(self, bucket: int) -> int:
        return self.bucket_min_ply.get(bucket, self.min_ply)


@dataclass
class SampleConfig:
    total_positions: int = 10_000
    val_fraction: float = 0.05
    seed: int = 42


@dataclass
class PipelineStats:
    chunks_scanned: int = 0
    records_seen: int = 0
    records_skipped_format: int = 0
    records_skipped_decode: int = 0
    records_skipped_delete: int = 0
    illegal_board: int = 0
    filtered_ply: int = 0
    filtered_visits: int = 0
    positions_valid: int = 0
    games_seen: int = 0
    bucket_counts_raw: Counter = field(default_factory=Counter)
    bucket_counts_after_filter: Counter = field(default_factory=Counter)


def _should_skip_record(invariance_info: int, *, skip_delete: bool) -> bool:
    if skip_delete and (invariance_info & INVARIANCE_DELETE_BIT):
        return True
    return False


def iter_chunk_positions(
    chunk_path: Path,
    *,
    stats: PipelineStats | None = None,
) -> Iterator[tuple[str, int, Lc0Position]]:
    """
    Yield (game_id, ply, position) from one chunk.

    New game when ``plies_left`` increases vs previous record (Lc0 chunk convention).
    """
    data = read_chunk(chunk_path)
    chunk_key = chunk_path.as_posix()
    game_idx = 0
    ply = 0
    prev_plies_left: float | None = None

    for record in _iter_records(data):
        if stats is not None:
            stats.records_seen += 1
        try:
            pos = decode_v6_record(record)
        except ValueError:
            if stats is not None:
                stats.records_skipped_decode += 1
            continue

        if _should_skip_record(pos.invariance_info, skip_delete=True):
            if stats is not None:
                stats.records_skipped_delete += 1
            continue

        if prev_plies_left is not None and pos.plies_left > prev_plies_left:
            game_idx += 1
            ply = 0
            if stats is not None:
                stats.games_seen += 1

        game_id = f"{chunk_path.stem}:{game_idx}"
        yield game_id, ply, pos
        ply += 1
        prev_plies_left = pos.plies_left

    if stats is not None and prev_plies_left is not None:
        stats.games_seen += 1


def parse_position(
    game_id: str,
    ply: int,
    pos: Lc0Position,
    chunk_path: Path,
    *,
    stats: PipelineStats | None = None,
) -> ParsedPosition | None:
    try:
        board = chess.Board(pos.fen)
    except ValueError:
        if stats is not None:
            stats.illegal_board += 1
        return None
    if not board.is_valid():
        if stats is not None:
            stats.illegal_board += 1
        return None

    b = bucket_id(board)
    if stats is not None:
        stats.bucket_counts_raw[b] += 1
        stats.positions_valid += 1

    return ParsedPosition(
        fen=board.fen(),
        bucket=b,
        ply=ply,
        game_id=game_id,
        chunk_path=chunk_path.as_posix(),
        piece_count=piece_count(board),
        has_queen=bool(has_queen(board)),
        stm_white=board.turn == chess.WHITE,
        source="lc0",
        best_q=pos.best_q,
        root_q=pos.root_q,
        plies_left=pos.plies_left,
        visits=pos.visits,
        adjudicated=bool(pos.invariance_info & INVARIANCE_ADJUDICATED_BIT),
    )


def passes_filter(parsed: ParsedPosition, cfg: FilterConfig, *, stats: PipelineStats | None = None) -> bool:
    min_ply = cfg.min_ply_for_bucket(parsed.bucket)
    if parsed.ply < min_ply:
        if stats is not None:
            stats.filtered_ply += 1
        return False
    if parsed.visits < cfg.min_visits:
        if stats is not None:
            stats.filtered_visits += 1
        return False
    if stats is not None:
        stats.bucket_counts_after_filter[parsed.bucket] += 1
    return True


def iter_filtered_positions(
    chunk_paths: list[Path],
    cfg: FilterConfig,
    *,
    max_records: int | None = None,
) -> Iterator[ParsedPosition]:
    stats = PipelineStats()
    yielded = 0
    for chunk_path in chunk_paths:
        stats.chunks_scanned += 1
        for game_id, ply, pos in iter_chunk_positions(chunk_path, stats=stats):
            parsed = parse_position(game_id, ply, pos, chunk_path, stats=stats)
            if parsed is None:
                continue
            if passes_filter(parsed, cfg, stats=stats):
                yield parsed
                yielded += 1
                if max_records is not None and yielded >= max_records:
                    return


def collect_bucket_survival(
    chunk_paths: list[Path],
    min_ply_values: list[int],
    *,
    bucket_min_ply: dict[int, int] | None = None,
    max_records: int = 50_000,
) -> pd.DataFrame:
    """Count per-bucket positions surviving each global min_ply (with bucket overrides)."""
    rows: list[dict] = []
    positions: list[ParsedPosition] = []
    for chunk in chunk_paths:
        for game_id, ply, pos in iter_chunk_positions(chunk):
            parsed = parse_position(game_id, ply, pos, chunk)
            if parsed is not None:
                positions.append(parsed)
            if len(positions) >= max_records:
                break
        if len(positions) >= max_records:
            break

    bucket_overrides = bucket_min_ply or DEFAULT_BUCKET_MIN_PLY
    for min_ply in min_ply_values:
        counts = Counter()
        for p in positions:
            threshold = bucket_overrides.get(p.bucket, min_ply)
            if p.ply >= threshold:
                counts[p.bucket] += 1
        for bucket in range(NUM_BUCKETS):
            rows.append(
                {
                    "min_ply_global": min_ply,
                    "bucket": bucket,
                    "count": counts[bucket],
                    "bucket_min_ply_used": bucket_overrides.get(bucket, min_ply),
                }
            )
    return pd.DataFrame(rows)


def stratified_sample(
    positions: list[ParsedPosition],
    cfg: SampleConfig,
) -> tuple[list[ParsedPosition], list[ParsedPosition]]:
    """Reservoir-style stratified split into train / val per bucket."""
    rng = random.Random(cfg.seed)
    by_bucket: dict[int, list[ParsedPosition]] = defaultdict(list)
    for p in positions:
        by_bucket[p.bucket].append(p)

    target_total = min(cfg.total_positions, len(positions))
    per_bucket = max(1, target_total // NUM_BUCKETS)
    val_per_bucket = max(0, int(per_bucket * cfg.val_fraction))

    train: list[ParsedPosition] = []
    val: list[ParsedPosition] = []
    for bucket in range(NUM_BUCKETS):
        pool = by_bucket[bucket]
        rng.shuffle(pool)
        take = min(per_bucket, len(pool))
        val_take = min(val_per_bucket, max(0, take - 1))
        val.extend(pool[:val_take])
        train.extend(pool[val_take:take])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def positions_to_dataframe(positions: list[ParsedPosition]) -> pd.DataFrame:
    """Export ASSETS pre-label schema + optional Lc0 debug fields."""
    rows = []
    for p in positions:
        rows.append(
            {
                "fen": p.fen,
                "bucket_id": p.bucket,
                "piece_count": p.piece_count,
                "has_queen": p.has_queen,
                "stm_white": p.stm_white,
                "ply": p.ply,
                "source": p.source,
                "game_id": p.game_id,
                # optional Lc0 metadata (not training targets)
                "chunk_path": p.chunk_path,
                "best_q": p.best_q,
                "root_q": p.root_q,
                "plies_left": p.plies_left,
                "visits": p.visits,
                "adjudicated": p.adjudicated,
            }
        )
    return pd.DataFrame(rows)


def discover_chunks(chunks_dir: Path, *, limit: int | None = None) -> list[Path]:
    paths = sorted(chunks_dir.rglob("*.gz"))
    if limit is not None:
        return paths[:limit]
    return paths


def save_stats_report(stats: PipelineStats, path: Path) -> None:
    payload = {
        "chunks_scanned": stats.chunks_scanned,
        "records_seen": stats.records_seen,
        "records_skipped_decode": stats.records_skipped_decode,
        "records_skipped_delete": stats.records_skipped_delete,
        "illegal_board": stats.illegal_board,
        "filtered_ply": stats.filtered_ply,
        "filtered_visits": stats.filtered_visits,
        "positions_valid": stats.positions_valid,
        "games_seen": stats.games_seen,
        "bucket_counts_raw": dict(stats.bucket_counts_raw),
        "bucket_counts_after_filter": dict(stats.bucket_counts_after_filter),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")