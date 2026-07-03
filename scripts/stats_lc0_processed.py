#!/usr/bin/env python3
"""Bucket / ply survival stats on Lc0 chunks (pre–Stockfish labeling check)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import LC0_CHUNKS_DIR, LC0_PROCESSED_DIR, PROJECT_ROOT
from tinymlinternship.data.lc0_preprocess import (
    DEFAULT_BUCKET_MIN_PLY,
    DEFAULT_MIN_PLY,
    FilterConfig,
    PipelineStats,
    collect_bucket_survival,
    discover_chunks,
    iter_chunk_positions,
    parse_position,
    passes_filter,
    save_stats_report,
)
from tinymlinternship.features.bucket import NUM_BUCKETS


def run_filter_pass(
    chunk_paths: list[Path],
    cfg: FilterConfig,
    *,
    max_records: int,
) -> PipelineStats:
    stats = PipelineStats()
    seen = 0
    for chunk_path in chunk_paths:
        stats.chunks_scanned += 1
        for game_id, ply, pos in iter_chunk_positions(chunk_path, stats=stats):
            parsed = parse_position(game_id, ply, pos, chunk_path, stats=stats)
            if parsed is None:
                continue
            passes_filter(parsed, cfg, stats=stats)
            seen += 1
            if seen >= max_records:
                return stats
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lc0 preprocessing survival statistics")
    parser.add_argument("--chunks-dir", type=Path, default=LC0_CHUNKS_DIR)
    parser.add_argument("--max-chunks", type=int, default=100, help="Chunks to scan")
    parser.add_argument("--max-records", type=int, default=50_000, help="Cap parsed positions")
    parser.add_argument(
        "--min-ply-sweep",
        type=str,
        default="0,8,16,32,48",
        help="Comma-separated global min_ply values for survival table",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=DEFAULT_MIN_PLY,
        help=f"Default min_ply for single filter pass (default {DEFAULT_MIN_PLY} = 16 full moves)",
    )
    parser.add_argument(
        "--bucket7-min-ply",
        type=int,
        default=DEFAULT_BUCKET_MIN_PLY[7],
        help="Relaxed min_ply for bucket 7 only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LC0_PROCESSED_DIR / "stats",
    )
    args = parser.parse_args(argv)

    chunks = discover_chunks(args.chunks_dir, limit=args.max_chunks)
    if not chunks:
        print(f"No chunks under {args.chunks_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    min_ply_values = [int(x.strip()) for x in args.min_ply_sweep.split(",") if x.strip()]

    print(f"Project:  {PROJECT_ROOT}")
    print(f"Chunks:   {len(chunks)} (cap {args.max_chunks})")
    print(f"Records:  up to {args.max_records:,}")
    print()

    sweep_df = collect_bucket_survival(
        chunks,
        min_ply_values,
        bucket_min_ply={7: args.bucket7_min_ply},
        max_records=args.max_records,
    )
    sweep_path = args.output_dir / "bucket_survival_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print("Bucket survival sweep (global min_ply × bucket):")
    pivot = sweep_df.pivot(index="min_ply_global", columns="bucket", values="count")
    print(pivot.to_string())
    print(f"\nSaved: {sweep_path}")

    cfg = FilterConfig(
        min_ply=args.min_ply,
        bucket_min_ply={7: args.bucket7_min_ply},
    )
    stats = run_filter_pass(chunks, cfg, max_records=args.max_records)
    report_path = args.output_dir / "filter_pass_report.json"
    save_stats_report(stats, report_path)

    print(f"\nFilter pass: min_ply={cfg.min_ply}, bucket7_min_ply={args.bucket7_min_ply}")
    print(f"  records seen:     {stats.records_seen:,}")
    print(f"  valid positions:  {stats.positions_valid:,}")
    print(f"  after ply filter: {sum(stats.bucket_counts_after_filter.values()):,}")
    print(f"  filtered (ply):   {stats.filtered_ply:,}")
    print(f"  illegal FEN:      {stats.illegal_board:,}")
    print("  bucket raw:      ", dict(sorted(stats.bucket_counts_raw.items())))
    print("  bucket filtered: ", dict(sorted(stats.bucket_counts_after_filter.items())))

    b7 = stats.bucket_counts_after_filter.get(7, 0)
    target_per_bucket = max(1, sum(stats.bucket_counts_after_filter.values()) // NUM_BUCKETS)
    if b7 < target_per_bucket:
        print(
            f"\n⚠ Bucket 7 only {b7:,} positions after filter "
            f"(target ~{target_per_bucket:,} for uniform N/8). "
            f"Consider lowering --bucket7-min-ply (now {args.bucket7_min_ply})."
        )
    else:
        print(f"\n✓ Bucket 7 has {b7:,} positions — likely enough for stratified sampling.")

    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())