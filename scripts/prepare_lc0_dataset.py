#!/usr/bin/env python3
"""Parse Lc0 chunks → filter → sample → positions.parquet.

Pilot subset used bucket-stratified sampling for exploration; production
training keeps natural bucket distribution (blueprint §Training pipeline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import LC0_CHUNKS_DIR, LC0_PROCESSED_DIR, PROJECT_ROOT
from tinymlinternship.data.lc0_preprocess import (
    DEFAULT_BUCKET_MIN_PLY,
    DEFAULT_MIN_PLY,
    FilterConfig,
    PipelineStats,
    SampleConfig,
    discover_chunks,
    iter_chunk_positions,
    parse_position,
    passes_filter,
    positions_to_dataframe,
    save_stats_report,
    stratified_sample,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Lc0 position dataset (no SF labels)")
    parser.add_argument("--chunks-dir", type=Path, default=LC0_CHUNKS_DIR)
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit chunks scanned")
    parser.add_argument("--max-scan", type=int, default=200_000, help="Max valid candidates to scan")
    parser.add_argument("--total", type=int, default=10_000, help="Stratified sample size")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-ply",
        type=int,
        default=DEFAULT_MIN_PLY,
        help=f"Global opening skip in half-moves (default {DEFAULT_MIN_PLY} = 16 full moves)",
    )
    parser.add_argument(
        "--bucket7-min-ply",
        type=int,
        default=DEFAULT_BUCKET_MIN_PLY[7],
        help="Relaxed opening skip for bucket 7 (p=32)",
    )
    parser.add_argument("--min-visits", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LC0_PROCESSED_DIR,
    )
    args = parser.parse_args(argv)

    chunks = discover_chunks(args.chunks_dir, limit=args.max_chunks)
    if not chunks:
        print(f"No chunks under {args.chunks_dir}", file=sys.stderr)
        return 1

    filter_cfg = FilterConfig(
        min_ply=args.min_ply,
        bucket_min_ply={7: args.bucket7_min_ply},
        min_visits=args.min_visits,
    )
    sample_cfg = SampleConfig(
        total_positions=args.total,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    stats = PipelineStats()
    candidates = []
    for chunk_path in chunks:
        stats.chunks_scanned += 1
        for game_id, ply, pos in iter_chunk_positions(chunk_path, stats=stats):
            parsed = parse_position(game_id, ply, pos, chunk_path, stats=stats)
            if parsed is None:
                continue
            if passes_filter(parsed, filter_cfg, stats=stats):
                candidates.append(parsed)
            if len(candidates) >= args.max_scan:
                break
        if len(candidates) >= args.max_scan:
            break

    if not candidates:
        print("No positions passed filters.", file=sys.stderr)
        return 1

    # positions.parquet = natural mix (all filter-pass); stratified only for pilot splits
    train, val = stratified_sample(candidates, sample_cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "val.parquet"
    all_path = args.output_dir / "positions.parquet"

    positions_to_dataframe(candidates).to_parquet(all_path, index=False)
    positions_to_dataframe(train).to_parquet(train_path, index=False)
    positions_to_dataframe(val).to_parquet(val_path, index=False)

    from collections import Counter

    from tinymlinternship.features.bucket import NUM_BUCKETS

    bucket_counts = Counter(p.bucket for p in candidates)
    manifest = {
        "min_ply": args.min_ply,
        "bucket7_min_ply": args.bucket7_min_ply,
        "min_visits": args.min_visits,
        "chunks_scanned": stats.chunks_scanned,
        "candidates_after_filter": len(candidates),
        "positions_parquet_rows": len(candidates),
        "positions_parquet_policy": "natural_all_filter_pass",
        "stratified_train_rows": len(train),
        "stratified_val_rows": len(val),
        "seed": args.seed,
        "bucket_counts": {str(b): int(bucket_counts.get(b, 0)) for b in range(NUM_BUCKETS)},
        "buckets_missing": sorted(b for b in range(NUM_BUCKETS) if bucket_counts.get(b, 0) == 0),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    save_stats_report(stats, args.output_dir / "prepare_stats.json")

    print(f"Project:    {PROJECT_ROOT}")
    print(f"Scanned:    {stats.chunks_scanned} chunks, {stats.records_seen:,} records")
    print(f"Candidates: {len(candidates):,} (after filter) → positions.parquet (natural)")
    print(f"Stratified: train={len(train):,}  val={len(val):,} (pilot splits only)")
    print(f"Buckets:    {manifest['bucket_counts']}  missing={manifest['buckets_missing']}")
    print(f"Output:     {all_path}")
    print(f"            {train_path}")
    print(f"            {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())