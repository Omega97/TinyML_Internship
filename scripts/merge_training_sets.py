#!/usr/bin/env python3
"""Merge labeled Lichess + Lc0 parquets → train/val + manifest (ASSETS schema).

Split is by ``game_id`` (fixed seed) to avoid position leakage. No stratified
bucket rebalance — natural mix preserved.

Example::

    py -3.12 scripts/merge_training_sets.py \\
        --inputs data/processed/labeled/lichess_labeled.parquet \\
                 data/processed/labeled/lc0_labeled.parquet \\
        --output-dir data/processed/labeled \\
        --val-fraction 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tinymlinternship.config.settings import (
    LC0_BINARY,
    LC0_NETWORK_DEFAULT,
    PROJECT_ROOT,
)
from tinymlinternship.data.schema import (
    LABELED_REQUIRED_COLUMNS,
    build_manifest,
    ensure_prelabel_columns,
    sha256_file,
    split_by_game,
    validate_labeled_frame,
    write_manifest,
)


def load_labeled(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    df = ensure_prelabel_columns(df)
    if "expected_reward" not in df.columns:
        raise ValueError(f"{path} has no expected_reward — run label_positions.py first")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge labeled parquets → train/val + manifest.json (ASSETS)"
    )
    parser.add_argument(
        "--inputs",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        help="One or more labeled parquet/CSV files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "labeled",
        help="Directory for train.parquet, val.parquet, manifest.json",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--teacher-network",
        type=str,
        default=None,
        help="Override teacher net name recorded in manifest (default: from data or 791556)",
    )
    parser.add_argument(
        "--allow-mixed-teachers",
        action="store_true",
        help="Allow multiple teacher_network values (not recommended)",
    )
    args = parser.parse_args(argv)

    frames: list[pd.DataFrame] = []
    input_rels: list[str] = []
    for p in args.inputs:
        if not p.exists():
            print(f"Input not found: {p}", file=sys.stderr)
            return 1
        frames.append(load_labeled(p.resolve()))
        try:
            input_rels.append(str(p.resolve().relative_to(PROJECT_ROOT)))
        except ValueError:
            input_rels.append(str(p.resolve()))

    df = pd.concat(frames, ignore_index=True)

    # Teacher consistency
    if "teacher_network" in df.columns:
        nets = sorted(df["teacher_network"].astype(str).unique())
        if len(nets) > 1 and not args.allow_mixed_teachers:
            print(
                f"Multiple teacher_network values: {nets}. "
                "Re-label with one net or pass --allow-mixed-teachers.",
                file=sys.stderr,
            )
            return 1
        teacher_network = args.teacher_network or nets[0]
    else:
        teacher_network = args.teacher_network or LC0_NETWORK_DEFAULT.name
        df["teacher_network"] = teacher_network

    try:
        validate_labeled_frame(df, require_source=True)
    except ValueError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    train, val = split_by_game(df, val_fraction=args.val_fraction, seed=args.seed)

    # Prefer ASSETS column order, keep extras at end
    preferred = list(LABELED_REQUIRED_COLUMNS) + [
        "wdl_win",
        "wdl_draw",
        "wdl_loss",
        "teacher_network",
    ]
    ordered = [c for c in preferred if c in train.columns]
    ordered += [c for c in train.columns if c not in ordered]
    train = train[ordered]
    val = val[ordered] if len(val) else val

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    manifest_path = out_dir / "manifest.json"

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    # Resolve teacher net file for sha256 if it matches a known install
    net_path = None
    for candidate in (
        PROJECT_ROOT / "models" / "teacher" / "lc0" / teacher_network,
        PROJECT_ROOT / "models" / "teacher" / "networks" / teacher_network,
        LC0_NETWORK_DEFAULT,
    ):
        if candidate.is_file() and candidate.name == teacher_network:
            net_path = candidate
            break
        if candidate.is_file() and teacher_network in candidate.name:
            net_path = candidate
            break
    # Also try exact default path by name
    if net_path is None:
        guess = PROJECT_ROOT / "models" / "teacher" / "lc0" / teacher_network
        if guess.is_file():
            net_path = guess

    net_sha = sha256_file(net_path) if net_path and net_path.is_file() else None
    binary_rel = None
    if LC0_BINARY.is_file():
        try:
            binary_rel = str(LC0_BINARY.relative_to(PROJECT_ROOT))
        except ValueError:
            binary_rel = str(LC0_BINARY)

    sources = sorted(df["source"].astype(str).unique().tolist())
    manifest = build_manifest(
        train=train,
        val=val,
        sources=sources,
        teacher_network=teacher_network,
        teacher_network_sha256=net_sha,
        teacher_binary=binary_rel,
        filters={},
        sample_rate={},
        extra={
            "input_files": input_rels,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "split": "by_game_id",
            "train_path": str(train_path.relative_to(PROJECT_ROOT))
            if train_path.is_relative_to(PROJECT_ROOT)
            else str(train_path),
            "val_path": str(val_path.relative_to(PROJECT_ROOT))
            if val_path.is_relative_to(PROJECT_ROOT)
            else str(val_path),
        },
    )
    write_manifest(manifest_path, manifest)

    print(
        json.dumps(
            {
                "train_rows": len(train),
                "val_rows": len(val),
                "train_games": int(train["game_id"].nunique()),
                "val_games": int(val["game_id"].nunique()) if len(val) else 0,
                "sources": sources,
                "teacher_network": teacher_network,
                "output_dir": str(out_dir.relative_to(PROJECT_ROOT))
                if out_dir.is_relative_to(PROJECT_ROOT)
                else str(out_dir),
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT))
                if manifest_path.is_relative_to(PROJECT_ROOT)
                else str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
