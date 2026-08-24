#!/usr/bin/env python3
"""Encode each fen-value-visits slice folder → features.npz (train-ready).

For every child of ``data/processed/board_eval/fen_value_visits/``:

    <slice>/fen_value_visits_….json  →  <slice>/features.npz

Arrays: sparse dual-POV 844 indices + White-POV ``value``. **No visits.**
Training loads the npz; chess encoding is not repeated.

    py -3.12 -u scripts/encode_slice_features.py
    py -3.12 -u scripts/encode_slice_features.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import (
    discover_slice_folders,
    ensure_slice_feature_db,
    slice_db_is_valid,
    slice_features_path,
    slice_source_json,
)

DEFAULT_ROOT = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Encode each slice JSON to features.npz in the same folder"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Parent of per-slice folders",
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    root = _resolve(args.root)
    folders = discover_slice_folders(root)
    if not folders:
        print(f"no slice folders in {root}", file=sys.stderr)
        return 1

    for folder in folders:
        source = slice_source_json(folder)
        assert source is not None
        npz = slice_features_path(folder)
        if not args.rebuild and slice_db_is_valid(folder, source, max_active=128):
            print(f"ok    {npz.relative_to(PROJECT_ROOT)}")
            continue
        print(f"encode {source.relative_to(PROJECT_ROOT)}")
        ensure_slice_feature_db(folder, rebuild=args.rebuild, progress=True)
        print(f"wrote  {npz.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
