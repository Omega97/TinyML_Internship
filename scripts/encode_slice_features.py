#!/usr/bin/env python3
"""Encode each fen-value-visits slice folder → features.npz (train-ready).

For every child of ``data/processed/board_eval/fen_value_visits/``:

    <slice>/fen_value_visits_….json  →  <slice>/features.npz

Arrays: sparse dual-POV 844 indices + STM ``wdl`` ``(N, 3)``. **No visits.**
JSON that only has White-POV ``value`` is converted to STM WDL at encode time.
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


def encode_slice_folder(
    folder: Path,
    *,
    rebuild: bool = False,
    progress: bool = True,
) -> Path:
    """JSON/parquet in ``folder`` → ``folder/features.npz``. Rebuild if the source changed."""
    folder = _resolve(folder)
    source = slice_source_json(folder)
    if source is None:
        raise FileNotFoundError(f"no slice JSON in {folder}")
    npz = slice_features_path(folder)
    if not rebuild and slice_db_is_valid(folder, source, max_active=128):
        try:
            print(f"ok    {npz.relative_to(PROJECT_ROOT)}")
        except ValueError:
            print(f"ok    {npz}")
        return npz
    try:
        print(f"encode {source.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"encode {source}")
    ensure_slice_feature_db(folder, rebuild=rebuild, progress=progress)
    try:
        print(f"wrote  {npz.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"wrote  {npz}")
    return npz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Encode each slice JSON to features.npz (sparse 844 + STM WDL)"
    )
    parser.add_argument(
        "slice",
        nargs="?",
        type=Path,
        default=None,
        help="One slice folder (default: every child of --root)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Parent of per-slice folders",
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    if args.slice is not None:
        folder = _resolve(args.slice)
        if slice_source_json(folder) is None:
            print(f"no slice JSON in {folder}", file=sys.stderr)
            return 1
        folders = [folder]
    else:
        root = _resolve(args.root)
        folders = discover_slice_folders(root)
    if not folders:
        where = args.slice if args.slice is not None else args.root
        print(f"no slice folders in {where}", file=sys.stderr)
        return 1

    for folder in folders:
        encode_slice_folder(folder, rebuild=args.rebuild, progress=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
