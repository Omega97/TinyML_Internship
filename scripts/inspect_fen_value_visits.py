#!/usr/bin/env python3
"""Print shape, on-disk size, and column dtypes for the slim train table.

Defaults to the **joined** table
``data/processed/board_eval/fen_value_visits.parquet``.
Per-source slices live in ``data/processed/board_eval/fen_value_visits/``.
Same report as ``inspect_dataset.py``.

Example::

    py -3.12 scripts/inspect_fen_value_visits.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import (
    BOARD_EVAL_DIR_NAME,
    FEN_VALUE_VISITS_DIR_NAME,
    FEN_VALUE_VISITS_JOINED_NAME,
)

from inspect_dataset import _resolve, inspect_path

DEFAULT_PATH = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_JOINED_NAME
SOURCES_DIR = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print fen/value/visits dataset shape, file size, and column types"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"Parquet/CSV/JSON file(s) (default: {DEFAULT_PATH.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Also inspect per-source files under board_eval/fen_value_visits/",
    )
    args = parser.parse_args(argv)

    paths = [_resolve(p) for p in args.paths] if args.paths else [_resolve(DEFAULT_PATH)]
    if args.sources:
        for child in sorted(SOURCES_DIR.glob("fen_value_visits_*.parquet")):
            resolved = child.resolve()
            if resolved not in paths:
                paths.append(resolved)
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"missing: {p}", file=sys.stderr)
        return 1

    for i, path in enumerate(paths):
        if i:
            print()
        try:
            inspect_path(path)
        except Exception as exc:  # noqa: BLE001
            print(f"failed {path}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
