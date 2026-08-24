#!/usr/bin/env python3
"""Encode {fen, value, visits} tables to 844-feature caches (once).

Writes ``<table>.nnue/`` next to each source (npy arrays + meta.json).
Later ``scripts/train_nnue.py`` loads those arrays into RAM and skips encode.

Defaults: joined train parquet + ``test_set_2026-07_100000-100100.json``.

    py -3.12 -u scripts/encode_fen_value_visits.py
    py -3.12 -u scripts/encode_fen_value_visits.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_JOINED_NAME
from tinymlinternship.nnue.dataset import (
    cache_is_valid,
    default_cache_dir,
    ensure_feature_cache,
    prefer_parquet,
)

DEFAULT_TRAIN = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_JOINED_NAME
DEFAULT_TEST = (
    PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "test_set_2026-07_100000-100100.json"
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode fen-value-visits → 844 cache")
    parser.add_argument(
        "tables",
        nargs="*",
        type=Path,
        help="JSON/parquet tables (default: joined train + test holdout)",
    )
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing caches")
    args = parser.parse_args(argv)

    tables = [_resolve(p) for p in args.tables] if args.tables else [
        DEFAULT_TRAIN,
        DEFAULT_TEST,
    ]
    for raw in tables:
        source = prefer_parquet(raw)
        if not source.is_file():
            print(f"missing: {source}", file=sys.stderr)
            return 1
        cache_dir = default_cache_dir(source)
        if not args.rebuild and cache_is_valid(cache_dir, source, max_active=128):
            print(f"cache ok  {cache_dir}")
            continue
        print(f"encoding {source} → {cache_dir}")
        ensure_feature_cache(source, cache_dir=cache_dir, rebuild=args.rebuild, progress=True)
        print(f"wrote    {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
