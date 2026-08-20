#!/usr/bin/env python3
"""Print shape, on-disk size, and column dtypes for a tabular dataset.

Defaults to the merged labeled training set
``data/processed/labeled/train.parquet``.

Example::

    py -3.12 scripts/inspect_dataset.py
    py -3.12 scripts/inspect_dataset.py data/processed/labeled/val.parquet
    py -3.12 scripts/inspect_dataset.py data/raw/games.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT

DEFAULT_PATH = PROCESSED_DATA_DIR / "labeled" / "train.parquet"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{n} B"


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported format {suffix!r} (use parquet, csv, or json)")


def _sample_python_type(series: pd.Series) -> str:
    for value in series:
        if pd.isna(value):
            continue
        return type(value).__name__
    return "empty"


def inspect_path(path: Path) -> None:
    disk = path.stat().st_size
    df = load_table(path)
    mem = int(df.memory_usage(deep=True).sum())
    rows, cols = df.shape

    print(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)
    print(f"  disk:       {_human_bytes(disk)} ({disk:,} bytes)")
    print(f"  in-memory:  {_human_bytes(mem)} ({mem:,} bytes)")
    print(f"  shape:      {rows:,} rows × {cols} columns")
    print()
    name_w = max(8, max((len(str(c)) for c in df.columns), default=8))
    dtype_w = max(8, max((len(str(t)) for t in df.dtypes), default=8))
    header = f"  {'column':<{name_w}}  {'dtype':<{dtype_w}}  {'python':<10}  non-null"
    print(header)
    print(f"  {'-' * name_w}  {'-' * dtype_w}  {'-' * 10}  --------")
    non_null = df.notna().sum()
    for col, dtype in df.dtypes.items():
        py_type = _sample_python_type(df[col])
        print(
            f"  {str(col):<{name_w}}  {str(dtype):<{dtype_w}}  {py_type:<10}  "
            f"{int(non_null[col]):,}/{rows:,}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print dataset shape, file size, and column types"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"Parquet/CSV/JSON file(s) (default: {DEFAULT_PATH.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Also inspect data/processed/labeled/val.parquet",
    )
    args = parser.parse_args(argv)

    paths = [_resolve(p) for p in args.paths] if args.paths else [_resolve(DEFAULT_PATH)]
    if args.val:
        val = PROCESSED_DATA_DIR / "labeled" / "val.parquet"
        if val not in paths:
            paths.append(val)

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
