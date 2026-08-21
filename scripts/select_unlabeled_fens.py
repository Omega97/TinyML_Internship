#!/usr/bin/env python3
"""Write unique unlabeled FENs from an extract parquet (Goal §1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tinymlinternship.config.settings import PROJECT_ROOT
from tinymlinternship.data.board_store import board_hash


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def labeled_epd_keys(labeled_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted(labeled_dir.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["fen"])
        for fen in df["fen"].astype(str).tolist():
            try:
                keys.add(board_hash(fen))
            except Exception:
                continue
    return keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select unlabeled unique FENs")
    parser.add_argument("--extract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "labeled",
    )
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args(argv)

    extract = _resolve(args.extract)
    output = _resolve(args.output)
    labeled_dir = _resolve(args.labeled_dir)
    print(f"loading labeled keys from {labeled_dir} …")
    seen = labeled_epd_keys(labeled_dir)
    print(f"  {len(seen):,} already labeled unique EPDs")

    df = pd.read_parquet(extract)
    if "fen" not in df.columns:
        print(f"no fen column in {extract}", file=sys.stderr)
        return 1
    source = args.source or (str(df["source"].iloc[0]) if "source" in df.columns else "unknown")

    rows: list[dict] = []
    local: set[str] = set()
    for rec in df.to_dict(orient="records"):
        fen = rec.get("fen")
        if fen is None or (isinstance(fen, float) and pd.isna(fen)):
            continue
        try:
            key = board_hash(str(fen))
        except Exception:
            continue
        if key in seen or key in local:
            continue
        local.add(key)
        row = dict(rec)
        row["fen"] = str(fen)
        row["source"] = source
        rows.append(row)
        if args.limit is not None and len(rows) >= args.limit:
            break

    if not rows:
        print("no unlabeled FENs")
        return 1
    out = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    print(f"wrote {len(out):,} unlabeled unique FENs → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
