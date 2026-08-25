#!/usr/bin/env python3
"""Count unique EPDs vs total rows in fen-value-visits slices.

Walks ``data/processed/board_eval/fen_value_visits/<slice>/`` (same discovery as
``join_fen_value_visits.py``: parquet wins over JSON for the same stem). Does
**not** write the joined table.

- **total rows** — sum of slice rows (each file is already unique-EPD locally,
  except clock-only FEN variants, which collapse).
- **unique EPDs** — distinct board + STM + castling + EP across all slices
  (halfmove/fullmove ignored).
- **visits sum** — observation count (``visits`` column summed).

Example::

    py -3.12 -u scripts/count_fen_value_visits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME

import pandas as pd

from join_fen_value_visits import DEFAULT_SOURCES_DIR, discover_slices, epd_key

DEFAULT_DIR = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_count_table(path: Path) -> pd.DataFrame:
    """Fen + visits only (no WDL conversion). Parquet or JSON."""
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_parquet(path)
    if "fen" not in df.columns:
        raise ValueError(f"{path} missing column fen")
    out = pd.DataFrame({"fen": df["fen"].astype(str)})
    if "visits" in df.columns:
        out["visits"] = pd.to_numeric(df["visits"], errors="coerce").fillna(1).astype("int64")
    else:
        out["visits"] = 1
    return out.loc[out["visits"] > 0].copy()


def count_slices(paths: list[Path]) -> dict:
    """Load each slice once; accumulate row/visit totals and a global EPD set."""
    seen: set[str] = set()
    slice_stats: list[dict] = []
    total_rows = 0
    visits_sum = 0
    for path in paths:
        df = load_count_table(path)
        keys = df["fen"].map(epd_key)
        n_rows = int(len(df))
        n_unique = int(keys.nunique())
        vsum = int(df["visits"].sum())
        seen.update(keys.tolist())
        total_rows += n_rows
        visits_sum += vsum
        slice_stats.append(
            {
                "path": _rel(path),
                "rows": n_rows,
                "unique_epds": n_unique,
                "visits_sum": vsum,
            }
        )
    unique_epds = int(len(seen))
    return {
        "key": "EPD (fen fields 1–4); halfmove/fullmove ignored",
        "n_slices": len(paths),
        "total_rows": total_rows,
        "unique_epds": unique_epds,
        "visits_sum": visits_sum,
        "cross_slice_overlap_rows": total_rows - unique_epds,
        "slices": slice_stats,
    }


def print_report(stats: dict, *, quiet: bool = False) -> None:
    if not quiet:
        for item in stats["slices"]:
            extra = ""
            if item["unique_epds"] != item["rows"]:
                extra = f", unique_in_slice={item['unique_epds']:,}"
            print(
                f"slice {item['path']}: {item['rows']:,} rows{extra}, "
                f"visits_sum={item['visits_sum']:,}"
            )
        if stats["slices"]:
            print("---")
    print(
        f"slices: {stats['n_slices']}\n"
        f"total rows (sum of slice rows): {stats['total_rows']:,}\n"
        f"unique EPDs (fields 1–4): {stats['unique_epds']:,}\n"
        f"visits sum: {stats['visits_sum']:,}\n"
        f"cross-slice overlap (total rows − unique): {stats['cross_slice_overlap_rows']:,}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count unique EPDs and total rows in fen-value-visits slices"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_SOURCES_DIR,
        help=f"Parent of per-slice folders (default: {DEFAULT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print totals only (no per-slice lines)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the same stats as JSON",
    )
    args = parser.parse_args(argv)

    sources_dir = _resolve(args.input_dir)
    slices = discover_slices(sources_dir)
    if not slices:
        print(f"no slices in {sources_dir}", file=sys.stderr)
        return 1

    stats = count_slices(slices)
    print_report(stats, quiet=args.quiet)
    if args.json_out is not None:
        out = _resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"json → {_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
