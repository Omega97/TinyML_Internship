#!/usr/bin/env python3
"""Join per-source ``{fen, value, visits}`` slices into one table.

Reads every ``fen_value_visits_*`` JSON/parquet under
``data/processed/board_eval/fen_value_visits/`` (parquet preferred when both
exist). Same EPD (board + STM + castling + EP; clocks ignored) is merged:
visits are **summed**, value is the **visit-weighted** mean of slice values.

Writes, sorted by visits descending (then fen ascending):

    data/processed/board_eval/fen_value_visits.parquet
    data/processed/board_eval/fen_value_visits.json

Example::

    py -3.12 scripts/join_fen_value_visits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import (
    BOARD_EVAL_DIR_NAME,
    FEN_VALUE_VISITS_DIR_NAME,
    FEN_VALUE_VISITS_JOINED_NAME,
)

DEFAULT_SOURCES_DIR = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_OUTPUT = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_JOINED_NAME
SLICE_GLOB = "fen_value_visits_*"
SLICE_SUFFIXES = {".json", ".parquet"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def epd_key(fen: str) -> str:
    """Board + side + castling + EP (halfmove/fullmove dropped)."""
    parts = str(fen).split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return str(fen).strip()


def discover_slices(sources_dir: Path) -> list[Path]:
    """One path per stem; parquet wins over JSON so twins are not double-counted."""
    by_stem: dict[str, Path] = {}
    if not sources_dir.is_dir():
        return []
    for path in sources_dir.glob(SLICE_GLOB):
        if path.suffix.lower() not in SLICE_SUFFIXES or not path.is_file():
            continue
        prev = by_stem.get(path.stem)
        if prev is None or (path.suffix.lower() == ".parquet" and prev.suffix.lower() != ".parquet"):
            by_stem[path.stem] = path
    return [by_stem[key] for key in sorted(by_stem)]


def load_slice(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_parquet(path)
    missing = {"fen", "value", "visits"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    out = pd.DataFrame(
        {
            "fen": df["fen"].astype(str),
            "value": pd.to_numeric(df["value"], errors="coerce"),
            "visits": pd.to_numeric(df["visits"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["fen", "value", "visits"])
    out["visits"] = out["visits"].astype("int64")
    return out.loc[out["visits"] > 0].copy()


def merge_slices(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["fen", "value", "visits"])
    df = pd.concat(frames, ignore_index=True)
    df["epd"] = df["fen"].map(epd_key)
    df["wv"] = df["value"] * df["visits"].astype("float64")
    # Keep the FEN from the slice row with the most visits (stable on ties).
    order = df.sort_values(["visits", "fen"], ascending=[False, True])
    fen_keep = order.drop_duplicates("epd", keep="first").set_index("epd")["fen"]
    grouped = df.groupby("epd", sort=False, as_index=True)
    visits = grouped["visits"].sum()
    value = grouped["wv"].sum() / visits
    out = pd.DataFrame(
        {
            "fen": fen_keep.reindex(visits.index),
            "value": value.astype("float64"),
            "visits": visits.astype("int64"),
        }
    )
    return out.sort_values(["visits", "fen"], ascending=[False, True]).reset_index(drop=True)


def write_join(df: pd.DataFrame, parquet_path: Path, *, also_json: bool = True) -> Path | None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    json_path: Path | None = None
    if also_json:
        json_path = parquet_path.with_suffix(".json")
        payload = [
            {"fen": str(row["fen"]), "value": float(row["value"]), "visits": int(row["visits"])}
            for row in df.to_dict(orient="records")
        ]
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join fen-value-visits slices (sum visits, sort descending)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_SOURCES_DIR,
        help="Directory of per-source fen_value_visits_* JSON/parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Joined parquet (JSON twin written next to it)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Write parquet only",
    )
    args = parser.parse_args(argv)

    sources_dir = _resolve(args.input_dir)
    output = _resolve(args.output)
    slices = discover_slices(sources_dir)
    if not slices:
        print(f"no slices in {sources_dir}", file=sys.stderr)
        return 1

    frames: list[pd.DataFrame] = []
    file_stats: list[dict] = []
    for path in slices:
        df = load_slice(path)
        try:
            rel = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel = str(path)
        print(f"slice {rel}: {len(df):,} rows, visits_sum={int(df['visits'].sum()):,}")
        file_stats.append(
            {
                "path": rel,
                "rows": int(len(df)),
                "visits_sum": int(df["visits"].sum()),
            }
        )
        frames.append(df)

    joined = merge_slices(frames)
    if joined.empty:
        print("join is empty", file=sys.stderr)
        return 1

    json_path = write_join(joined, output, also_json=not args.no_json)
    visits_sum = int(joined["visits"].sum())
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "columns": ["fen", "value", "visits"],
        "key": "EPD (fen fields 1–4); halfmove/fullmove ignored",
        "visits": "sum of visits over slices for the same EPD",
        "value": "visit-weighted mean of slice teacher values",
        "order": "visits descending, then fen ascending",
        "n_positions": int(len(joined)),
        "visits_sum": visits_sum,
        "visits_min": int(joined["visits"].min()),
        "visits_max": int(joined["visits"].max()),
        "value_min": float(joined["value"].min()),
        "value_max": float(joined["value"].max()),
        "value_mean": float(joined["value"].mean()),
        "slices": file_stats,
        "output": str(output.relative_to(PROJECT_ROOT))
        if output.is_relative_to(PROJECT_ROOT)
        else str(output),
    }
    if json_path is not None:
        try:
            meta["output_json"] = str(json_path.relative_to(PROJECT_ROOT))
        except ValueError:
            meta["output_json"] = str(json_path)
    meta_path = output.with_suffix(".manifest.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"joined {len(joined):,} unique EPDs → {meta['output']}\n"
        f"  visits sum={visits_sum:,} min={meta['visits_min']} max={meta['visits_max']}\n"
        f"  value mean={meta['value_mean']:.4f}"
    )
    if json_path is not None:
        print(f"JSON → {meta.get('output_json', json_path)}")
    print(f"manifest → {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
