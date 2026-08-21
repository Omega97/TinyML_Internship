#!/usr/bin/env python3
"""Export the current labeled set as ``fen, value, visits``.

* **fen** — canonical FEN (from the labeled train/val merge).
* **value** — teacher ``expected_reward`` (White POV, ``[-1, +1]``). Duplicate
  labels on the same EPD are averaged.
* **visits** — how many times that position appears in *extract* parquets
  (not labeled re-exports, which would double-count). A labeled position
  missing from extracts gets ``visits =`` its labeled multiplicity.

Dedup key is SHA-256 of EPD (board + STM + castling + EP).

Writes:

* joined table → ``data/processed/board_eval/fen_value_visits.parquet``
* per-source slices → ``data/processed/board_eval/fen_value_visits/fen_value_visits_<source>.parquet``
  (and a JSON copy ``fen_value_visits_<source>.json`` next to each parquet)
* extra labeled slices (not merged into the join) → ``fen_value_visits_<slug>.parquet`` / ``.json``

Example::

    py -3.12 scripts/export_fen_value_visits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR
from tinymlinternship.data.board_store import (
    BOARD_EVAL_DIR_NAME,
    FEN_VALUE_VISITS_DIR_NAME,
    FEN_VALUE_VISITS_JOINED_NAME,
    add_teacher_value,
    bump_visits,
    fen_value_visits_source_filename,
    slim_fen_value_visits,
)
from tinymlinternship.data.schema import LABEL_FORMULA

# Current production labeled set (row universe + teacher values)
VALUE_SOURCES = [
    PROCESSED_DATA_DIR / "labeled" / "train.parquet",
    PROCESSED_DATA_DIR / "labeled" / "val.parquet",
    PROCESSED_DATA_DIR / "labeled" / "lichess_puzzles.parquet",
]

# Observation counts — raw extracts only (same policy as build_dataset_json.py)
VISIT_SOURCES = [
    RAW_DATA_DIR / "lc0" / "positions.parquet",
    RAW_DATA_DIR / "lichess" / "positions.parquet",
    RAW_DATA_DIR / "kaggle" / "kaggle_games_positions.parquet",
    RAW_DATA_DIR / "lichess" / "puzzles_sample.parquet",
    RAW_DATA_DIR / "lichess" / "puzzles_sample_16k.parquet",
]

# Extra labeled parquets written as their own files (values not mixed into the join).
EXTRA_VALUE_SLICES: list[tuple[Path, str]] = [
    (PROCESSED_DATA_DIR / "labeled" / "lc0_large_25k.parquet", "lc0_large_25k"),
    (PROCESSED_DATA_DIR / "labeled" / "lc0_large_40k.parquet", "lc0_large_40k"),
    (PROCESSED_DATA_DIR / "labeled" / "lichess_kaggle_10k.parquet", "lichess_kaggle_10k"),
    (PROCESSED_DATA_DIR / "labeled" / "lichess_kaggle_40k.parquet", "lichess_kaggle_40k"),
    (PROCESSED_DATA_DIR / "labeled" / "lichess_puzzles_16k.parquet", "lichess_puzzles_16k"),
]

DEFAULT_OUTPUT = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_JOINED_NAME
DEFAULT_SOURCES_DIR = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _iter_fens(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if "fen" not in df.columns:
        raise ValueError(f"{path} has no fen column")
    return df


def _records_payload(df: pd.DataFrame) -> list[dict]:
    return [
        {"fen": str(row["fen"]), "value": float(row["value"]), "visits": int(row["visits"])}
        for row in df.to_dict(orient="records")
    ]


def _write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _visit_shells(store: dict[str, dict]) -> dict[str, dict]:
    return {
        key: {
            "fen": rec["fen"],
            "visits": int(rec["visits"]),
            "_value_sum": 0.0,
            "_value_n": 0,
        }
        for key, rec in store.items()
    }


def _add_labels(
    store: dict[str, dict],
    path: Path,
    stats: dict[str, object],
    by_source: dict[str, set[str]] | None = None,
) -> int:
    df = _iter_fens(path)
    if "expected_reward" not in df.columns:
        raise ValueError(f"need expected_reward in {path}")
    n = 0
    sources = (
        df["source"].tolist() if "source" in df.columns else ["unknown"] * len(df)
    )
    for fen, reward, source in zip(
        df["fen"].tolist(),
        df["expected_reward"].tolist(),
        sources,
        strict=True,
    ):
        if fen is None or reward is None or (isinstance(reward, float) and pd.isna(reward)):
            continue
        try:
            key = add_teacher_value(store, str(fen), float(reward))
            if by_source is not None:
                src = (
                    "unknown"
                    if source is None or (isinstance(source, float) and pd.isna(source))
                    else str(source)
                )
                by_source.setdefault(src, set()).add(key)
            n += 1
        except Exception as exc:  # noqa: BLE001
            print(f"warn value {path.name}: {exc}", file=sys.stderr)
    stats["label_rows"] = int(stats.get("label_rows", 0)) + n
    files = stats.setdefault("files", {})
    assert isinstance(files, dict)
    files.setdefault(str(path), {})["label_rows"] = n
    print(f"values {path.name}: {n:,}")
    return n


def _write_table(rows: list[dict], path: Path, *, also_json: bool = False) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=["fen", "value", "visits"])
    out["visits"] = out["visits"].astype("int64")
    out = out.sort_values(["visits", "fen"], ascending=[False, True]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    if also_json:
        _write_json(_records_payload(out), path.with_suffix(".json"))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export fen, value (teacher), visits (extract multiplicity)"
    )
    parser.add_argument(
        "--value-input",
        type=Path,
        action="append",
        default=None,
        help="Labeled parquet/CSV with fen + expected_reward (repeatable)",
    )
    parser.add_argument(
        "--visit-input",
        type=Path,
        action="append",
        default=None,
        help="Extract parquet/CSV with fen for visit counts (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Joined parquet at board_eval root",
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=DEFAULT_SOURCES_DIR,
        help="Per-source parquets: fen_value_visits_<source>.parquet",
    )
    parser.add_argument(
        "--only-extra",
        action="store_true",
        help="Write EXTRA_VALUE_SLICES only; do not rewrite the joined table",
    )
    args = parser.parse_args(argv)

    value_paths = [_resolve(p) for p in (args.value_input or VALUE_SOURCES)]
    visit_paths = [_resolve(p) for p in (args.visit_input or VISIT_SOURCES)]
    output = _resolve(args.output)
    sources_dir = _resolve(args.sources_dir)

    store: dict[str, dict] = {}
    by_source: dict[str, set[str]] = {}
    stats: dict[str, object] = {"visit_rows": 0, "label_rows": 0, "files": {}}

    for path in visit_paths:
        if not path.exists():
            print(f"skip missing visits: {path}", file=sys.stderr)
            continue
        df = _iter_fens(path)
        n = 0
        for fen in df["fen"].tolist():
            if fen is None or (isinstance(fen, float) and pd.isna(fen)):
                continue
            try:
                bump_visits(store, str(fen))
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"warn visit {path.name}: {exc}", file=sys.stderr)
        stats["visit_rows"] = int(stats["visit_rows"]) + n
        stats["files"][str(path)] = {"visit_rows": n}
        print(f"visits {path.name}: {n:,}")

    visit_shells = _visit_shells(store)

    extra_files: dict[str, str] = {}
    source_files: dict[str, str] = {}
    out: pd.DataFrame | None = None

    if not args.only_extra:
        for path in value_paths:
            if not path.exists():
                print(f"skip missing labels: {path}", file=sys.stderr)
                continue
            try:
                _add_labels(store, path, stats, by_source)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1

        rows = slim_fen_value_visits(store, labeled_only=True)
        if not rows:
            print("no labeled rows", file=sys.stderr)
            return 1

        out = _write_table(rows, output)
        for source, keys in sorted(by_source.items()):
            subset = {k: store[k] for k in keys if k in store}
            src_rows = slim_fen_value_visits(subset, labeled_only=True)
            if not src_rows:
                continue
            src_path = sources_dir / fen_value_visits_source_filename(source)
            _write_table(src_rows, src_path, also_json=True)
            rel = str(src_path.relative_to(PROJECT_ROOT))
            source_files[source] = rel
            print(f"source {source}: {len(src_rows):,} → {rel} + {src_path.with_suffix('.json').name}")

    for extra_path, slug in EXTRA_VALUE_SLICES:
        extra_path = _resolve(extra_path)
        if not extra_path.exists():
            print(f"skip missing extra: {extra_path}", file=sys.stderr)
            continue
        extra_store = {k: dict(v) for k, v in visit_shells.items()}
        try:
            _add_labels(extra_store, extra_path, stats)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        extra_rows = slim_fen_value_visits(extra_store, labeled_only=True)
        if not extra_rows:
            print(f"no labeled rows in extra {slug}", file=sys.stderr)
            continue
        extra_out = sources_dir / fen_value_visits_source_filename(slug)
        extra_df = _write_table(extra_rows, extra_out, also_json=True)
        rel = str(extra_out.relative_to(PROJECT_ROOT))
        extra_files[slug] = rel
        print(
            f"extra {slug}: {len(extra_df):,} → {rel} + {extra_out.with_suffix('.json').name}"
        )

    if out is not None:
        meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "columns": ["fen", "value", "visits"],
            "value": "teacher expected_reward (White POV, [-1, +1])",
            "label_formula": LABEL_FORMULA,
            "visits": "observation count over extract parquets (EPD hash)",
            "n_positions": int(len(out)),
            "visits_sum": int(out["visits"].sum()),
            "visits_min": int(out["visits"].min()),
            "visits_max": int(out["visits"].max()),
            "value_min": float(out["value"].min()),
            "value_max": float(out["value"].max()),
            "value_mean": float(out["value"].mean()),
            "source_files": source_files,
            "extra_files": extra_files,
            "stats": stats,
            "output": str(output.relative_to(PROJECT_ROOT)),
        }
        meta_path = output.with_suffix(".manifest.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(
            f"Joined {len(out):,} unique positions → {output.relative_to(PROJECT_ROOT)}\n"
            f"  visits sum={meta['visits_sum']:,} min={meta['visits_min']} "
            f"max={meta['visits_max']}  value mean={meta['value_mean']:.4f}"
        )
        print(f"Manifest → {meta_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
