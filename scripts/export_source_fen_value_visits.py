#!/usr/bin/env python3
"""Write one {fen, value, visits} JSON for a teacher-labeled source.

Visits come from extract parquets (EPD hash). Values come from labeled parquets
with matching ``source``. Output: data/processed/board_eval/fen_value_visits/

Example::

    py -3.12 scripts/export_source_fen_value_visits.py --source lc0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR
from tinymlinternship.data.board_store import (
    BOARD_EVAL_DIR_NAME,
    FEN_VALUE_VISITS_DIR_NAME,
    add_teacher_value,
    bump_visits,
    fen_value_visits_source_filename,
    slim_fen_value_visits,
)

EXTRACTS: dict[str, list[Path]] = {
    "lc0": [
        RAW_DATA_DIR / "lc0" / "positions.parquet",
        RAW_DATA_DIR / "lc0" / "extract_250k" / "positions.parquet",
    ],
    "lichess": [
        RAW_DATA_DIR / "lichess" / "positions.parquet",
    ],
    "lichess_puzzles": [
        RAW_DATA_DIR / "lichess" / "puzzles_sample.parquet",
        RAW_DATA_DIR / "lichess" / "puzzles_sample_16k.parquet",
    ],
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export one source to fen-value-visits JSON")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "labeled",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME,
    )
    args = parser.parse_args(argv)
    source = args.source
    labeled_dir = _resolve(args.labeled_dir)
    out_dir = _resolve(args.output_dir)

    store: dict[str, dict] = {}
    for path in EXTRACTS.get(source, []):
        path = _resolve(path)
        if not path.exists():
            print(f"skip missing extract: {path}", file=sys.stderr)
            continue
        df = pd.read_parquet(path, columns=["fen"])
        n = 0
        for fen in df["fen"].astype(str).tolist():
            try:
                bump_visits(store, fen)
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"warn visit: {exc}", file=sys.stderr)
        print(f"visits {path.name}: {n:,}")

    n_lab = 0
    for path in sorted(labeled_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        if "expected_reward" not in df.columns or "fen" not in df.columns:
            continue
        if "source" in df.columns:
            df = df[df["source"].astype(str) == source]
        else:
            continue
        for fen, reward in zip(df["fen"].astype(str), df["expected_reward"], strict=True):
            if reward is None or (isinstance(reward, float) and pd.isna(reward)):
                continue
            try:
                add_teacher_value(store, str(fen), float(reward))
                n_lab += 1
            except Exception as exc:  # noqa: BLE001
                print(f"warn value {path.name}: {exc}", file=sys.stderr)
    print(f"labels source={source}: {n_lab:,}")

    rows = slim_fen_value_visits(store, labeled_only=True)
    if not rows:
        print("no labeled rows", file=sys.stderr)
        return 1
    out = pd.DataFrame(rows, columns=["fen", "value", "visits"])
    out["visits"] = out["visits"].astype("int64")
    out = out.sort_values(["visits", "fen"], ascending=[False, True]).reset_index(drop=True)
    pq = out_dir / fen_value_visits_source_filename(source)
    js = pq.with_suffix(".json")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq, index=False)
    payload = [
        {"fen": str(r["fen"]), "value": float(r["value"]), "visits": int(r["visits"])}
        for r in out.to_dict(orient="records")
    ]
    js.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"{source}: {len(out):,} unique → {pq.relative_to(PROJECT_ROOT)} "
        f"+ {js.name}  visits_sum={int(out['visits'].sum()):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
