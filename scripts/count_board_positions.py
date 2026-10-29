#!/usr/bin/env python3
"""Count board positions across unpacked slice folders.

Each unpacked slice folder holds ``features.meta.json`` (with ``n_rows``) plus a
``features.npz`` (the encoded rows). This walks a directory tree, finds every
slice folder, and sums the row counts — one row = one unique board position
(EPD). ``.zip`` archives are ignored (only directories are traversed).

Example::

    python scripts/count_board_positions.py
    python scripts/count_board_positions.py --root data/processed/board_eval_2
    python scripts/count_board_positions.py --json-out RESULTS/dataset_count.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = _PROJECT_ROOT / "data" / "processed" / "board_eval_2"
SLICE_META_NAME = "features.meta.json"


def _resolve(path: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def discover_slice_folders(root: Path) -> list[Path]:
    """Directories under ``root`` that contain a ``features.meta.json``."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted({meta.parent for meta in root.rglob(SLICE_META_NAME)})


def count_positions(root: Path) -> dict:
    folders = discover_slice_folders(root)
    per_group: dict[str, dict] = {}
    total = 0
    total_slices = 0
    for folder in folders:
        data = json.loads((folder / SLICE_META_NAME).read_text(encoding="utf-8"))
        n = int(data["n_rows"])
        rel = folder.relative_to(root)
        group = rel.parts[0] if len(rel.parts) > 1 else folder.name
        g = per_group.setdefault(group, {"slices": 0, "rows": 0})
        g["slices"] += 1
        g["rows"] += n
        total += n
        total_slices += 1
    return {
        "root": str(root),
        "n_slices": total_slices,
        "total_rows": total,
        "groups": [
            {"group": name, "slices": g["slices"], "rows": g["rows"]}
            for name, g in sorted(per_group.items())
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root of unpacked slice folders (default: {DEFAULT_ROOT.relative_to(_PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the same report as JSON",
    )
    args = parser.parse_args(argv)

    root = _resolve(args.root)
    report = count_positions(root)
    if report["n_slices"] == 0:
        print(f"no unpacked slice folders (features.meta.json) under {root}", file=sys.stderr)
        return 1

    for g in report["groups"]:
        print(f"{g['group']}: {g['slices']} slices, {g['rows']:,} positions")
    print("---")
    print(f"slices: {report['n_slices']}")
    print(f"board positions (total rows): {report['total_rows']:,}")

    if args.json_out is not None:
        out = _resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"json → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
