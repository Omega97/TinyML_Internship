#!/usr/bin/env python3
"""Download ChessBench (DeepMind searchless_chess) test splits in Research .bag format."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import CHESSBENCH_RAW_DIR

BASE = "https://storage.googleapis.com/searchless_chess/data"

FILES: dict[str, tuple[str, str]] = {
    "action_value": ("test/action_value_data.bag", "action-value: (fen, move_uci, win_prob)"),
    "state_value": ("test/state_value_data.bag", "state-value: (fen, win_prob)"),
    "behavioral_cloning": ("test/behavioral_cloning_data.bag", "BC: (fen, move_uci)"),
    "puzzles": ("puzzles.csv", "10k puzzle test CSV"),
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"skip (exists): {dest} ({dest.stat().st_size:,} B)")
        return
    print(f"downloading {url} …")
    urllib.request.urlretrieve(url, dest)
    print(f"saved {dest} ({dest.stat().st_size:,} B)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ChessBench Research-format files")
    parser.add_argument(
        "--files",
        nargs="*",
        choices=list(FILES),
        default=["state_value", "action_value"],
        help="Which files to fetch (default: state_value + action_value test splits)",
    )
    args = parser.parse_args(argv)

    for key in args.files:
        rel, desc = FILES[key]
        dest = CHESSBENCH_RAW_DIR / rel
        download(f"{BASE}/{rel}", dest)
        print(f"  {desc}")

    print(f"\nChessBench root: {CHESSBENCH_RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())