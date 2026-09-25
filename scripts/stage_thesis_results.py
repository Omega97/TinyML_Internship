#!/usr/bin/env python3
"""Copy finished base nets and the existing MoE runs into RESULTS/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import PROJECT_ROOT
from tinymlinternship.nnue.results_battery import stage_existing


def main(argv: list[str] | None = None) -> int:
    del argv
    manifest = stage_existing(PROJECT_ROOT / "RESULTS")
    n_base = len(manifest["bases"])
    n_moe = len(manifest["preliminary_moe"])
    print(f"staged {n_base} base nets and {n_moe} preliminary MoE runs into RESULTS/")
    print(f"summary: {PROJECT_ROOT / 'RESULTS' / 'tables' / 'ce.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
