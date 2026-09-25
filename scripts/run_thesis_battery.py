#!/usr/bin/env python3
"""Print or run the thesis MoE battery (k-means and DBSCAN, B = 2, 4, 8).

Default is to print the commands. ``--run`` executes a wave (default wave 1).
``--summary`` rebuilds RESULTS/tables/ce.csv and the comparison plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import PROJECT_ROOT
from tinymlinternship.nnue.results_battery import (
    format_pipeline_command,
    run_battery,
    select_runs,
    write_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wave",
        default=None,
        help="1 (reference grid), 2 (label source and longer fine-tune), 3 (other widths), or all.",
    )
    parser.add_argument("--print", dest="do_print", action="store_true")
    parser.add_argument("--run", dest="do_run", action="store_true")
    parser.add_argument("--summary", dest="do_summary", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    if args.do_summary and not args.do_print and not args.do_run:
        rows = write_summary(PROJECT_ROOT / "RESULTS")
        print(f"wrote {len(rows)} rows to RESULTS/tables/ce.csv")
        return 0

    wave = args.wave if args.wave is not None else ("1" if args.do_run else "all")
    runs = select_runs(wave)
    if not runs:
        print(f"no runs for wave {wave!r}", file=sys.stderr)
        return 1

    print_runs = args.do_print or not args.do_run
    if print_runs:
        current = None
        for run in runs:
            if run.wave != current:
                current = run.wave
                print(f"\n# wave {current}")
            print(f"# {run.note}")
            print(format_pipeline_command(run, python=args.python))
            print()
    if args.do_run:
        run_battery(runs, python=args.python)
    elif args.do_summary:
        write_summary(PROJECT_ROOT / "RESULTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
