#!/usr/bin/env python3
"""Stage the dense base baselines (linear + single-hidden FFNN) into RESULTS/base/.

Reads checkpoints produced by ``scripts/run_base_baselines.py`` from
``models/checkpoints/baselines/`` and copies each run's artifacts into
``RESULTS/base/<Name>/``, writing a ``source.json`` per run and appending the
entries to ``RESULTS/manifest.json`` (``bases``).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import PROJECT_ROOT

CHECKPOINTS = PROJECT_ROOT / "models" / "checkpoints" / "baselines"
RESULTS_BASE = PROJECT_ROOT / "RESULTS" / "base"

COPY_FILES = ("best.pt", "config.json", "history.json", "ce.png", "metrics.json")

BASELINES: tuple[dict[str, Any], ...] = (
    {
        "name": "Linear",
        "run": "linear_wdl",
        "role": "Linear dual-POV WDL baseline (concat [STM||opp] -> 3 logits, no hidden layers)",
    },
    {
        "name": "FFNN_H64",
        "run": "medium_h64",
        "role": "Single-hidden FFNN (H=64) over concat [STM||opp]",
    },
    {
        "name": "FFNN_H128",
        "run": "medium_h128",
        "role": "Single-hidden FFNN (H=128) over concat [STM||opp]",
    },
    {
        "name": "FFNN_H256",
        "run": "medium_h256",
        "role": "Single-hidden FFNN (H=256) over concat [STM||opp]",
    },
    {
        "name": "FFNN2_H64",
        "run": "ffnn2_h64",
        "role": "Dual-hidden FFNN (H1=H2=64) over concat [STM||opp]",
    },
    {
        "name": "FFNN2_H128",
        "run": "ffnn2_h128",
        "role": "Dual-hidden FFNN (H1=H2=128) over concat [STM||opp]",
    },
    {
        "name": "FFNN2_H256",
        "run": "ffnn2_h256",
        "role": "Dual-hidden FFNN (H1=H2=256) over concat [STM||opp]",
    },
)


def _history_ce(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    scored = [row for row in rows if isinstance(row, dict) and "test_ce" in row]
    if not scored:
        return {}
    last = scored[-1]
    return {
        "epochs_logged": int(last.get("epoch", len(scored))),
        "last_test_ce": float(last["test_ce"]),
        "best_test_ce": min(float(row["test_ce"]) for row in scored),
    }


def main() -> int:
    entries: list[dict[str, Any]] = []
    for spec in BASELINES:
        src = CHECKPOINTS / spec["run"]
        dest = RESULTS_BASE / spec["name"]
        dest.mkdir(parents=True, exist_ok=True)
        copied = [name for name in COPY_FILES if (src / name).is_file()]
        for name in copied:
            shutil.copy2(src / name, dest / name)

        cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
        entry: dict[str, Any] = {
            "name": spec["name"],
            "role": spec["role"],
            "source": str(src),
            "architecture": cfg.get("architecture"),
            "hidden_dim": cfg.get("hidden_dim"),
            "hidden1_dim": cfg.get("hidden1_dim"),
            "hidden2_dim": cfg.get("hidden2_dim"),
            "parameters": cfg.get("parameters"),
            "configured_epochs": cfg.get("epochs"),
            "copied": copied,
        }
        entry.update(_history_ce(src / "history.json"))
        entry.update(
            {
                "test_ce": metrics.get("test_ce"),
                "test_mae": metrics.get("test_mae"),
                "test_mse": metrics.get("test_mse"),
                "test_r2": metrics.get("test_r2"),
                "n_test": metrics.get("n_test"),
                "train_nps": metrics.get("train_nps"),
                "infer_nps": metrics.get("infer_nps"),
            }
        )
        (dest / "source.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        entries.append(entry)
        print(f"staged {spec['name']} <- {spec['run']} (test_ce={entry.get('test_ce')})")

    manifest_path = PROJECT_ROOT / "RESULTS" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dense_baselines"] = entries
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"updated {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
