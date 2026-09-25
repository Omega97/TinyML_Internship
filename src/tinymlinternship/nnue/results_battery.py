"""Stage thesis RESULTS and describe the MoE cross-entropy battery.

The five method steps already live in ``moe_pipeline``. This module copies
finished base nets and the two existing MoE runs into ``RESULTS/``, and it
lists the runs that still have to be trained: mini-batch k-means and DBSCAN
at B = 2, 4, 8.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinymlinternship.config.settings import PROJECT_ROOT
from tinymlinternship.nnue.moe_pipeline import link_gradient_cache
from tinymlinternship.nnue.moe_plots import plot_expert_ce_by_bucket, plot_run_ce_bars

REF_CHECKPOINT = "models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt"
REF_GRAD_DIR = "data/processed/board_eval/moe/moe_b4_2m"
DEFAULT_MAX_ROWS = 2_000_000
DEFAULT_MAX_TEST = 50_000

CANONICAL_BASES: tuple[dict[str, str], ...] = (
    {
        "name": "W32_H64",
        "source": "models/checkpoints/nnue/dual_h128_H256_e100_bpe512_bs10000",
        "role": "smallest DualHidden NNUE kept from the size sweep",
    },
    {
        "name": "W64_H128",
        "source": "models/checkpoints/nnue/dual_h128_H128_512x8198_e1000",
        "role": "small DualHidden NNUE",
    },
    {
        "name": "W128_H128",
        "source": "models/checkpoints/nnue/dual_h128_H128_e100_bpe512_bs10000",
        "role": "medium DualHidden NNUE with a narrower head",
    },
    {
        "name": "W128_H256",
        "source": "models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000",
        "role": "reference base; the existing 2M gradient cache was computed from this checkpoint",
    },
    {
        "name": "W256_H512",
        "source": "models/checkpoints/nnue/dual_h256_H512_e200_bpe512_bs10000",
        "role": "largest DualHidden NNUE",
    },
)

SCALING_FILES: tuple[str, ...] = (
    "plots/nnue_ce_vs_train_size/nnue_ce_vs_train_size.png",
    "plots/nnue_ce_vs_train_size/nnue_ce_vs_train_size.pdf",
    "plots/nnue_ce_vs_train_size/variable_dataset_size_ce_256.png",
    "plots/nnue_ce_vs_train_size/variable_dataset_size_ce_256.pdf",
    "plots/nnue_ce_vs_train_size/variable_dataset_size_ce_256.txt",
    "plots/nnue_ce_vs_train_size/variable_dataset_size_ce_64.png",
    "plots/nnue_ce_vs_train_size/variable_dataset_size_ce_64.pdf",
    "plots/nnue_ce_vs_train_size/variable_dataset_size_ce_64.txt",
)

PRELIMINARY: tuple[dict[str, str], ...] = (
    {
        "name": "kmeans_b3_1m",
        "source": "data/processed/board_eval/moe/moe_b3_1m",
        "plots": "plots/MoE/run-001",
    },
    {
        "name": "kmeans_b4_2m",
        "source": "data/processed/board_eval/moe/moe_b4_2m",
        "plots": "plots/MoE",
    },
)

PRELIM_FILES: tuple[str, ...] = (
    "moe.pt",
    "dispatcher.pt",
    "eval.json",
    "expert_metrics.json",
    "diagnostics.json",
    "summary.json",
    "dispatcher_history.json",
    "centroids.npy",
    "labels.npy",
)

BASE_COPY_FILES: tuple[str, ...] = ("best.pt", "config.json", "history.json", "ce.png")

_KMEANS_B = (2, 4, 8)
_DBSCAN_B = (2, 4, 8)


@dataclass(frozen=True)
class BatteryRun:
    rel_dir: str
    checkpoint: str
    algorithm: str
    n_clusters: int
    max_rows: int
    expert_epochs: int
    expert_labels: str
    reuse_gradients: str | None
    wave: str
    note: str


def _kmeans_run(
    *,
    rel_dir: str,
    checkpoint: str,
    n_clusters: int,
    expert_epochs: int,
    expert_labels: str,
    reuse_gradients: str | None,
    wave: str,
    note: str,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> BatteryRun:
    return BatteryRun(
        rel_dir=rel_dir,
        checkpoint=checkpoint,
        algorithm="minibatch_kmeans",
        n_clusters=n_clusters,
        max_rows=max_rows,
        expert_epochs=expert_epochs,
        expert_labels=expert_labels,
        reuse_gradients=reuse_gradients,
        wave=wave,
        note=note,
    )


def reference_runs() -> list[BatteryRun]:
    """W128/H256 grid. Wave 1 is the CE comparison. Wave 2 checks two training choices."""
    runs: list[BatteryRun] = []
    for b in _KMEANS_B:
        runs.append(
            _kmeans_run(
                rel_dir=f"moe/W128_H256/kmeans_b{b}",
                checkpoint=REF_CHECKPOINT,
                n_clusters=b,
                expert_epochs=2,
                expert_labels="dispatcher",
                reuse_gradients=REF_GRAD_DIR,
                wave="1",
                note="Reference net, mini-batch k-means, two expert epochs, dispatcher labels.",
            )
        )
    for b in _DBSCAN_B:
        runs.append(
            BatteryRun(
                rel_dir=f"moe/W128_H256/dbscan_b{b}",
                checkpoint=REF_CHECKPOINT,
                algorithm="dbscan",
                n_clusters=b,
                max_rows=DEFAULT_MAX_ROWS,
                expert_epochs=2,
                expert_labels="dispatcher",
                reuse_gradients=REF_GRAD_DIR,
                wave="1",
                note=(
                    "DBSCAN ε is chosen on the 0.1 quantile grid so the dense-cluster "
                    "count is as close as possible to this B. Noise is assigned to the "
                    "nearest core centroid. The density fit uses at most 100k rows."
                ),
            )
        )
    for b in _KMEANS_B:
        runs.append(
            _kmeans_run(
                rel_dir=f"moe/W128_H256/kmeans_b{b}_clusterlabels",
                checkpoint=REF_CHECKPOINT,
                n_clusters=b,
                expert_epochs=2,
                expert_labels="kmeans",
                reuse_gradients=REF_GRAD_DIR,
                wave="2",
                note="Same k-means partition, but experts train on cluster labels rather than dispatcher labels.",
            )
        )
    for b in _KMEANS_B:
        runs.append(
            _kmeans_run(
                rel_dir=f"moe/W128_H256/kmeans_b{b}_e8",
                checkpoint=REF_CHECKPOINT,
                n_clusters=b,
                expert_epochs=8,
                expert_labels="dispatcher",
                reuse_gradients=REF_GRAD_DIR,
                wave="2",
                note="Eight expert epochs. The two-epoch runs left expert holdout CE flat against the base head.",
            )
        )
    return runs


def size_probe_runs() -> list[BatteryRun]:
    """One k-means B=4 probe per non-reference width. Each probe computes its own gradients."""
    runs: list[BatteryRun] = []
    for spec in CANONICAL_BASES:
        if spec["name"] == "W128_H256":
            continue
        runs.append(
            _kmeans_run(
                rel_dir=f"moe/{spec['name']}/kmeans_b4",
                checkpoint=f"{spec['source']}/best.pt",
                n_clusters=4,
                expert_epochs=2,
                expert_labels="dispatcher",
                reuse_gradients=None,
                wave="3",
                note=f"Size probe for {spec['name']}. Fresh 2M gradient pass, then k-means B=4.",
            )
        )
    return runs


def all_runs() -> list[BatteryRun]:
    return reference_runs() + size_probe_runs()


def select_runs(wave: str | None = None) -> list[BatteryRun]:
    runs = all_runs()
    if wave is None or wave == "all":
        return runs
    wanted = {part.strip() for part in str(wave).split(",") if part.strip()}
    return [run for run in runs if run.wave in wanted]


def pipeline_argv(run: BatteryRun, *, python: str | None = None) -> list[str]:
    py = python or sys.executable
    work = f"RESULTS/{run.rel_dir}"
    argv = [
        py,
        "scripts/run_moe_pipeline.py",
        "--checkpoint",
        run.checkpoint,
        "--work-dir",
        work,
        "--plots-dir",
        f"{work}/plots",
        "--run-name",
        Path(run.rel_dir).name,
        "--max-rows",
        str(run.max_rows),
        "--max-test",
        str(DEFAULT_MAX_TEST),
        "--n-clusters",
        str(run.n_clusters),
        "--algorithm",
        run.algorithm,
        "--expert-labels",
        run.expert_labels,
        "--expert-epochs",
        str(run.expert_epochs),
        "--dispatcher-epochs",
        "8",
    ]
    if run.reuse_gradients:
        argv.extend(["--gradient-cache", run.reuse_gradients])
    return argv


def format_pipeline_command(run: BatteryRun, *, python: str | None = None) -> str:
    return " ".join(pipeline_argv(run, python=python))


def _copy_file(src: Path, dest: Path) -> bool:
    if not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _history_ce(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("history") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    scored = [row for row in rows if isinstance(row, dict) and "test_ce" in row]
    if not scored:
        return {"epochs_logged": len(rows)}
    last = scored[-1]
    return {
        "epochs_logged": int(last.get("epoch", len(scored))),
        "last_test_ce": float(last["test_ce"]),
        "best_test_ce": min(float(row["test_ce"]) for row in scored),
    }


def stage_existing(root: Path | None = None) -> dict[str, Any]:
    """Copy finished base checkpoints and the two existing MoE runs into RESULTS."""
    root = Path(root or (PROJECT_ROOT / "RESULTS"))
    bases: list[dict[str, Any]] = []
    for spec in CANONICAL_BASES:
        src = PROJECT_ROOT / spec["source"]
        dest = root / "base" / spec["name"]
        copied = [name for name in BASE_COPY_FILES if _copy_file(src / name, dest / name)]
        cfg_path = src / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        entry: dict[str, Any] = {
            "name": spec["name"],
            "role": spec["role"],
            "source": spec["source"],
            "hidden_dim": cfg.get("hidden_dim"),
            "hidden2_dim": cfg.get("hidden2_dim"),
            "parameters": cfg.get("parameters"),
            "configured_epochs": cfg.get("epochs"),
            "copied": copied,
        }
        entry.update(_history_ce(src / "history.json"))
        (dest / "source.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        bases.append(entry)

    scaling_dest = root / "base" / "scaling"
    scaling = [
        Path(rel).name
        for rel in SCALING_FILES
        if _copy_file(PROJECT_ROOT / rel, scaling_dest / Path(rel).name)
    ]

    preliminary: list[dict[str, Any]] = []
    for spec in PRELIMINARY:
        src = PROJECT_ROOT / spec["source"]
        dest = root / "moe" / "preliminary" / spec["name"]
        copied = [name for name in PRELIM_FILES if _copy_file(src / name, dest / name)]
        plot_src = PROJECT_ROOT / spec["plots"]
        plot_copied: list[str] = []
        if plot_src.is_dir():
            for png in sorted(plot_src.glob("*.png")):
                if _copy_file(png, dest / "plots" / png.name):
                    plot_copied.append(png.name)
        summary: dict[str, Any] = {}
        summary_path = src / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        preliminary.append(
            {
                "name": spec["name"],
                "source": spec["source"],
                "plots_source": spec["plots"],
                "copied": copied,
                "plots": plot_copied,
                "n_clusters": summary.get("n_clusters"),
                "max_rows": summary.get("max_rows"),
                "algorithm": "minibatch_kmeans",
                "expert_labels": summary.get("expert_labels"),
                "checkpoint": summary.get("checkpoint"),
                "metrics": summary.get("metrics"),
            }
        )

    manifest = {
        "bases": bases,
        "scaling_files": scaling,
        "preliminary_moe": preliminary,
        "reference_checkpoint": REF_CHECKPOINT,
        "reference_gradient_cache": REF_GRAD_DIR,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(root)
    return manifest


def collect_eval_rows(root: Path) -> list[dict[str, Any]]:
    moe = root / "moe"
    if not moe.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for eval_path in sorted(moe.rglob("eval.json")):
        if eval_path.parent.name == "plots":
            continue
        metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        rel = eval_path.parent.relative_to(root).as_posix()
        row: dict[str, Any] = {"name": rel.removeprefix("moe/"), "path": rel}
        row.update(metrics)
        expert_path = eval_path.parent / "expert_metrics.json"
        if expert_path.is_file():
            payload = json.loads(expert_path.read_text(encoding="utf-8"))
            buckets = [
                bucket
                for bucket in payload.get("buckets", [])
                if not bucket.get("skipped") and "base_hold_ce" in bucket
            ]
            row["expert_labels"] = payload.get("expert_labels")
            row["n_experts_scored"] = len(buckets)
            if buckets:
                weights = [float(bucket.get("n_hold") or 1.0) for bucket in buckets]
                weight_sum = sum(weights) or 1.0
                row["base_hold_ce"] = (
                    sum(float(bucket["base_hold_ce"]) * w for bucket, w in zip(buckets, weights))
                    / weight_sum
                )
                row["expert_hold_ce"] = (
                    sum(float(bucket["expert_hold_ce"]) * w for bucket, w in zip(buckets, weights))
                    / weight_sum
                )
        diag_path = eval_path.parent / "diagnostics.json"
        if diag_path.is_file():
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            row["algorithm"] = diag.get("algorithm", "minibatch_kmeans")
            row["cluster_count"] = diag.get("n_clusters")
            row["dbscan_epsilon"] = diag.get("dbscan_epsilon")
            row["inertia"] = diag.get("inertia")
            row["n_noise"] = diag.get("n_noise")
        history_path = eval_path.parent / "dispatcher_history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            row["dispatcher_val_acc"] = history.get("best_val_acc")
        rows.append(row)
    return rows


def _bucket_bars(root: Path) -> tuple[list[str], list[float], list[float]]:
    names: list[str] = []
    base: list[float] = []
    expert: list[float] = []
    moe = root / "moe"
    if not moe.is_dir():
        return names, base, expert
    for path in sorted(moe.rglob("expert_metrics.json")):
        if path.parent.name == "plots":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rel = path.parent.relative_to(moe).as_posix()
        for bucket in payload.get("buckets", []):
            if bucket.get("skipped") or "base_hold_ce" not in bucket:
                continue
            names.append(f"{rel} e{bucket.get('expert')}")
            base.append(float(bucket["base_hold_ce"]))
            expert.append(float(bucket["expert_hold_ce"]))
    return names, base, expert


def write_summary(root: Path) -> list[dict[str, Any]]:
    """Write RESULTS/tables/ce.csv and the comparison bar charts."""
    root = Path(root)
    rows = collect_eval_rows(root)
    table = root / "tables"
    table.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "algorithm",
        "cluster_count",
        "dbscan_epsilon",
        "n_test",
        "base_ce",
        "moe_ce",
        "oracle_ce",
        "base_mae",
        "moe_mae",
        "oracle_mae",
        "base_hold_ce",
        "expert_hold_ce",
        "n_experts_scored",
        "expert_labels",
        "inertia",
        "n_noise",
        "dispatcher_val_acc",
        "path",
    ]
    with (table / "ce.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        plot_run_ce_bars(rows, root / "plots" / "test_ce.png", title="Test CE, base vs MoE")
        hold_rows = [
            {"name": row["name"], "base_ce": row["base_hold_ce"], "moe_ce": row["expert_hold_ce"]}
            for row in rows
            if "base_hold_ce" in row and "expert_hold_ce" in row
        ]
        if hold_rows:
            plot_run_ce_bars(
                hold_rows,
                root / "plots" / "bucket_holdout_ce.png",
                title="Mean bucket-holdout CE, base head vs expert",
                series_labels={"moe_ce": "expert"},
            )
    tick, base_ce, expert_ce = _bucket_bars(root)
    if tick:
        plot_expert_ce_by_bucket(
            base_ce,
            expert_ce,
            root / "plots" / "bucket_ce_bars.png",
            title="Per-bucket holdout CE, base head vs expert",
            tick_labels=tick,
        )
    return rows


def run_battery(
    runs: list[BatteryRun],
    *,
    python: str | None = None,
) -> None:
    """Link shared gradients where requested, then run each pipeline cell."""
    py = python or sys.executable
    for run in runs:
        work = PROJECT_ROOT / "RESULTS" / run.rel_dir
        if run.reuse_gradients:
            link_gradient_cache(PROJECT_ROOT / run.reuse_gradients, work)
        print(format_pipeline_command(run, python=py), flush=True)
        subprocess.run(pipeline_argv(run, python=py), cwd=PROJECT_ROOT, check=True)
    write_summary(PROJECT_ROOT / "RESULTS")
