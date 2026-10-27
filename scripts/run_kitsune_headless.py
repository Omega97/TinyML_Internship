#!/usr/bin/env python3
"""Headless runner for KITSUNE hard-MoE runs (mirrors the MoE training UI).

Reuses the exact engine behind the UI (``scripts/moe-training-UI/runner.py``)
so results land in ``RESULTS/moe/<run_name>/`` identically to the UI-driven
``load_dual_*_kitsune_*`` runs.

Usage (from repo root)::

    python scripts/run_kitsune_headless.py --k 16 --max-rows 60
    python scripts/run_kitsune_headless.py --k 4 --max-rows 90

Rows are given in millions (matching the ``60M`` / ``90M`` run-name suffix).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "moe-training-UI"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from runner import CancelToken, TrainingConfig, run_moe_training  # noqa: E402

BASE_CHECKPOINT = (
    "/home/omar/jupyterlab/TinyML_Internship/models/checkpoints/nnue/"
    "dual_h128_H128_512x8198_e1000/best.pt"
)
CKPT_NAME = Path(BASE_CHECKPOINT).parent.name  # dual_h128_H128_512x8198_e1000


def build_config(args: argparse.Namespace) -> TrainingConfig:
    max_rows = args.max_rows * 1_000_000
    max_test = args.max_test * 1_000
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = "_".join(
            [
                "load",
                CKPT_NAME,
                "kitsune",
                f"k{args.k}",
                f"bprime{args.micro_factor}x",
                f"a{args.alpha:g}",
                "linear",
                f"ep{args.expert_epochs}",
                "rprop",
                f"{args.max_rows}M_{args.max_test}k",
            ]
        )
    return TrainingConfig(
        technique="hard_moe",
        base_source="load",
        base_checkpoint=BASE_CHECKPOINT,
        h=64,
        H=128,
        base_epochs=20,
        encoder="standard",
        world_model_source="train",
        wm_loss="vicreg",
        wm_tau=0.1,
        wm_epochs=5,
        wm_lr=1e-2,
        wm_normalize=True,
        wm_vicreg_gamma=1.0,
        max_rows=max_rows,
        max_test=max_test,
        clustering="kitsune",
        k=args.k,
        micro_bucket_factor=args.micro_factor,
        kitsune_alpha=args.alpha,
        dispatcher_type="linear",
        dispatcher_hidden=128,
        dispatcher_epochs=5,
        dispatcher_lr=1e-2,
        dispatcher_source="train",
        expert_epochs=args.expert_epochs,
        expert_lr=1e-4,
        expert_lr_end=1e-4,
        expert_optimizer="rprop",
        l1_frozen=True,
        expert_min_ce_decrease=0.01,
        switch_alpha=0.01,
        switch_epochs=40,
        switch_lr=1e-3,
        device="auto",
        run_name=run_name,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--max-rows", type=int, required=True, help="training rows in millions")
    p.add_argument("--max-test", type=int, default=100, help="test rows in thousands")
    p.add_argument("--micro-factor", type=int, default=4, help="B' = factor * K")
    p.add_argument("--alpha", type=float, default=1.0, help="KITSUNE balanced-draft fraction in [0,1]")
    p.add_argument("--expert-epochs", type=int, default=100)
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args(argv)

    cfg = build_config(args)
    t0 = time.time()

    def emit(ev: dict) -> None:
        stage = ev.get("stage", "?")
        event = ev.get("event", "")
        bits = []
        if "msg" in ev and ev["msg"]:
            bits.append(str(ev["msg"]))
        if "progress" in ev and ev["progress"] is not None:
            bits.append(f"progress={ev['progress']}")
        for key in (
            "best_val_acc",
            "sizes",
            "k",
            "clustering",
            "moe_ce",
            "base_ce",
            "oracle_ce",
            "moe_mae",
        ):
            if key in ev and ev[key] is not None:
                bits.append(f"{key}={ev[key]}")
        elapsed = time.time() - t0
        print(
            f"[{elapsed:8.1f}s] {stage:>10}/{event:<8} {' '.join(bits)}".rstrip(),
            flush=True,
        )
        if stage == "done":
            summary = ev.get("summary") or {}
            print("SUMMARY:", json.dumps(summary.get("metrics", {}), default=str), flush=True)

    cancel = CancelToken()
    metrics = run_moe_training(cfg, emit, cancel)
    print("FINAL_METRICS:", json.dumps(metrics, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
