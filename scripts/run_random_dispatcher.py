#!/usr/bin/env python3
"""GOAL.md §4 — Random Dispatcher Control.

For every dispatcher-routed hard MoE (L1-clustered and gradient-clustered),
replace the trained MLP dispatcher with a **random router** that assigns each
test position to a uniformly-random expert. This isolates routing quality from
raw capacity: if the MoE's (tiny) gain over the base comes from routing rather
than from simply having K experts, random routing should fall back to ≈ base.

Reports, per variant and K: random-router CE/MAE, the trained-dispatcher MoE
CE/MAE (read from the variant's eval.json), and the base CE/MAE.

Writes into ``RESULTS/moe/random_<variant>_k<K>/``: eval.json, summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import FenValueVisitsDataset
from tinymlinternship.nnue.moe import DualHiddenMoE, load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import (
    batches_to_device,
    plan_split_indices,
    slice_folders,
    subsample_parts,
)
from tinymlinternship.nnue.moe_pipeline import ce_and_mae, configure_torch

CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
MOE_ROOT = PROJECT_ROOT / "RESULTS" / "moe"
OUT_ROOT = PROJECT_ROOT / "RESULTS" / "moe"

VARIANTS = ("l1_clustered", "grad_clustered")
KS = (2, 4, 8, 16)
EVAL_BATCH = 4096
TEST_FRACTION = 0.01
MAX_TEST = 50_000
RANDOM_SEED = 0


def log(message: str) -> None:
    print(message, flush=True)


def load_hard_moe(base, variant: str, k: int, device: torch.device) -> DualHiddenMoE:
    path = MOE_ROOT / f"{variant}_k{k}" / "moe.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    moe = DualHiddenMoE.from_base(base, int(k)).to(device)
    missing, unexpected = moe.load_state_dict(payload["model_state_dict"], strict=False)
    if unexpected:
        unexpected = [key for key in unexpected if not key.startswith("dispatcher.")]
    if any(key.startswith(("experts_", "l1.")) for key in missing):
        raise RuntimeError(f"missing expert/L1 keys from {path}: {missing}")
    moe.freeze_l1()
    return moe


@torch.inference_mode()
def evaluate_random(
    base,
    moe: DualHiddenMoE,
    folders: list[Path],
    device: torch.device,
    *,
    max_test: int = MAX_TEST,
    batch_size: int = EVAL_BATCH,
    seed: int = RANDOM_SEED,
) -> dict:
    planned = plan_split_indices(folders, TEST_FRACTION, seed=0)
    test_parts = subsample_parts([p[2] for p in planned], max_test, seed=1)
    g = torch.Generator(device=device).manual_seed(int(seed))
    base_ce = base_mae = rand_ce = rand_mae = 0.0
    w_sum = 0.0
    n_rows = 0
    base.eval()
    moe.eval()
    for (folder, _tr, _te), rows in zip(planned, test_parts):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            continue
        ds = FenValueVisitsDataset(folder, progress=False)
        for start in range(0, int(rows.size), int(batch_size)):
            idx = rows[start : start + int(batch_size)]
            batch = batches_to_device(ds.gather(idx), device)
            target = batch["target"].float()
            weight = batch["weight"]
            base_logits = base(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            h = base.l1_concat(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            b = int(idx.size)
            expert_ids = torch.randint(0, moe.n_experts, (b,), device=device, generator=g)
            rand_logits, _ = moe.route_from_h(h, expert_ids=expert_ids)

            bce, bmae, w = ce_and_mae(base_logits, target, weight)
            rce, rmae, _ = ce_and_mae(rand_logits, target, weight)
            base_ce += float(bce.item())
            base_mae += float(bmae.item())
            rand_ce += float(rce.item())
            rand_mae += float(rmae.item())
            w_sum += float(w.item())
            n_rows += int(idx.size)
        del ds
    denom = max(w_sum, 1e-8)
    return {
        "n_test": n_rows,
        "base_ce": base_ce / denom,
        "base_mae": base_mae / denom,
        "random_ce": rand_ce / denom,
        "random_mae": rand_mae / denom,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--variants", type=str, nargs="+", default=list(VARIANTS))
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--max-test", type=int, default=MAX_TEST)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.max_test = 1024

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(device)

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else (PROJECT_ROOT / args.checkpoint)
    base = load_dual_hidden_checkpoint(ckpt, device=device)
    log(f"device={device} | checkpoint={ckpt}")

    slices_dir = args.slices_dir if args.slices_dir.is_absolute() else (PROJECT_ROOT / args.slices_dir)
    folders = slice_folders(slices_dir)
    if not folders:
        log(f"no slice folders in {slices_dir}")
        return 1

    results: dict[str, dict] = {}
    for variant in args.variants:
        for k in args.ks:
            run_dir = OUT_ROOT / f"random_{variant}_k{k}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log(f"=== random dispatcher | {variant} | K={k} ===")
            t0 = time.perf_counter()
            moe = load_hard_moe(base, variant, k, device)
            ev = evaluate_random(base, moe, folders, device, max_test=args.max_test)

            # pull the trained-dispatcher MoE result for comparison
            moe_path = MOE_ROOT / f"{variant}_k{k}" / "eval.json"
            actual = {}
            if moe_path.is_file():
                m = json.loads(moe_path.read_text(encoding="utf-8"))
                actual = {"moe_ce": m.get("moe_ce"), "moe_mae": m.get("moe_mae")}

            combined = dict(ev)
            combined.update({kk: vv for kk, vv in actual.items() if vv is not None})
            (run_dir / "eval.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
            summary = {
                "run_name": run_dir.name,
                "checkpoint": str(ckpt),
                "variant": variant,
                "n_experts": k,
                "router": "random (uniform) over K experts",
                "random_seed": RANDOM_SEED,
                "experts": f"{variant}_k{k}/moe.pt",
                "metrics": combined,
                "seconds": time.perf_counter() - t0,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            results[run_dir.name] = combined
            log(
                f"  {variant} K={k}: base_ce={combined['base_ce']:.5f} moe_ce={combined.get('moe_ce', float('nan')):.5f} "
                f"random_ce={combined['random_ce']:.5f}"
            )

    (OUT_ROOT / "random_dispatcher_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log("RANDOM_DISPATCHER_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
