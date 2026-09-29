#!/usr/bin/env python3
"""GOAL.md §4 — Oracle Upper-Bound MoE.

For each $K \\in \\{2, 4, 8, 16\\}$, load the gradient-clustered hard MoE's $K$
expert (L2, head) blocks (frozen L1) and compute the **true** routing upper
bound: for every test position, evaluate all $K$ experts and route to the
expert that yields the minimum per-position CE loss ($\\arg\\min_k \\mathcal{L}_k$).
This is the maximum any router could achieve with that expert set (it
retroactively uses the target to pick the best expert, so it is an unachievable
upper bound, not a real router).

Reports base / actual-MoE (MLP dispatcher) / oracle CE+MAE per $K$. The
actual-MoE numbers are read from ``grad_clustered_k<K>/eval.json`` (Run 7);
only base and oracle are recomputed here on the same 1% test split.

Writes into ``RESULTS/moe/oracle_k<K>/``: eval.json, summary.json, plots/.
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
import torch.nn.functional as F

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
from tinymlinternship.nnue.moe_plots import plot_moe_vs_base

CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
GRAD_ROOT = PROJECT_ROOT / "RESULTS" / "moe"
OUT_ROOT = PROJECT_ROOT / "RESULTS" / "moe"

KS = (2, 4, 8, 16)
EVAL_BATCH = 4096
TEST_FRACTION = 0.01
MAX_TEST = 50_000


def log(message: str) -> None:
    print(message, flush=True)


def load_grad_moe(base, k: int, device: torch.device) -> DualHiddenMoE:
    path = GRAD_ROOT / f"grad_clustered_k{k}" / "moe.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    moe = DualHiddenMoE.from_base(base, int(k)).to(device)
    missing, unexpected = moe.load_state_dict(payload["model_state_dict"], strict=False)
    if unexpected:
        # the MLP dispatcher keys are ignored; experts + frozen L1 are what we need
        unexpected = [k for k in unexpected if not k.startswith("dispatcher.")]
    if any(k.startswith(("experts_", "l1.")) for k in missing):
        raise RuntimeError(f"missing expert/L1 keys from {path}: {missing}")
    moe.freeze_l1()
    return moe


@torch.inference_mode()
def evaluate_oracle(
    base,
    moe: DualHiddenMoE,
    folders: list[Path],
    device: torch.device,
    *,
    max_test: int = MAX_TEST,
    batch_size: int = EVAL_BATCH,
) -> dict:
    planned = plan_split_indices(folders, TEST_FRACTION, seed=0)
    test_parts = subsample_parts([p[2] for p in planned], max_test, seed=1)
    base_ce = base_mae = oracle_ce = oracle_mae = 0.0
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
            stacked = moe.all_expert_logits_from_h(h)  # (B, K, 3)
            log_p = F.log_softmax(stacked.float(), dim=-1)
            nll_e = -(target[:, None, :] * log_p).sum(dim=-1)  # (B, K)
            oracle_ids = nll_e.argmin(dim=-1)  # (B,)
            rows_t = torch.arange(int(idx.size), device=device)
            oracle_logits = stacked[rows_t, oracle_ids, :]  # (B, 3)

            bce, bmae, w = ce_and_mae(base_logits, target, weight)
            oce, omae, _ = ce_and_mae(oracle_logits, target, weight)
            base_ce += float(bce.item())
            base_mae += float(bmae.item())
            oracle_ce += float(oce.item())
            oracle_mae += float(omae.item())
            w_sum += float(w.item())
            n_rows += int(idx.size)
        del ds
    denom = max(w_sum, 1e-8)
    return {
        "n_test": n_rows,
        "base_ce": base_ce / denom,
        "base_mae": base_mae / denom,
        "oracle_ce": oracle_ce / denom,
        "oracle_mae": oracle_mae / denom,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
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
    for k in args.ks:
        run_dir = OUT_ROOT / f"oracle_k{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== oracle K={k} ===")
        t0 = time.perf_counter()
        moe = load_grad_moe(base, k, device)
        ev = evaluate_oracle(base, moe, folders, device, max_test=args.max_test)

        # pull the actual-MoE (dispatcher) result from Run 7 for comparison
        moe_path = GRAD_ROOT / f"grad_clustered_k{k}" / "eval.json"
        actual = {}
        if moe_path.is_file():
            m = json.loads(moe_path.read_text(encoding="utf-8"))
            actual = {"moe_ce": m.get("moe_ce"), "moe_mae": m.get("moe_mae")}

        combined = dict(ev)
        combined.update({kk: vv for kk, vv in actual.items() if vv is not None})
        (run_dir / "eval.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        plot_moe_vs_base(combined, run_dir / "plots" / "moe_vs_base_ce.png")
        summary = {
            "run_name": run_dir.name,
            "checkpoint": str(ckpt),
            "n_experts": k,
            "experts": f"grad_clustered_k{k}/moe.pt",
            "oracle": "argmin_k per-position CE over all K experts (unachievable upper bound)",
            "metrics": combined,
            "seconds": time.perf_counter() - t0,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results[run_dir.name] = combined
        log(
            f"  K={k}: base_ce={combined['base_ce']:.5f} moe_ce={combined.get('moe_ce', float('nan')):.5f} "
            f"oracle_ce={combined['oracle_ce']:.5f} | oracle_mae={combined['oracle_mae']:.5f}"
        )

    (OUT_ROOT / "oracle_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log("ORACLE_MOE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
