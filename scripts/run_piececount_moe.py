#!/usr/bin/env python3
"""Piece-Count Routed MoE (GOAL.md §4): fixed 8-interval piece-count router + NNUE expert heads.

Routes each position to one of the 8 handcrafted piece-count buckets (the same
``PIECE_COUNT_BUCKETS`` as the §1 clustering baseline), then fine-tunes one
(L2, head) expert per bucket with the shared L1 frozen. No gradient computation
and no dispatcher training are needed — the router is the fixed piece-count rule.

Writes into ``--work-dir``:
  labels.npy, moe.pt, expert_metrics.json, eval.json, summary.json, plots/
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
from tinymlinternship.features import piece_square_count
from tinymlinternship.nnue.clustering_eval import (
    piece_bucket_names,
    piece_count_labels,
    piece_counts_from_indices,
)
from tinymlinternship.nnue.dataset import FenValueVisitsDataset
from tinymlinternship.nnue.moe_pipeline import (
    ce_and_mae,
    configure_torch,
    fine_tune_experts,
    load_base_on_device,
)
from tinymlinternship.nnue.moe_data import (
    batches_to_device,
    load_train_pack,
    plan_split_indices,
    slice_folders,
    subsample_parts,
)
from tinymlinternship.nnue.moe_plots import plot_expert_ce_by_bucket, plot_moe_vs_base

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
PACK_DIR = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe" / "moe_b4_2m"
WORK_DIR = PROJECT_ROOT / "RESULTS" / "moe" / "piececount_k8"
N_BUCKETS = 8
PACK_FILES = ("train_pack.pt", "slice_ids.npy", "local_rows.npy")
COUNT_CHUNK = 4096
EVAL_BATCH = 4096


def log(message: str) -> None:
    print(message, flush=True)


def build_piece_labels(pack) -> np.ndarray:
    n = len(pack)
    limit = piece_square_count()
    counts = np.empty(n, dtype=np.int16)
    started = time.perf_counter()
    for start in range(0, n, COUNT_CHUNK):
        end = min(start + COUNT_CHUNK, n)
        batch = pack.gather(torch.arange(start, end, dtype=torch.long))
        white_idx = batch["white_idx"].numpy()
        white_mask = batch["white_mask"].numpy()
        counts[start:end] = piece_counts_from_indices(
            white_idx, white_mask.sum(axis=1), limit=limit
        )
        if end == n or (start // COUNT_CHUNK) % 50 == 0:
            log(f"  piece counts {end:,}/{n:,} ({time.perf_counter() - started:.0f}s)")
    return piece_count_labels(counts)


def _batch_piece_ids(batch: dict, limit: int) -> np.ndarray:
    white_idx = batch["white_idx"].cpu().numpy()
    white_mask = batch["white_mask"].cpu().numpy()
    counts = piece_counts_from_indices(white_idx, white_mask.sum(axis=1), limit=limit)
    return piece_count_labels(counts).astype(np.int64)


@torch.inference_mode()
def evaluate_piececount_moe(
    base,
    moe,
    folders: list[Path],
    *,
    device: torch.device,
    test_fraction: float = 0.01,
    split_seed: int = 0,
    max_test: int = 50_000,
    subset_seed: int = 1,
    batch_size: int = EVAL_BATCH,
) -> dict[str, float]:
    planned = plan_split_indices(folders, test_fraction, seed=split_seed)
    test_parts = subsample_parts([p[2] for p in planned], max_test, seed=subset_seed)
    limit = piece_square_count()
    base.eval()
    moe.eval()
    base_ce = base_mae = moe_ce = moe_mae = 0.0
    w_sum = 0.0
    n_rows = 0
    for (folder, _tr, _te), rows in zip(planned, test_parts):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            continue
        ds = FenValueVisitsDataset(folder, progress=False)
        for start in range(0, int(rows.size), int(batch_size)):
            idx = rows[start : start + int(batch_size)]
            batch = batches_to_device(ds.gather(idx), device)
            base_logits = base(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            expert_ids = torch.from_numpy(_batch_piece_ids(batch, limit)).to(device)
            moe_logits = moe(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
                expert_ids=expert_ids,
            )
            bce, bmae, w = ce_and_mae(base_logits, batch["target"], batch["weight"])
            mce, mmae, _ = ce_and_mae(moe_logits, batch["target"], batch["weight"])
            base_ce += float(bce.item())
            base_mae += float(bmae.item())
            moe_ce += float(mce.item())
            moe_mae += float(mmae.item())
            w_sum += float(w.item())
            n_rows += int(idx.size)
        del ds
    denom = max(w_sum, 1e-8)
    return {
        "n_test": n_rows,
        "base_ce": base_ce / denom,
        "base_mae": base_mae / denom,
        "moe_ce": moe_ce / denom,
        "moe_mae": moe_mae / denom,
    }


def write_plots(work_dir: Path) -> list[str]:
    plots = work_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    expert_path = work_dir / "expert_metrics.json"
    if expert_path.is_file():
        payload = json.loads(expert_path.read_text(encoding="utf-8"))
        buckets = [b for b in payload.get("buckets", []) if not b.get("skipped")]
        if buckets:
            names = piece_bucket_names()
            written.append(
                str(
                    plot_expert_ce_by_bucket(
                        [float(b["base_hold_ce"]) for b in buckets],
                        [float(b["expert_hold_ce"]) for b in buckets],
                        plots / "expert_ce_by_bucket.png",
                        tick_labels=[names[int(b["expert"])] for b in buckets],
                    )
                )
            )
    eval_path = work_dir / "eval.json"
    if eval_path.is_file():
        metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        written.append(str(plot_moe_vs_base(metrics, plots / "moe_vs_base_ce.png")))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--expert-epochs", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = full pack; else subsample the loaded pack (for smoke)")
    parser.add_argument("--max-test", type=int, default=50_000)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.expert_epochs = 1
        args.max_test = 1024
        if args.max_rows == 0:
            args.max_rows = 50_000

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(device)

    slices_dir = args.slices_dir if args.slices_dir.is_absolute() else (PROJECT_ROOT / args.slices_dir)
    folders = slice_folders(slices_dir)
    if not folders:
        print(f"no slice folders in {slices_dir}", file=sys.stderr)
        return 1

    pack_dir = args.pack_dir if args.pack_dir.is_absolute() else (PROJECT_ROOT / args.pack_dir)
    work_dir = args.work_dir if args.work_dir.is_absolute() else (PROJECT_ROOT / args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else (PROJECT_ROOT / args.checkpoint)
    base = load_base_on_device(ckpt, device)
    log(f"device={device} | checkpoint={ckpt}")

    log(f"linking pack files from {pack_dir} -> {work_dir}")
    for name in PACK_FILES:
        src = pack_dir / name
        if not src.is_file():
            print(f"missing {src}", file=sys.stderr)
            return 1
        dest = work_dir / name
        if dest.is_symlink():
            if dest.resolve() == src.resolve():
                continue
            dest.unlink()
        elif dest.exists():
            continue
        dest.symlink_to(src.resolve())

    pack, _slice_ids, _local_rows = load_train_pack(work_dir)
    n = len(pack)
    log(f"train pack rows {n:,}")
    if args.max_rows > 0 and args.max_rows < n:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(n, size=args.max_rows, replace=False)).astype(np.int64)
        pack = pack.index_select(torch.from_numpy(keep))
        n = len(pack)
        log(f"subsampled pack to {n:,} rows")
    labels = build_piece_labels(pack)
    np.save(work_dir / "labels.npy", labels.astype(np.int16))
    sizes = np.bincount(labels.astype(np.int64), minlength=N_BUCKETS).astype(np.int64)
    log(f"piece-count sizes {sizes.tolist()}")

    moe = fine_tune_experts(
        base,
        work_dir,
        folders,
        device=device,
        n_clusters=N_BUCKETS,
        expert_labels="piece_count",
        epochs=args.expert_epochs,
    )

    metrics = evaluate_piececount_moe(
        base,
        moe,
        folders,
        device=device,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        max_test=args.max_test,
    )
    (work_dir / "eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(
        f"eval n={metrics['n_test']:,} | base_ce={metrics['base_ce']:.5f} | "
        f"moe_ce={metrics['moe_ce']:.5f}"
    )

    plots = write_plots(work_dir)
    summary = {
        "run_name": work_dir.name,
        "checkpoint": str(ckpt),
        "n_buckets": N_BUCKETS,
        "router": "piece_count",
        "expert_labels": "piece_count",
        "expert_epochs": args.expert_epochs,
        "train_rows": n,
        "bucket_names": piece_bucket_names(),
        "bucket_sizes": sizes.tolist(),
        "metrics": metrics,
        "plots": plots,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {work_dir / 'summary.json'}")
    log("PIECECOUNT_MOE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
