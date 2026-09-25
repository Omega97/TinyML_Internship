#!/usr/bin/env python3
"""GOAL.md §2 piece-count rule-based dispatcher (Board Input → Bucket ID).

A fixed, handcrafted router: count the pieces on the board (kings included) and
map the count to one of 8 disjoint intervals, the same ``PIECE_COUNT_BUCKETS``
used in the §1 clustering baseline. No training is involved, so the meaningful
evaluation is how well this fixed partition aligns with the learned clusters —
ARI/NMI vs. L1 k-means and vs. sample-gradient k-means labels at B in {2,4,8,16}.

Writes:
  RESULTS/dispatcher/piececount/labels.npy
  RESULTS/dispatcher/piececount/sizes.csv
  RESULTS/dispatcher/piececount/alignment.csv
  RESULTS/dispatcher/piececount/summary.json
  RESULTS/dispatcher/piececount/plots/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROJECT_ROOT
from tinymlinternship.features import piece_square_count
from tinymlinternship.nnue.cluster import fit_minibatch_kmeans
from tinymlinternship.nnue.clustering_eval import (
    overlap_scores,
    piece_bucket_names,
    piece_count_labels,
    piece_counts_from_indices,
)
from tinymlinternship.nnue.moe import load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import load_train_pack

PACK_DIR = PROJECT_ROOT / "data" / "processed" / "board_eval" / "moe" / "moe_b3_1m"
CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
OUT_DIR = PROJECT_ROOT / "RESULTS" / "dispatcher" / "piececount"
COUNT_CHUNK = 4096
L1_CHUNK = 16_384
KS = (2, 4, 8, 16)


def log(message: str) -> None:
    print(message, flush=True)


def build_piece_counts(pack) -> np.ndarray:
    n = len(pack)
    limit = piece_square_count()
    counts = np.empty(n, dtype=np.int16)
    for start in range(0, n, COUNT_CHUNK):
        end = min(start + COUNT_CHUNK, n)
        batch = pack.gather(torch.arange(start, end, dtype=torch.long))
        white_idx = batch["white_idx"].numpy()
        white_mask = batch["white_mask"].numpy()
        counts[start:end] = piece_counts_from_indices(
            white_idx, white_mask.sum(axis=1), limit=limit
        )
        if end == n or (start // COUNT_CHUNK) % 50 == 0:
            log(f"  counts {end:,}/{n:,}")
    return counts


def build_l1(pack, model, device: torch.device) -> np.ndarray:
    n = len(pack)
    hidden = int(model.hidden_dim)
    out = np.empty((n, 2 * hidden), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, n, L1_CHUNK):
            end = min(start + L1_CHUNK, n)
            batch = pack.gather(torch.arange(start, end, dtype=torch.long))
            h = model.l1_concat(
                batch["white_idx"].to(device),
                batch["black_idx"].to(device),
                batch["stm_white"].to(device),
                batch["white_mask"].to(device),
                batch["black_mask"].to(device),
            )
            out[start:end] = h.detach().float().cpu().numpy()
            if end == n or (start // L1_CHUNK) % 10 == 0:
                log(f"  l1 {end:,}/{n:,}")
    return out


def load_gradients(pack_dir: Path) -> np.ndarray:
    g = np.load(pack_dir / "gradients.npy", mmap_mode="r")
    return np.ascontiguousarray(g.astype(np.float32))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_sizes(sizes: np.ndarray, names: list[str]) -> None:
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(sizes.size)
    ax.bar(x, sizes, color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("positions")
    ax.set_title("Piece-count dispatcher bucket sizes")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, plots / "sizes.png")


def plot_alignment(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(KS))
    for target, color, label in (("l1", "#4c72b0", "vs L1 k-means"), ("gradient", "#dd8452", "vs gradient k-means")):
        vals = [next(r["ari"] for r in rows if r["target"] == target and r["b"] == b) for b in KS]
        ax.plot(x, vals, marker="o", color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in KS])
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Piece-count buckets vs learned clusters")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, plots / "alignment.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    max_rows = args.max_rows
    if args.smoke and max_rows is None:
        max_rows = 20_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} | pack={PACK_DIR}")
    pack, _slice_ids, _local_rows = load_train_pack(PACK_DIR)
    n = len(pack)
    log(f"rows {n:,}")
    if max_rows is not None and max_rows < n:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(n, size=max_rows, replace=False)).astype(np.int64)
        pack = pack.index_select(torch.from_numpy(keep))

    log("computing piece counts")
    counts = build_piece_counts(pack)
    piece_labels = piece_count_labels(counts)
    model = load_dual_hidden_checkpoint(CHECKPOINT, device=device)
    log("building L1 activations")
    l1 = build_l1(pack, model, device)
    del pack, model
    log("loading sample gradients")
    gradients = load_gradients(PACK_DIR)
    n = int(l1.shape[0])
    if gradients.shape[0] != n:
        if max_rows is not None and gradients.shape[0] >= n:
            gradients = gradients[:n]
        else:
            raise ValueError(f"gradient rows {gradients.shape[0]} != sample rows {n}")

    names = piece_bucket_names()
    sizes = np.bincount(piece_labels.astype(np.int64), minlength=len(names)).astype(np.int64)
    proportions = (sizes / max(n, 1)).astype(np.float64)

    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "labels.npy", piece_labels)

    sizes_path = OUT_DIR / "sizes.csv"
    with sizes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket", "name", "size", "proportion"])
        for i, (nm, sz, pr) in enumerate(zip(names, sizes, proportions)):
            writer.writerow([i, nm, int(sz), float(pr)])

    alignment_rows: list[dict] = []
    for target, matrix in (("l1", l1), ("gradient", gradients)):
        for b in KS:
            km = fit_minibatch_kmeans(matrix, b, seed=0)
            target_labels = np.asarray(km.labels_, dtype=np.int16)
            ov = overlap_scores(piece_labels, target_labels)
            alignment_rows.append({"target": target, "b": b, "ari": ov["ari"], "nmi": ov["nmi"]})
            log(f"  vs {target} B={b}: ari={ov['ari']:.4f} nmi={ov['nmi']:.4f}")

    align_path = OUT_DIR / "alignment.csv"
    with align_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "b", "ari", "nmi"])
        writer.writeheader()
        writer.writerows(alignment_rows)

    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "source": str(PACK_DIR),
                "checkpoint": str(CHECKPOINT),
                "n_rows": n,
                "n_buckets": len(names),
                "bucket_names": names,
                "sizes": sizes.tolist(),
                "proportions": [float(p) for p in proportions],
                "note": "Fixed piece-count router (8 disjoint intervals). No training; "
                "evaluated by ARI/NMI alignment with L1 and gradient k-means clusters.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_sizes(sizes, names)
    plot_alignment(alignment_rows)
    log(f"wrote {OUT_DIR / 'alignment.csv'} and sizes.csv")
    log("PIECECOUNT_DISPATCHER_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
