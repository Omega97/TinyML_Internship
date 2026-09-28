#!/usr/bin/env python3
"""Per-bucket KL-divergence estimate: teacher (Lc0) vs base, and teacher vs expert.

For the piece-count MoE (RESULTS/moe/piececount_k8), on the same 10% holdout used
for the reported base/expert CEs. KL(p || q) = CE(p, q) - H(p), where H(p) is the
teacher's own WDL entropy, so it isolates the model's excess loss above the
irreducible uncertainty of the target.

Writes RESULTS/moe/piececount_k8/plots/kl_divergence.png.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from tinymlinternship.config.settings import PROJECT_ROOT
from tinymlinternship.nnue.clustering_eval import piece_bucket_names
from tinymlinternship.nnue.moe import DualHiddenMoE, load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import batches_to_device, load_train_pack

PACK_DIR = PROJECT_ROOT / "data" / "processed" / "board_eval" / "moe" / "moe_b4_2m"
WORK_DIR = PROJECT_ROOT / "RESULTS" / "moe" / "piececount_k8"
CHECKPOINT = (
    PROJECT_ROOT / "models" / "checkpoints" / "nnue" / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
)
N_BUCKETS = 8
HOLDOUT_FRACTION = 0.10
BATCH = 4096


@torch.inference_mode()
def per_position(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (CE, entropy H) per position. KL = CE - H."""
    p = target.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_q = torch.log_softmax(logits.float(), dim=-1)
    ce = -(p * log_q).sum(dim=-1)
    p_safe = torch.clamp(p, min=1e-12)
    h = -(p * torch.log(p_safe)).sum(dim=-1)
    return ce, h


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_dual_hidden_checkpoint(CHECKPOINT, device=device)
    moe = DualHiddenMoE.load(WORK_DIR / "moe.pt", device=device)
    base.eval()
    moe.eval()

    pack, _slice_ids, _local_rows = load_train_pack(PACK_DIR)
    labels = np.load(WORK_DIR / "labels.npy")
    n = len(pack)

    # Reproduce the holdout split exactly as fine_tune_experts (seed 0, 10%).
    rng = np.random.RandomState(0)
    holdout = np.zeros(n, dtype=bool)
    for expert_id in range(N_BUCKETS):
        idx_all = np.flatnonzero(labels.astype(np.int64) == expert_id)
        perm = rng.permutation(idx_all.size)
        n_hold = max(1, int(idx_all.size * HOLDOUT_FRACTION))
        holdout[np.sort(idx_all[perm[:n_hold]])] = True

    names = piece_bucket_names()
    kl_base = np.zeros(N_BUCKETS)
    kl_expert = np.zeros(N_BUCKETS)
    ce_base = np.zeros(N_BUCKETS)
    ce_expert = np.zeros(N_BUCKETS)
    ent = np.zeros(N_BUCKETS)

    for b in range(N_BUCKETS):
        pos = np.flatnonzero(holdout & (labels.astype(np.int64) == b)).astype(np.int64)
        ce_b_sum = ce_e_sum = h_sum = 0.0
        cnt = 0
        for start in range(0, int(pos.size), BATCH):
            idx = pos[start : start + BATCH]
            batch = batches_to_device(pack.gather(torch.from_numpy(idx)), device)
            base_logits = base(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            expert_ids = torch.full((idx.size,), b, dtype=torch.long, device=device)
            expert_logits = moe(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
                expert_ids=expert_ids,
            )
            cb, hb = per_position(base_logits, batch["target"])
            ce_, h_ = per_position(expert_logits, batch["target"])
            ce_b_sum += float(cb.sum().item())
            ce_e_sum += float(ce_.sum().item())
            h_sum += float(hb.sum().item())
            cnt += int(idx.size)
        n_pos = max(cnt, 1)
        ce_base[b] = ce_b_sum / n_pos
        ce_expert[b] = ce_e_sum / n_pos
        ent[b] = h_sum / n_pos
        kl_base[b] = ce_base[b] - ent[b]
        kl_expert[b] = ce_expert[b] - ent[b]
        print(
            f"bucket {names[b]:>6s} | H={ent[b]:.4f} | CE_base={ce_base[b]:.4f} "
            f"CE_expert={ce_expert[b]:.4f} | KL_base={kl_base[b]:.4f} KL_expert={kl_expert[b]:.4f}"
        )

    import matplotlib.pyplot as plt

    x = np.arange(N_BUCKETS)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    w = 0.38
    ax.bar(x - w / 2, kl_base, w, label="teacher vs base", color="#4c72b0")
    ax.bar(x + w / 2, kl_expert, w, label="teacher vs expert", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("piece-count bucket")
    ax.set_ylabel("KL divergence (nats)")
    ax.set_title("KL(teacher ‖ model) per bucket — piece-count MoE (K=8)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = WORK_DIR / "plots" / "kl_divergence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

    payload = {
        "bucket_names": names,
        "entropy": ent.tolist(),
        "ce_base": ce_base.tolist(),
        "ce_expert": ce_expert.tolist(),
        "kl_base": kl_base.tolist(),
        "kl_expert": kl_expert.tolist(),
        "note": "KL(p||q) = CE - H(p), estimated on the 10% holdout of each bucket "
        "(same split as the reported base/expert CEs).",
    }
    (WORK_DIR / "kl_divergence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
