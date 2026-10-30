"""Training engine for the MoE training UI.

Runs the hard-MoE pipeline (base model -> data pack -> clustering -> dispatcher
-> per-expert fine-tune -> evaluation) and reports fine-grained progress to a
callback. The callback receives plain dicts (one per event) that the UI turns
into per-stage progress bars and metric readouts.

Events share a ``"stage"`` key. The UI-visible stages are:

* ``base``       — base model load or (mini) training
* ``loading``    — data pack build + gradient cache
* ``cluster``    — bucket assignment (gradient / L1 k-means / piece-count rule)
* ``dispatcher`` — router training (epoch / val-acc)
* ``experts``    — per-expert fine-tune (epoch / holdout CE)
* ``eval``       — full-MoE evaluation (base / MoE / perfect-routing / best-expert)
* ``done``       — terminal summary
"""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, SARDINE_MODELS_DIR
from tinymlinternship.data.board_store import BOARD_EVAL_2_DIR_NAME
from tinymlinternship.features import piece_square_count
from tinymlinternship.nnue.clustering_eval import piece_count_labels, piece_counts_from_indices
from tinymlinternship.nnue.cluster import (
    assign_to_centroids,
    cluster_diagnostics,
    fit_minibatch_kmeans,
    save_cluster_run,
)
from tinymlinternship.nnue.dataset import FenValueVisitsDataset
from tinymlinternship.nnue.model import DualHiddenNNUE
from tinymlinternship.nnue.moe import (
    DualHiddenMoE,
    KitsuneDispatcher,
    LinearDispatcher,
    MLPDispatcher,
    SoftGatedMoE,
    checkpoint_encoder_info,
    dispatcher_checkpoint_info,
    load_dispatcher,
    load_dual_hidden_checkpoint,
    save_dispatcher,
)
from tinymlinternship.nnue.world_model import train_world_model_encoder
from tinymlinternship.nnue.moe_data import (
    batches_to_device,
    load_train_pack,
    pack_parts,
    plan_split_indices,
    save_train_pack,
    slice_folders,
    subsample_parts,
    total_rows,
)
from tinymlinternship.nnue.moe_pipeline import (
    DEFAULT_REDUCE_DIM,
    ce_and_mae,
    compute_gradients,
    configure_torch,
)
from tinymlinternship.nnue.moe_plots import (
    plot_cluster_magnitude_sorted,
    plot_gradient_magnitude_by_bucket,
    plot_overfit_latency,
)
from tinymlinternship.nnue.optimizers import Rprop
from tinymlinternship.nnue.sample_gradients import (
    head_parameter_dim,
    make_projection_matrix,
    sample_head_gradients,
)

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_2_DIR_NAME
DEFAULT_CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
UI_PACK_ROOT = PROCESSED_DATA_DIR / BOARD_EVAL_2_DIR_NAME / "moe"
DISPATCHER_CHECKPOINTS_DIR = SARDINE_MODELS_DIR / "checkpoints" / "dispatchers"
MOE_RESULTS_ROOT = PROJECT_ROOT / "RESULTS" / "moe"

L1_CHUNK = 16_384
CLUSTER_CHUNK = 1_000_000
# KITSUNE clustering is bounded to a fixed representative subset so the whole
# cluster stage is O(1) in the dataset size (per-sample head gradients are the
# expensive part). The subset is deterministic (seed 0) and shared by the L1
# build, gradient computation, and micro-labeling so they stay aligned.
CLUSTER_MAX_ROWS = 5_000_000
CLUSTER_SUBSET_SEED = 0
DISPATCHER_BATCH = 4096
DISPATCHER_SUBSET_SEED = 0
EXPERT_BATCH = 2048
EVAL_BATCH = 4096
SWITCH_BATCH = 2048
PIECE_COUNT_BUCKETS_K = 8
# Number of training rows to re-evaluate for the train-vs-test overfit metric
# (evaluating the whole pack would be as slow as training; a fixed cap is enough
# to compare against the test set).
TRAIN_CE_MAX_ROWS = 100_000
# How long to run the single-position latency benchmark.
BENCH_SECONDS = 1.0

# Consecutive iterations whose holdout-CE decrease falls below
# ``expert_min_ce_decrease`` before the expert is declared stalled and skipped.
EXPERT_STALL_STEPS = 10


class CancelledError(Exception):
    """Raised inside the pipeline when the user stops the run."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise CancelledError("cancelled by user")


@dataclasses.dataclass
class TrainingConfig:
    # technique
    technique: str = "hard_moe"  # "hard_moe" | "switch" | "base"
    # base model
    base_source: str = "load"  # "load" | "new"
    base_checkpoint: str = str(DEFAULT_CHECKPOINT)
    h: int = 128
    H: int = 256
    base_epochs: int = 20
    # encoder (L1): standard accumulator vs self-supervised world model
    encoder: str = "standard"  # "standard" | "world_model"
    world_model_source: str = "train"  # "train" | "load"
    wm_checkpoint: str = ""
    wm_loss: str = "infonce"  # "infonce" | "vicreg"
    wm_tau: float = 0.1
    wm_epochs: int = 5
    wm_lr: float = 1e-3
    wm_normalize: bool = True
    wm_vicreg_gamma: float = 1.0
    # data
    max_rows: int = 2_000_000
    max_test: int = 50_000
    # clustering (hard MoE)
    clustering: str = "gradient"  # "gradient" | "l1" | "piece_count" | "kitsune"
    k: int = 8
    # KITSUNE micro-bucket resolution: B' = micro_bucket_factor * B
    micro_bucket_factor: int = 2
    # KITSUNE balanced-draft fraction: share of micro-buckets claimed by the
    # round-robin draft (even split); the rest fall back to nearest centroid.
    kitsune_alpha: float = 1.0  # 0 = pure nearest-centroid, 1 = fully balanced
    # dispatcher (hard MoE)
    dispatcher_type: str = "mlp"  # "mlp" | "linear"
    dispatcher_hidden: int = 64
    dispatcher_epochs: int = 8
    dispatcher_lr: float = 1e-2
    # Early stop once val accuracy improves by less than ``dispatcher_min_acc_gain``%
    # (relative) for ``dispatcher_patience`` epochs in a row. 0 disables it.
    dispatcher_patience: int = 5
    dispatcher_min_acc_gain: float = 1.0
    # Cap on rows used to TRAIN the dispatcher (its L1 build + SGD). 0 = no cap
    # (train on all rows). The trained router is still applied to the full
    # dataset afterward, so this only bounds the training cost.
    dispatcher_max_train: int = 1_000_000
    dispatcher_source: str = "train"  # "train" | "load"
    dispatcher_checkpoint: str = ""
    # experts (hard MoE)
    expert_epochs: int = 2
    expert_lr: float = 1e-3
    expert_lr_end: float = 1e-4
    expert_optimizer: str = "adam"  # "adam" | "sgd" | "rprop"
    l1_frozen: bool = True
    expert_min_ce_decrease: float = 0.01  # min % CE decrease per iteration; 0 = off
    # Cap on train / holdout rows per expert (0 = no cap). The frozen shared L1
    # already extracts strong features, so the shallow expert heads (L2 + head)
    # converge on a large representative subset — making the fine-tune stage
    # O(1) in the dataset size with minimal CE loss.
    expert_max_train: int = 2_000_000
    expert_max_hold: int = 100_000
    # switch (end-to-end top-1)
    switch_alpha: float = 0.01
    switch_epochs: int = 5
    switch_lr: float = 1e-3
    # misc
    device: str = "auto"
    run_name: str = "ui_run"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


EmitFn = Callable[[dict], None]


def _resolve_device(device: str) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _n_active_features() -> int:
    return piece_square_count()


def available_dataset_rows() -> int:
    """Total rows across the default slice directory (0 if unavailable).

    Used by the UI to bound the row-count spinboxes to the data that actually
    exists instead of a hardcoded cap.
    """
    try:
        folders = slice_folders(DEFAULT_SLICES)
    except Exception:
        return 0
    if not folders:
        return 0
    try:
        return total_rows(folders)
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Base model stage
# --------------------------------------------------------------------------- #
def _train_new_base(
    base: DualHiddenNNUE,
    pack,
    cfg: TrainingConfig,
    device: torch.device,
    emit: EmitFn,
    cancel: CancelToken,
    *,
    freeze_l1: bool = False,
) -> None:
    """Mini base-model training (all parameters) on the packed rows.

    When ``freeze_l1`` is set the L1 layer is frozen and only ``l2`` + ``head``
    are trained (used after contrastive world-model L1 training).
    """
    if freeze_l1:
        for param in base.l1.parameters():
            param.requires_grad = False
    n = len(pack)
    opt = torch.optim.Adam([p for p in base.parameters() if p.requires_grad], lr=1e-2)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=1e-3 / 1e-2, total_iters=max(cfg.base_epochs, 1)
    )
    n_train = max(1, int(n * 0.9))

    base.eval()
    running0 = 0.0
    count0 = 0
    with torch.inference_mode():
        for start in range(0, n_train, EXPERT_BATCH):
            cancel.check()
            end = min(start + EXPERT_BATCH, n_train)
            batch = batches_to_device(
                pack.gather(torch.arange(start, end, dtype=torch.long)), device
            )
            logits = base(
                batch["white_idx"], batch["black_idx"], batch["stm_white"],
                batch["white_mask"], batch["black_mask"],
            )
            ce, _mae, w = ce_and_mae(logits, batch["target"], batch["weight"])
            running0 += float(ce.item())
            count0 += int(w.item())
    emit(
        {
            "stage": "base",
            "event": "epoch",
            "epoch": 0,
            "epochs": cfg.base_epochs,
            "ce": running0 / max(count0, 1e-8),
            "progress": 0.0,
        }
    )

    for epoch in range(1, cfg.base_epochs + 1):
        cancel.check()
        base.train()
        running = 0.0
        count = 0
        for start in range(0, n_train, EXPERT_BATCH):
            cancel.check()
            end = min(start + EXPERT_BATCH, n_train)
            batch = batches_to_device(
                pack.gather(torch.arange(start, end, dtype=torch.long)), device
            )
            logits = base(
                batch["white_idx"], batch["black_idx"], batch["stm_white"],
                batch["white_mask"], batch["black_mask"],
            )
            ce, _mae, w = ce_and_mae(logits, batch["target"], batch["weight"])
            loss = ce / w.clamp_min(1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(ce.item())
            count += int(w.item())
        sched.step()
        emit(
            {
                "stage": "base",
                "event": "epoch",
                "epoch": epoch,
                "epochs": cfg.base_epochs,
                "ce": running / max(count, 1e-8),
                "progress": epoch / cfg.base_epochs,
            }
        )


# --------------------------------------------------------------------------- #
# Loading stage (pack + gradient cache)
# --------------------------------------------------------------------------- #
def _loading_forwarder(emit: EmitFn, cancel: CancelToken) -> Callable[[str], None]:
    """Turn ``compute_gradients`` log lines into ``loading`` progress events."""
    packed_re = re.compile(r"packed\s+([\d,]+)/([\d,]+)")
    grads_re = re.compile(r"grads\s+([\d,]+)/([\d,]+)")

    def _strip(x: str) -> int:
        return int(x.replace(",", ""))

    def forward(message: str) -> None:
        cancel.check()
        text = str(message)
        m = packed_re.search(text)
        if m:
            emit(
                {
                    "stage": "loading",
                    "event": "progress",
                    "msg": f"packing rows {m.group(1)}/{m.group(2)}",
                    "progress": _strip(m.group(1)) / max(_strip(m.group(2)), 1) * 0.6,
                }
            )
            return
        m = grads_re.search(text)
        if m:
            emit(
                {
                    "stage": "loading",
                    "event": "progress",
                    "msg": f"sample gradients {m.group(1)}/{m.group(2)}",
                    "progress": 0.6 + _strip(m.group(1)) / max(_strip(m.group(2)), 1) * 0.4,
                }
            )
            return
        emit({"stage": "loading", "event": "progress", "msg": text})

    return forward


def _build_pack(
    folders: list[Path],
    pack_dir: Path,
    cfg: TrainingConfig,
    emit: EmitFn,
    cancel: CancelToken,
) -> None:
    planned = plan_split_indices(folders, 0.01, seed=0)
    folder_list = [p[0] for p in planned]
    train_parts = subsample_parts([p[1] for p in planned], cfg.max_rows, seed=0)
    n = int(sum(int(np.asarray(p).size) for p in train_parts))

    packed_re = re.compile(r"packed\s+([\d,]+)/([\d,]+)")

    def pack_log(message: str) -> None:
        cancel.check()
        text = str(message)
        m = packed_re.search(text)
        if m:
            done = int(m.group(1).replace(",", ""))
            total = int(m.group(2).replace(",", ""))
            emit(
                {
                    "stage": "loading",
                    "event": "progress",
                    "msg": f"packing rows {m.group(1)}/{m.group(2)}",
                    "progress": done / max(total, 1),
                }
            )
        else:
            emit({"stage": "loading", "event": "progress", "msg": text, "progress": None})

    pack, slice_ids, local_rows = pack_parts(folder_list, train_parts, log=pack_log)
    save_train_pack(pack_dir, pack, slice_ids, local_rows)
    emit(
        {
            "stage": "loading",
            "event": "pack_done",
            "rows": n,
            "progress": 1.0,
            "msg": f"packed {n:,} rows",
        }
    )


# --------------------------------------------------------------------------- #
# L1 activations (dispatcher input)
# --------------------------------------------------------------------------- #
def build_l1(
    pack,
    base: DualHiddenNNUE,
    device: torch.device,
    cancel: CancelToken,
    emit: EmitFn,
    path: Path,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    if rows is None:
        n = len(pack)
    else:
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        n = int(rows.shape[0])
    hidden = int(base.hidden_dim)
    # Materialize the L1 activation matrix on disk via mmap instead of a single
    # in-RAM array: at max_rows=99M and hidden=256 this is ~189 GiB of float32,
    # which exceeds system RAM. The callers (minibatch k-means, dispatcher
    # training/prediction) already read it in chunks, so an mmap is transparent.
    out = np.memmap(str(path), dtype=np.float32, mode="w+", shape=(n, 2 * hidden))
    base.eval()
    with torch.inference_mode():
        for start in range(0, n, L1_CHUNK):
            cancel.check()
            end = min(start + L1_CHUNK, n)
            if rows is None:
                idx = torch.arange(start, end, dtype=torch.long)
            else:
                idx = torch.from_numpy(rows[start:end])
            batch = pack.gather(idx)
            h = base.l1_concat(
                batch["white_idx"].to(device),
                batch["black_idx"].to(device),
                batch["stm_white"].to(device),
                batch["white_mask"].to(device),
                batch["black_mask"].to(device),
            )
            out[start:end] = h.detach().float().cpu().numpy()
            emit(
                {
                    "stage": "loading",
                    "event": "progress",
                    "msg": f"L1 activations {end:,}/{n:,}",
                    "progress": end / n,
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Clustering stage
# --------------------------------------------------------------------------- #
def _predict_kmeans(
    km, matrix: np.ndarray, cancel: CancelToken, emit: EmitFn, *, stage: str = "cluster"
) -> np.ndarray:
    """Chunked ``km.predict`` over a (possibly disk-backed) matrix."""
    n = int(matrix.shape[0])
    labels = np.empty(n, dtype=np.int16)
    for start in range(0, n, CLUSTER_CHUNK):
        cancel.check()
        end = min(start + CLUSTER_CHUNK, n)
        labels[start:end] = km.predict(
            np.ascontiguousarray(matrix[start:end], dtype=np.float32)
        ).astype(np.int16)
        emit(
            {
                "stage": stage,
                "event": "progress",
                "progress": end / n,
                "msg": f"assigning labels {end:,}/{n:,}",
            }
        )
    return labels


def _kmeans_labels(
    matrix: np.ndarray, k: int, cancel: CancelToken, emit: EmitFn
) -> tuple[np.ndarray, np.ndarray, dict, object]:
    def progress(frac: float, msg: str) -> None:
        cancel.check()
        emit({"stage": "cluster", "event": "progress", "progress": frac, "msg": msg})

    km = fit_minibatch_kmeans(matrix, k, seed=0, max_fit_rows=5_000_000, progress=progress)
    labels = _predict_kmeans(km, matrix, cancel, emit)
    centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
    diag = cluster_diagnostics(labels, centroids, inertia=float(km.inertia_))
    return labels, centroids, diag, km


def _piece_labels(pack, cancel: CancelToken, emit: EmitFn) -> np.ndarray:
    n = len(pack)
    limit = _n_active_features()
    counts = np.empty(n, dtype=np.int16)
    for start in range(0, n, 4096):
        cancel.check()
        end = min(start + 4096, n)
        batch = pack.gather(torch.arange(start, end, dtype=torch.long))
        white_idx = batch["white_idx"].numpy()
        white_mask = batch["white_mask"].numpy()
        counts[start:end] = piece_counts_from_indices(
            white_idx, white_mask.sum(axis=1), limit=limit
        )
        emit(
            {
                "stage": "cluster",
                "event": "progress",
                "msg": f"piece counts {end:,}/{n:,}",
                "progress": end / n,
            }
        )
    return piece_count_labels(counts).astype(np.int16)


# --------------------------------------------------------------------------- #
# KITSUNE (two-stage micro-bucket → macro-expert recombination)
# --------------------------------------------------------------------------- #
def _micro_bucket_means(
    gradients,
    micro_labels: np.ndarray,
    b_prime: int,
    cancel: CancelToken,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean sample gradient per micro-bucket (chunked over the mmap'd cache)."""
    labels = np.asarray(micro_labels, dtype=np.int64)
    n = int(labels.shape[0])
    d = int(gradients.shape[1])
    sums = np.zeros((int(b_prime), d), dtype=np.float64)
    sizes = np.zeros(int(b_prime), dtype=np.int64)
    for start in range(0, n, 1_000_000):
        cancel.check()
        end = min(start + 1_000_000, n)
        block = np.asarray(gradients[start:end], dtype=np.float32)
        lab = labels[start:end]
        np.add.at(sums, lab, block.astype(np.float64))
        sizes += np.bincount(lab, minlength=int(b_prime))
    means = np.zeros((int(b_prime), d), dtype=np.float32)
    nonempty = sizes > 0
    if nonempty.any():
        means[nonempty] = (sums[nonempty] / sizes[nonempty, None]).astype(np.float32)
    return means, sizes


def _kitsune_macro_cluster(
    mean_grads: np.ndarray,
    bucket_sizes: np.ndarray,
    b: int,
    alpha: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Balanced-draft recombination of ``B'`` gradient prototypes into ``B`` experts.

    Stage A — centroids: spherical k-means on the unit-norm prototype vectors
    yields ``B`` centroids (the "default" nearest-centroid partition).

    Stage B — balanced draft: the centroids are iterated round-robin and each
    expert, in turn, claims the closest *unclaimed* prototype. After
    ``round(alpha * N)`` claims the remaining prototypes are assigned to their
    nearest centroid (default assignment). ``alpha`` therefore trades balance
    against centroid purity: ``alpha=1`` splits the micro-buckets evenly (every
    expert gets ``floor(N / B)`` or ``ceil(N / B)``), while ``alpha=0`` is a pure
    nearest-centroid partition.

    Returns a ``MAP[k'] → b`` table over all ``B'`` micro-buckets (empty buckets
    are left unassigned and re-keyed deterministically afterwards).
    """
    from sklearn.cluster import KMeans

    alpha = float(np.clip(alpha, 0.0, 1.0))
    b_prime = int(mean_grads.shape[0])
    nonempty = np.flatnonzero(np.asarray(bucket_sizes) > 0).astype(np.int64)
    n_proto = int(nonempty.size)
    if n_proto < 2:
        raise RuntimeError(f"KITSUNE needs >= 2 non-empty micro-buckets, got {n_proto}")
    n_groups = max(2, min(int(b), n_proto))

    prototypes = np.ascontiguousarray(mean_grads[nonempty], dtype=np.float32)
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    unit = prototypes / np.maximum(norms, 1e-8)

    # Stage A: centroids via spherical k-means (Euclidean on the sphere == cosine).
    km = KMeans(n_clusters=n_groups, n_init=10, random_state=0)
    km.fit(unit)
    centroids = km.cluster_centers_.astype(np.float32)
    cnorms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_unit = centroids / np.maximum(cnorms, 1e-8)

    # cosine similarity of each prototype to each centroid: sim[i, e] = <unit_i, c_e>
    sim = unit @ centroids_unit.T  # (n_proto, n_groups)

    # Stage B: round-robin balanced draft over round(alpha * N) prototypes.
    n_claim = max(0, min(n_proto, int(round(alpha * n_proto))))
    assign = np.full(n_proto, -1, dtype=np.int64)
    claimed = np.zeros(n_proto, dtype=bool)
    scores = np.empty(n_proto, dtype=np.float32)
    for t in range(n_claim):
        e = t % n_groups
        # closest unclaimed prototype to centroid e (ties broken by lowest index)
        np.copyto(scores, sim[:, e])
        scores[claimed] = -np.inf
        i = int(np.argmax(scores))
        claimed[i] = True
        assign[i] = e

    # Default assignment: nearest centroid for everything the draft did not claim.
    unclaimed = ~claimed
    assign[unclaimed] = sim[unclaimed].argmax(axis=1)

    # Guarantee every expert owns at least one prototype so all B experts survive
    # (relevant for small ``alpha``): steal the closest prototype from a donor.
    counts = np.bincount(assign, minlength=n_groups)
    while int(counts.min()) == 0:
        e = int(np.argmin(counts))
        i = int(np.argmax(sim[:, e]))
        assign[i] = e
        counts = np.bincount(assign, minlength=n_groups)

    map_table = np.zeros(b_prime, dtype=np.int64)
    map_table[nonempty] = assign
    # Re-key the macro labels so expert 0 is the largest (deterministic).
    order = np.argsort(-counts, kind="stable")
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size, dtype=np.int64)
    map_table = remap[map_table].astype(np.int64)
    diag = {
        "kitsune_micro_buckets": b_prime,
        "kitsune_macro_buckets": int(n_groups),
        "kitsune_n_nonempty": n_proto,
        "kitsune_algorithm": "balanced_draft",
        "kitsune_metric": "cosine",
        "kitsune_alpha": float(alpha),
        "kitsune_n_claimed": int(n_claim),
        "kitsune_micro_sizes": np.asarray(bucket_sizes).tolist(),
    }
    return map_table, diag


def _gradient_magnitude_by_partition(
    pack_dir: Path,
    labels: np.ndarray,
    n_partitions: int,
    work_dir: Path,
    emit: EmitFn,
    cancel: CancelToken,
    representation: str,
) -> dict:
    """Aggregate per-partition gradient magnitude before fine-tuning.

    Reads ``gradient_norms.npy`` (written alongside ``gradients.npy`` by
    ``compute_gradients``) and groups the L2 norms of the per-sample head
    gradients by the final expert partition. Writes a JSON summary and a PNG
    plot into ``work_dir``; returns the summary dict (empty if unavailable).
    """
    norm_path = pack_dir / "gradient_norms.npy"
    if not norm_path.is_file():
        emit(
            {
                "stage": "cluster",
                "event": "note",
                "msg": "gradient_norms.npy missing; skipping per-partition magnitude",
            }
        )
        return {}

    norms = np.load(norm_path, mmap_mode="r")
    n = int(norms.shape[0])
    labels = np.asarray(labels, dtype=np.int64)
    if int(labels.shape[0]) != n:
        emit(
            {
                "stage": "cluster",
                "event": "note",
                "msg": f"gradient norms {n} != labels {labels.shape[0]}; skipping magnitude",
            }
        )
        return {}

    n_partitions = int(n_partitions)
    sums = np.zeros(n_partitions, dtype=np.float64)
    sumsq = np.zeros(n_partitions, dtype=np.float64)
    mins = np.full(n_partitions, np.inf, dtype=np.float64)
    maxs = np.full(n_partitions, -np.inf, dtype=np.float64)
    sizes = np.zeros(n_partitions, dtype=np.int64)

    chunk = 1_000_000
    for start in range(0, n, chunk):
        cancel.check()
        end = min(start + chunk, n)
        block = np.asarray(norms[start:end], dtype=np.float64).reshape(-1)
        lab = labels[start:end]
        np.add.at(sums, lab, block)
        np.add.at(sumsq, lab, block * block)
        np.minimum.at(mins, lab, block)
        np.maximum.at(maxs, lab, block)
        sizes += np.bincount(lab, minlength=n_partitions).astype(np.int64)

    nonzero = sizes > 0
    means = np.where(nonzero, sums / np.maximum(sizes, 1), 0.0)
    var = np.where(
        nonzero,
        np.maximum(sumsq / np.maximum(sizes, 1) - means * means, 0.0),
        0.0,
    )
    stds = np.sqrt(var)
    mins = np.where(nonzero, mins, 0.0)
    maxs = np.where(nonzero, maxs, 0.0)

    order = np.argsort(means)[::-1]
    summary = {
        "representation": representation,
        "n_partitions": n_partitions,
        "n_rows": n,
        "mean": np.round(means, 6).tolist(),
        "std": np.round(stds, 6).tolist(),
        "min": np.round(mins, 6).tolist(),
        "max": np.round(maxs, 6).tolist(),
        "sizes": sizes.astype(int).tolist(),
        "order_by_magnitude": order.astype(int).tolist(),
    }
    (work_dir / "gradient_magnitude.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_path = plot_gradient_magnitude_by_bucket(
        means.tolist(),
        stds.tolist(),
        sizes.tolist(),
        work_dir / "gradient_magnitude_by_partition.png",
    )
    sorted_plot_path = plot_cluster_magnitude_sorted(
        means.tolist(),
        work_dir / "gradient_magnitude_sorted.png",
        stds=stds.tolist(),
    )
    emit(
        {
            "stage": "cluster",
            "event": "magnitude",
            "msg": f"per-partition gradient magnitude → {plot_path.name}, {sorted_plot_path.name}",
            "magnitude_mean": summary["mean"],
            "magnitude_std": summary["std"],
            "magnitude_order": summary["order_by_magnitude"],
            "sizes": summary["sizes"],
        }
    )
    return summary


# --------------------------------------------------------------------------- #
# Dispatcher stage
# --------------------------------------------------------------------------- #
def _dispatcher_train_rows(n: int, cap: int) -> np.ndarray:
    """Deterministic dispatcher training subset (``cap`` rows, 0 = all rows).

    Sorted indices into the pack. Keeping this subset fixed (seed 0) means the
    dispatcher's training cost is O(cap) regardless of the dataset size.
    """
    n = int(n)
    cap = int(cap)
    if cap > 0 and n > cap:
        return np.sort(
            np.random.RandomState(DISPATCHER_SUBSET_SEED)
            .choice(n, size=cap, replace=False)
            .astype(np.int64)
        )
    return np.arange(n, dtype=np.int64)


def _gather_l1_rows(
    l1: np.ndarray,
    idx: np.ndarray,
    cancel: CancelToken,
    emit: EmitFn,
    label: str,
) -> np.ndarray:
    """Copy ``l1[idx]`` into RAM in chunks, reporting progress.

    Fancy-indexing a large disk-backed matrix in one shot stalls on a single
    huge gather; this spreads the reads out and lets the UI show progress.
    """
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty((int(idx.size), int(l1.shape[1])), dtype=np.float32)
    chunk = 1_000_000
    for start in range(0, int(idx.size), chunk):
        cancel.check()
        end = min(start + chunk, int(idx.size))
        out[start:end] = l1[idx[start:end]]
        emit(
            {
                "stage": "dispatcher",
                "event": "progress",
                "progress": end / int(idx.size),
                "msg": f"gathering {label} rows {end:,}/{int(idx.size):,}",
            }
        )
    return out


def _train_dispatcher(
    l1: np.ndarray,
    labels: np.ndarray,
    cfg: TrainingConfig,
    device: torch.device,
    emit: EmitFn,
    cancel: CancelToken,
    n_classes: int | None = None,
    map_table: np.ndarray | None = None,
) -> tuple[object, dict]:
    n = int(l1.shape[0])
    in_dim = int(l1.shape[1])
    n_out = int(cfg.k if n_classes is None else n_classes)
    if cfg.dispatcher_type == "mlp":
        disp = MLPDispatcher(in_dim, n_out, hidden_dim=cfg.dispatcher_hidden).to(device)
    else:
        disp = LinearDispatcher(in_dim, n_out).to(device)

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.10))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    # The caller caps the training subset via ``dispatcher_max_train`` and passes
    # an already-bounded L1 + labels, so this split is over that subset only.
    X_train = torch.from_numpy(_gather_l1_rows(l1, train_idx, cancel, emit, "training")).to(device)
    X_val = torch.from_numpy(_gather_l1_rows(l1, val_idx, cancel, emit, "validation")).to(device)
    y_train = torch.from_numpy(labels[train_idx].astype(np.int64)).to(device)
    y_val = torch.from_numpy(labels[val_idx].astype(np.int64)).to(device)

    opt = torch.optim.Adam(disp.parameters(), lr=cfg.dispatcher_lr)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1.0,
        end_factor=1e-3 / cfg.dispatcher_lr,
        total_iters=max(cfg.dispatcher_epochs, 1),
    )
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in disp.state_dict().items()}
    n_train = int(train_idx.shape[0])
    n_val_v = int(val_idx.shape[0])
    min_acc_gain = cfg.dispatcher_min_acc_gain / 100.0  # percent -> fraction
    stall_steps = 0
    stopped_early = False

    disp.eval()
    train_correct0 = 0
    with torch.no_grad():
        for start in range(0, n_train, DISPATCHER_BATCH):
            end = min(start + DISPATCHER_BATCH, n_train)
            x = X_train[start:end]
            y = y_train[start:end]
            train_correct0 += int((disp(x).argmax(dim=-1) == y).sum().item())
    val_correct0 = 0
    with torch.no_grad():
        for start in range(0, n_val_v, DISPATCHER_BATCH):
            end = min(start + DISPATCHER_BATCH, n_val_v)
            x = X_val[start:end]
            y = y_val[start:end]
            val_correct0 += int((disp(x).argmax(dim=-1) == y).sum().item())
    emit(
        {
            "stage": "dispatcher",
            "event": "epoch",
            "epoch": 0,
            "epochs": cfg.dispatcher_epochs,
            "train_acc": train_correct0 / max(n_train, 1),
            "val_acc": val_correct0 / max(n_val_v, 1),
            "progress": 0.0,
        }
    )

    for epoch in range(1, cfg.dispatcher_epochs + 1):
        cancel.check()
        disp.train()
        train_correct = 0
        train_loss = 0.0
        for start in range(0, n_train, DISPATCHER_BATCH):
            cancel.check()
            end = min(start + DISPATCHER_BATCH, n_train)
            x = X_train[start:end]
            y = y_train[start:end]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(disp(x), y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * int(end - start)
            train_correct += int((disp(x).argmax(dim=-1) == y).sum().item())
        sched.step()
        disp.eval()
        val_correct = 0
        with torch.no_grad():
            for start in range(0, n_val_v, DISPATCHER_BATCH):
                end = min(start + DISPATCHER_BATCH, n_val_v)
                x = X_val[start:end]
                y = y_val[start:end]
                val_correct += int((disp(x).argmax(dim=-1) == y).sum().item())
        train_acc = train_correct / max(n_train, 1)
        val_acc = val_correct / max(n_val_v, 1)
        emit(
            {
                "stage": "dispatcher",
                "event": "epoch",
                "epoch": epoch,
                "epochs": cfg.dispatcher_epochs,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_loss": train_loss / max(n_train, 1),
                "progress": epoch / cfg.dispatcher_epochs,
            }
        )
        best_before = best_acc
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in disp.state_dict().items()}
        # Early stop: halt once the relative val-accuracy gain over the best
        # model seen so far stays below ``dispatcher_min_acc_gain``% for
        # ``dispatcher_patience`` consecutive epochs.
        if cfg.dispatcher_min_acc_gain > 0:
            if best_before <= 0.0:
                gain = float("inf")
            else:
                gain = (val_acc - best_before) / best_before
            stall_steps = 0 if gain > min_acc_gain else stall_steps + 1
            if stall_steps >= cfg.dispatcher_patience:
                stopped_early = True
                break

    disp.load_state_dict(best_state)
    disp.eval()
    # Post-stitching accuracy: when the caller supplies the KITSUNE MAP table,
    # re-score the *macro* partitions (MAP[argmax micro-bucket] vs MAP[true micro
    # bucket]) on the same val split. This is the accuracy on the final B experts
    # rather than on the B' micro-buckets.
    macro_val_acc = None
    if map_table is not None:
        table = torch.as_tensor(np.asarray(map_table, dtype=np.int64), device=device)
        correct = 0
        with torch.no_grad():
            for start in range(0, n_val_v, DISPATCHER_BATCH):
                end = min(start + DISPATCHER_BATCH, n_val_v)
                micro_pred = disp.predict(X_val[start:end]).long()
                macro_pred = table[micro_pred]
                macro_true = table[y_val[start:end].long()]
                correct += int((macro_pred == macro_true).sum().item())
        macro_val_acc = correct / max(n_val_v, 1)
    if stopped_early:
        emit(
            {
                "stage": "dispatcher",
                "event": "early_stop",
                "epoch": epoch,
                "epochs": cfg.dispatcher_epochs,
                "msg": (
                    f"dispatcher stopped early — val-acc gain < "
                    f"{cfg.dispatcher_min_acc_gain:g}% for "
                    f"{cfg.dispatcher_patience} epochs"
                ),
            }
        )
    return disp, {"best_val_acc": best_acc, "stopped_early": stopped_early, "macro_val_acc": macro_val_acc}


def _predict_labels_from_pack(disp, pack, base, device: torch.device, cancel: CancelToken) -> np.ndarray:
    """Apply the trained dispatcher to every pack row (streamed, no full L1).

    Computes L1 features in ``L1_CHUNK`` chunks straight from the pack, predicts,
    and discards, so the label application is O(n) forward passes with no full
    L1 materialization.
    """
    n = int(len(pack))
    out = np.empty(n, dtype=np.int16)
    with torch.no_grad():
        for start in range(0, n, L1_CHUNK):
            cancel.check()
            end = min(start + L1_CHUNK, n)
            idx = torch.arange(start, end, dtype=torch.long)
            batch = pack.gather(idx)
            h = base.l1_concat(
                batch["white_idx"].to(device),
                batch["black_idx"].to(device),
                batch["stm_white"].to(device),
                batch["white_mask"].to(device),
                batch["black_mask"].to(device),
            )
            out[start:end] = disp.predict(h).cpu().numpy().astype(np.int16)
    return out


# --------------------------------------------------------------------------- #
# Expert stage
# --------------------------------------------------------------------------- #
def _snapshot_params(model) -> dict:
    """Detached on-device copy of every parameter (for weight-update measurement)."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _weight_update_norm(model, before: dict) -> float:
    """L2 norm of the weight-space change since ``before`` (single device sync)."""
    total = torch.zeros((), device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        total = total + (p.detach() - before[name]).float().pow(2).sum()
    return float(math.sqrt(total.item()))


def _fine_tune_experts(
    base: DualHiddenNNUE,
    labels: np.ndarray,
    pack,
    cfg: TrainingConfig,
    device: torch.device,
    emit: EmitFn,
    cancel: CancelToken,
    dispatcher: LinearDispatcher | MLPDispatcher | None = None,
) -> DualHiddenMoE:
    in_dim = base.hidden_dim * 2
    if dispatcher is None:
        if cfg.dispatcher_type == "mlp":
            dispatcher = MLPDispatcher(in_dim, cfg.k, hidden_dim=cfg.dispatcher_hidden)
        else:
            dispatcher = LinearDispatcher(in_dim, cfg.k)
    moe = DualHiddenMoE.from_base(base, cfg.k, dispatcher=dispatcher).to(device)
    if cfg.l1_frozen:
        moe.freeze_l1()

    rng = np.random.RandomState(0)
    for expert_id in range(cfg.k):
        cancel.check()
        idx_all = np.flatnonzero(labels.astype(np.int64) == expert_id)
        if idx_all.size < 4:
            emit(
                {
                    "stage": "expert",
                    "expert_id": expert_id,
                    "k": cfg.k,
                    "event": "skip",
                    "msg": f"expert {expert_id}: too few rows ({idx_all.size})",
                }
            )
            continue
        perm = rng.permutation(idx_all.size)
        n_hold = max(1, int(idx_all.size * 0.10))
        if cfg.expert_max_hold > 0:
            n_hold = min(n_hold, int(cfg.expert_max_hold))
        hold_pos = np.sort(idx_all[perm[:n_hold]])
        train_pos = np.sort(idx_all[perm[n_hold:]])
        if cfg.expert_max_train > 0 and int(train_pos.size) > int(cfg.expert_max_train):
            train_pos = train_pos[: int(cfg.expert_max_train)]

        expert = DualHiddenNNUE(
            hidden_dim=base.hidden_dim, hidden2_dim=base.hidden2_dim, crelu_clip=base.crelu_clip
        )
        expert.load_state_dict(base.state_dict())
        expert.to(device)
        if cfg.l1_frozen:
            for param in expert.l1.parameters():
                param.requires_grad = False
        params = [p for p in expert.parameters() if p.requires_grad]
        if cfg.expert_optimizer == "sgd":
            opt = torch.optim.SGD(params, lr=cfg.expert_lr)
        elif cfg.expert_optimizer == "rprop":
            opt = Rprop(params, lr=cfg.expert_lr)
        else:
            opt = torch.optim.Adam(params, lr=cfg.expert_lr)
        sched = None
        if cfg.expert_optimizer != "rprop" and cfg.expert_lr_end is not None and cfg.expert_epochs > 1:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.expert_epochs, eta_min=cfg.expert_lr_end
            )
        # Rprop needs the noiseless full-dataset gradient (sign-based updates
        # are unstable on mini-batch noise), so it accumulates gradients over the
        # whole bucket and steps once per epoch. Adam/SGD keep per-batch steps.
        full_batch = cfg.expert_optimizer == "rprop"

        def _batches(positions: np.ndarray):
            for start in range(0, int(positions.size), EXPERT_BATCH):
                idx = positions[start : start + EXPERT_BATCH]
                yield batches_to_device(pack.gather(torch.from_numpy(idx)), device)

        @torch.inference_mode()
        def _eval(positions: np.ndarray) -> float:
            expert.eval()
            ce_sum = torch.zeros((), device=device)
            w_sum = torch.zeros((), device=device)
            for batch in _batches(positions):
                logits = expert(
                    batch["white_idx"], batch["black_idx"], batch["stm_white"],
                    batch["white_mask"], batch["black_mask"],
                )
                ce, _mae, w = ce_and_mae(logits, batch["target"], batch["weight"])
                ce_sum += ce
                w_sum += w
            return float((ce_sum / w_sum.clamp_min(1e-8)).item())

        base_hold = _eval(hold_pos)
        best_ce = base_hold
        best_state = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}
        prev_hold = base_hold
        stall_steps = 0
        stopped_early = False
        last_delta = 0.0
        base_params = _snapshot_params(expert)
        min_ce_decrease = cfg.expert_min_ce_decrease / 100.0  # percent -> fraction
        for epoch in range(1, cfg.expert_epochs + 1):
            cancel.check()
            expert.train()
            if full_batch:
                # snapshot weights before the step for loss-based backtracking
                prev_state = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}
            else:
                # snapshot once so ``last_delta`` reflects the whole-epoch update
                # (cloning + norming every batch was pure overhead).
                epoch_before = _snapshot_params(expert)
            opt.zero_grad(set_to_none=True)
            for batch in _batches(train_pos):
                logits = expert(
                    batch["white_idx"], batch["black_idx"], batch["stm_white"],
                    batch["white_mask"], batch["black_mask"],
                )
                ce, _mae, w = ce_and_mae(logits, batch["target"], batch["weight"])
                loss = ce / w.clamp_min(1e-8)
                loss.backward()
                if not full_batch:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            if full_batch:
                before = _snapshot_params(expert)
                opt.step()
                last_delta = _weight_update_norm(expert, before)
            else:
                last_delta = _weight_update_norm(expert, epoch_before)
            if sched is not None:
                sched.step()
            hold_ce = _eval(hold_pos)
            if full_batch and hold_ce > prev_hold:
                # Loss went up on the noiseless full-batch step: backtrack to
                # the previous weights and reduce the step sizes so it can only
                # go back down (guaranteed non-increasing holdout CE).
                expert.load_state_dict(prev_state)
                opt.shrink_step_sizes()
                opt.reset_tracking()
                hold_ce = prev_hold
            # Early stop: stop once the holdout CE has decreased by less than
            # ``expert_min_ce_decrease`` (relative) for 10 iterations in a row.
            ce_decrease = prev_hold - hold_ce
            if cfg.expert_min_ce_decrease > 0:
                threshold = min_ce_decrease * max(prev_hold, 1e-12)
                stall_steps = stall_steps + 1 if ce_decrease < threshold else 0
                if stall_steps >= EXPERT_STALL_STEPS:
                    stopped_early = True
            prev_hold = hold_ce
            delta_base = _weight_update_norm(expert, base_params)
            emit(
                {
                    "stage": "expert",
                    "expert_id": expert_id,
                    "k": cfg.k,
                    "event": "epoch",
                    "epoch": epoch,
                    "epochs": cfg.expert_epochs,
                    "hold_ce": hold_ce,
                    "base_hold_ce": base_hold,
                    "n_train": int(train_pos.size),
                    "weight_delta": last_delta,
                    "weight_delta_base": delta_base,
                    "progress": epoch / cfg.expert_epochs,
                }
            )
            if hold_ce < best_ce:
                best_ce = hold_ce
                best_state = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}
            if stopped_early:
                break

        expert.load_state_dict(best_state)
        moe.experts_l2[expert_id].load_state_dict(expert.l2.state_dict())
        moe.experts_head[expert_id].load_state_dict(expert.head.state_dict())
        if stopped_early:
            emit(
                {
                    "stage": "expert",
                    "expert_id": expert_id,
                    "k": cfg.k,
                    "event": "early_stop",
                    "epoch": epoch,
                    "epochs": cfg.expert_epochs,
                    "msg": (
                        f"expert {expert_id}: stopped early — CE decrease < "
                        f"{cfg.expert_min_ce_decrease:g}% for {EXPERT_STALL_STEPS} iterations"
                    ),
                }
            )
        emit(
            {
                "stage": "expert",
                "expert_id": expert_id,
                "k": cfg.k,
                "event": "done",
                "hold_ce": best_ce if best_ce < float("inf") else base_hold,
                "base_hold_ce": base_hold,
                "n_train": int(train_pos.size),
                "progress": 1.0,
                "stopped_early": stopped_early,
            }
        )

    return moe


# --------------------------------------------------------------------------- #
# Evaluation stage
# --------------------------------------------------------------------------- #
def _piece_ids(batch: dict, limit: int) -> np.ndarray:
    white_idx = batch["white_idx"].cpu().numpy()
    white_mask = batch["white_mask"].cpu().numpy()
    counts = piece_counts_from_indices(white_idx, white_mask.sum(axis=1), limit=limit)
    return piece_count_labels(counts).astype(np.int64)


@torch.inference_mode()
def _evaluate_base(
    base: DualHiddenNNUE,
    folders: list[Path],
    cfg: TrainingConfig,
    device: torch.device,
    emit: EmitFn,
    cancel: CancelToken,
) -> dict:
    """Test-set CE/MAE for the base model alone (K=1, no MoE)."""
    planned = plan_split_indices(folders, 0.01, seed=0)
    test_parts = subsample_parts([p[2] for p in planned], cfg.max_test, seed=1)
    base_ce = base_mae = 0.0
    w_sum = 0.0
    n_rows = 0
    total = int(sum(int(np.asarray(p).size) for p in test_parts))
    base.eval()
    for (folder, _tr, _te), rows in zip(planned, test_parts):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            continue
        ds = FenValueVisitsDataset(folder, progress=False)
        for start in range(0, int(rows.size), EVAL_BATCH):
            cancel.check()
            idx = rows[start : start + EVAL_BATCH]
            batch = batches_to_device(ds.gather(idx), device)
            logits = base(
                batch["white_idx"], batch["black_idx"], batch["stm_white"],
                batch["white_mask"], batch["black_mask"],
            )
            bce, bmae, w = ce_and_mae(logits, batch["target"].float(), batch["weight"])
            base_ce += float(bce.item())
            base_mae += float(bmae.item())
            w_sum += float(w.item())
            n_rows += int(idx.size)
            emit(
                {
                    "stage": "eval",
                    "event": "progress",
                    "msg": f"evaluating {n_rows:,}/{total:,}",
                    "progress": n_rows / max(total, 1),
                }
            )
        del ds
    denom = max(w_sum, 1e-8)
    return {
        "n_test": n_rows,
        "base_ce": base_ce / denom,
        "base_mae": base_mae / denom,
    }


@torch.inference_mode()
def _evaluate(
    base: DualHiddenNNUE,
    moe: DualHiddenMoE,
    folders: list[Path],
    cfg: TrainingConfig,
    device: torch.device,
    emit: EmitFn,
    cancel: CancelToken,
    centroids: np.ndarray | None = None,
    map_table: np.ndarray | None = None,
) -> dict:
    planned = plan_split_indices(folders, 0.01, seed=0)
    test_parts = subsample_parts([p[2] for p in planned], cfg.max_test, seed=1)
    limit = _n_active_features()
    projection = None
    if cfg.clustering == "gradient" and centroids is not None:
        projection = make_projection_matrix(
            head_parameter_dim(base.hidden_dim, base.hidden2_dim),
            DEFAULT_REDUCE_DIM,
            seed=0,
            device=device,
            dtype=torch.float32,
        )
    base_ce = base_mae = moe_ce = moe_mae = 0.0
    perfect_ce = perfect_mae = best_expert_ce = best_expert_mae = 0.0
    w_sum = 0.0
    n_rows = 0
    total = int(sum(int(np.asarray(p).size) for p in test_parts))
    base.eval()
    moe.eval()
    for (folder, _tr, _te), rows in zip(planned, test_parts):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            continue
        ds = FenValueVisitsDataset(folder, progress=False)
        for start in range(0, int(rows.size), EVAL_BATCH):
            cancel.check()
            idx = rows[start : start + EVAL_BATCH]
            batch = batches_to_device(ds.gather(idx), device)
            target = batch["target"].float()
            weight = batch["weight"]
            base_logits = base(
                batch["white_idx"], batch["black_idx"], batch["stm_white"],
                batch["white_mask"], batch["black_mask"],
            )
            if cfg.clustering == "piece_count":
                expert_ids = torch.from_numpy(_piece_ids(batch, limit)).to(device)
                moe_logits = moe(
                    batch["white_idx"], batch["black_idx"], batch["stm_white"],
                    batch["white_mask"], batch["black_mask"], expert_ids=expert_ids,
                )
            else:
                moe_logits = moe(
                    batch["white_idx"], batch["black_idx"], batch["stm_white"],
                    batch["white_mask"], batch["black_mask"],
                )
            h = base.l1_concat(
                batch["white_idx"], batch["black_idx"], batch["stm_white"],
                batch["white_mask"], batch["black_mask"],
            )

            # Perfect dispatcher: route each row to its true cluster.
            if cfg.clustering == "l1" and centroids is not None:
                perfect_ids = assign_to_centroids(h.detach().cpu().numpy(), centroids)
                perfect_logits, _ = moe.route_from_h(
                    h, expert_ids=torch.from_numpy(perfect_ids.astype(np.int64)).to(device)
                )
            elif cfg.clustering == "kitsune" and centroids is not None and map_table is not None:
                micro_ids = assign_to_centroids(h.detach().cpu().numpy(), centroids)
                perfect_ids = np.asarray(map_table, dtype=np.int64)[micro_ids.astype(np.int64)]
                perfect_logits, _ = moe.route_from_h(
                    h, expert_ids=torch.from_numpy(perfect_ids.astype(np.int64)).to(device)
                )
            elif cfg.clustering == "gradient" and projection is not None and centroids is not None:
                grads = sample_head_gradients(base, batch, projection)
                perfect_ids = assign_to_centroids(grads.detach().cpu().numpy(), centroids)
                perfect_logits, _ = moe.route_from_h(
                    h, expert_ids=torch.from_numpy(perfect_ids.astype(np.int64)).to(device)
                )
            else:
                perfect_logits = moe_logits

            # Best expert: per-row argmin CE over all experts (upper bound).
            stacked = moe.all_expert_logits_from_h(h)
            log_p = F.log_softmax(stacked.float(), dim=-1)
            nll_e = -(target[:, None, :] * log_p).sum(dim=-1)
            best_expert_ids = nll_e.argmin(dim=-1)
            rows_t = torch.arange(int(idx.size), device=device)
            best_expert_logits = stacked[rows_t, best_expert_ids, :]

            bce, bmae, w = ce_and_mae(base_logits, target, weight)
            mce, mmae, _ = ce_and_mae(moe_logits, target, weight)
            pce, pmae, _ = ce_and_mae(perfect_logits, target, weight)
            ece, emae, _ = ce_and_mae(best_expert_logits, target, weight)
            base_ce += float(bce.item())
            base_mae += float(bmae.item())
            moe_ce += float(mce.item())
            moe_mae += float(mmae.item())
            perfect_ce += float(pce.item())
            perfect_mae += float(pmae.item())
            best_expert_ce += float(ece.item())
            best_expert_mae += float(emae.item())
            w_sum += float(w.item())
            n_rows += int(idx.size)
            emit(
                {
                    "stage": "eval",
                    "event": "progress",
                    "msg": f"evaluating {n_rows:,}/{total:,}",
                    "progress": n_rows / max(total, 1),
                }
            )
        del ds
    denom = max(w_sum, 1e-8)
    return {
        "n_test": n_rows,
        "base_ce": base_ce / denom,
        "base_mae": base_mae / denom,
        "moe_ce": moe_ce / denom,
        "moe_mae": moe_mae / denom,
        "perfect_ce": perfect_ce / denom,
        "perfect_mae": perfect_mae / denom,
        "best_expert_ce": best_expert_ce / denom,
        "best_expert_mae": best_expert_mae / denom,
    }


# --------------------------------------------------------------------------- #
# Switch (end-to-end sparse top-1 MoE)
# --------------------------------------------------------------------------- #
def _load_balance_loss(gate_probs: torch.Tensor, topk_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    b = max(int(gate_probs.shape[0]), 1)
    f = torch.zeros(n_experts, device=gate_probs.device, dtype=gate_probs.dtype)
    idx = topk_idx.reshape(-1)
    f = f.scatter_add(0, idx, torch.ones_like(idx, dtype=gate_probs.dtype)) / b
    p = gate_probs.mean(dim=0)
    return float(n_experts) * (f * p).sum()


@torch.inference_mode()
def _switch_eval(
    base: DualHiddenNNUE,
    moe: SoftGatedMoE,
    folders: list[Path],
    cfg: TrainingConfig,
    device: torch.device,
    cancel: CancelToken,
    *,
    full: bool = True,
    emit: EmitFn | None = None,
) -> dict:
    planned = plan_split_indices(folders, 0.01, seed=0)
    test_parts = subsample_parts([p[2] for p in planned], cfg.max_test, seed=1)
    base_ce = base_mae = moe_ce = moe_mae = 0.0
    best_expert_ce = best_expert_mae = 0.0
    entropy = 0.0
    load = np.zeros(moe.n_experts, dtype=np.float64)
    w_sum = 0.0
    n_rows = 0
    total = int(sum(int(np.asarray(p).size) for p in test_parts))
    base.eval()
    moe.eval()
    for (folder, _tr, _te), rows in zip(planned, test_parts):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            continue
        ds = FenValueVisitsDataset(folder, progress=False)
        for start in range(0, int(rows.size), EVAL_BATCH):
            cancel.check()
            idx = rows[start : start + EVAL_BATCH]
            batch = batches_to_device(ds.gather(idx), device)
            target = batch["target"].float()
            weight = batch["weight"]
            moe_logits, gate_probs, topk_idx = moe(
                batch["white_idx"], batch["black_idx"], batch["stm_white"],
                batch["white_mask"], batch["black_mask"],
            )
            mce, mmae, w = ce_and_mae(moe_logits, target, weight)
            moe_ce += float(mce.item())
            moe_mae += float(mmae.item())
            w_sum += float(w.item())
            n_rows += int(idx.size)
            ent = -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=-1).mean()
            entropy += float(ent.item()) * int(idx.size)
            for e in topk_idx.reshape(-1).cpu().numpy():
                load[int(e)] += 1.0
            if full:
                base_logits = base(
                    batch["white_idx"], batch["black_idx"], batch["stm_white"],
                    batch["white_mask"], batch["black_mask"],
                )
                bce, bmae, _ = ce_and_mae(base_logits, target, weight)
                base_ce += float(bce.item())
                base_mae += float(bmae.item())
                h = moe.l1_concat(
                    batch["white_idx"], batch["black_idx"], batch["stm_white"],
                    batch["white_mask"], batch["black_mask"],
                )
                stacked = moe.all_expert_logits_from_h(h)
                log_p = F.log_softmax(stacked.float(), dim=-1)
                nll_e = -(target[:, None, :] * log_p).sum(dim=-1)
                best_ids = nll_e.argmin(dim=-1)
                rows_t = torch.arange(int(idx.size), device=device)
                best_logits = stacked[rows_t, best_ids, :]
                ece, emae, _ = ce_and_mae(best_logits, target, weight)
                best_expert_ce += float(ece.item())
                best_expert_mae += float(emae.item())
            if emit is not None:
                emit(
                    {
                        "stage": "eval",
                        "event": "progress",
                        "msg": f"evaluating {n_rows:,}/{total:,}",
                        "progress": n_rows / max(total, 1),
                    }
                )
        del ds
    denom = max(w_sum, 1e-8)
    load = load / max(float(load.sum()), 1.0)
    metrics: dict = {
        "n_test": n_rows,
        "moe_ce": moe_ce / denom,
        "moe_mae": moe_mae / denom,
        "gate_entropy": entropy / max(n_rows, 1),
        "load_distribution": [round(float(x), 5) for x in load],
        "load_variance": float(np.var(load)),
        "load_max_min_ratio": float(load.max() / max(load.min(), 1e-9)),
    }
    if full:
        metrics["base_ce"] = base_ce / denom
        metrics["base_mae"] = base_mae / denom
        metrics["best_expert_ce"] = best_expert_ce / denom
        metrics["best_expert_mae"] = best_expert_mae / denom
    return metrics


def _train_switch(
    base: DualHiddenNNUE,
    pack,
    folders: list[Path],
    cfg: TrainingConfig,
    device: torch.device,
    emit: EmitFn,
    cancel: CancelToken,
) -> tuple[SoftGatedMoE, float]:
    n = len(pack)
    moe = SoftGatedMoE(base, n_experts=cfg.k, top_k=1).to(device)
    moe.freeze_l1()
    opt = torch.optim.Adam([p for p in moe.parameters() if p.requires_grad], lr=cfg.switch_lr)
    rng = np.random.RandomState(0)
    best_test = float("inf")
    best_state = None
    for epoch in range(1, cfg.switch_epochs + 1):
        cancel.check()
        moe.train()
        perm = rng.permutation(n).astype(np.int64)
        ce_acc = 0.0
        lb_acc = 0.0
        n_batches = 0
        for start in range(0, n, SWITCH_BATCH):
            cancel.check()
            idx = perm[start : start + SWITCH_BATCH]
            batch = batches_to_device(pack.gather(torch.from_numpy(idx)), device)
            opt.zero_grad(set_to_none=True)
            logits, gate_probs, topk_idx = moe(
                batch["white_idx"], batch["black_idx"], batch["stm_white"],
                batch["white_mask"], batch["black_mask"],
            )
            ce, _mae, w = ce_and_mae(logits, batch["target"], batch["weight"])
            ce_loss = ce / w.clamp_min(1e-8)
            lb = _load_balance_loss(gate_probs, topk_idx, moe.n_experts)
            (ce_loss + cfg.switch_alpha * lb).backward()
            opt.step()
            ce_acc += float(ce_loss.item())
            lb_acc += float(lb.item())
            n_batches += 1
        test = _switch_eval(base, moe, folders, cfg, device, cancel, full=False)
        emit(
            {
                "stage": "switch",
                "event": "epoch",
                "epoch": epoch,
                "epochs": cfg.switch_epochs,
                "train_ce": ce_acc / max(n_batches, 1),
                "lb": lb_acc / max(n_batches, 1),
                "test_ce": test["moe_ce"],
                "gate_entropy": test.get("gate_entropy"),
                "progress": epoch / cfg.switch_epochs,
            }
        )
        if test["moe_ce"] < best_test:
            best_test = test["moe_ce"]
            best_state = {kk: v.detach().cpu().clone() for kk, v in moe.state_dict().items()}
    if best_state is not None:
        moe.load_state_dict(best_state)
    return moe, best_test


def _train_ce_on_pack(
    model,
    pack,
    device: torch.device,
    cancel: CancelToken,
    *,
    max_rows: int = TRAIN_CE_MAX_ROWS,
) -> float | None:
    """Weighted CE of ``model`` over the first ``max_rows`` rows of the pack.

    Used for the train-vs-test overfit metric. Returns ``None`` when the pack is
    missing (e.g. a base-only run that loaded its checkpoint).
    """
    if pack is None:
        return None
    n = min(int(len(pack)), int(max_rows))
    if n <= 0:
        return None
    model.eval()
    ce_sum = 0.0
    w_sum = 0.0
    with torch.inference_mode():
        for start in range(0, n, EVAL_BATCH):
            cancel.check()
            idx = torch.arange(start, min(start + EVAL_BATCH, n), dtype=torch.long)
            batch = batches_to_device(pack.gather(idx), device)
            out = model(
                batch["white_idx"], batch["black_idx"], batch["stm_white"],
                batch["white_mask"], batch["black_mask"],
            )
            logits = out[0] if isinstance(out, tuple) else out
            ce, _mae, w = ce_and_mae(logits, batch["target"].float(), batch["weight"])
            ce_sum += float(ce.item())
            w_sum += float(w.item())
    return ce_sum / max(w_sum, 1e-8)


def _benchmark_call_time(model, device: torch.device, *, n_seconds: float = BENCH_SECONDS) -> tuple[float, float]:
    """Avg seconds per single-position call and calls/sec over ``n_seconds``.

    Calls the model on one synthetic position (32 active features per side) as
    many times as possible within the time budget, mirroring a realistic
    per-position inference cost.
    """
    model.eval()
    active = 32
    idx = torch.arange(active, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones((1, active), dtype=torch.bool, device=device)
    stm = torch.ones(1, dtype=torch.bool, device=device)

    def _call() -> None:
        model(idx, idx, stm, mask, mask)

    with torch.inference_mode():
        for _ in range(10):
            _call()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        n = 0
        while True:
            _call()
            n += 1
            if time.perf_counter() - start >= n_seconds:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    avg_s = elapsed / max(n, 1)
    return avg_s, n / max(elapsed, 1e-9)


def _measure_overfit_and_latency(
    base: DualHiddenNNUE,
    moe,
    pack,
    test_metrics: dict,
    work_dir: Path,
    device: torch.device,
    cancel: CancelToken,
) -> dict:
    """Compute train CE + single-position latency and write the summary plot.

    Returns a dict of extra metrics to merge into ``eval.json`` / ``summary.json``.
    """
    extra: dict = {}

    base_train_ce = _train_ce_on_pack(base, pack, device, cancel)
    extra["base_train_ce"] = base_train_ce
    base_avg, base_cps = _benchmark_call_time(base, device)
    extra["base_call_time_s"] = base_avg
    extra["base_calls_per_sec"] = base_cps

    moe_train_ce = None
    moe_avg = moe_cps = None
    if moe is not None:
        moe_train_ce = _train_ce_on_pack(moe, pack, device, cancel)
        moe_avg, moe_cps = _benchmark_call_time(moe, device)
    extra["moe_train_ce"] = moe_train_ce
    extra["moe_call_time_s"] = moe_avg
    extra["moe_calls_per_sec"] = moe_cps

    try:
        base_info = {
            "train_ce": base_train_ce,
            "test_ce": test_metrics.get("base_ce"),
            "call_ms": base_avg * 1000.0 if base_avg is not None else None,
        }
        moe_info = None
        if moe is not None:
            moe_info = {
                "train_ce": moe_train_ce,
                "test_ce": test_metrics.get("moe_ce"),
                "call_ms": moe_avg * 1000.0 if moe_avg is not None else None,
            }
        plot_overfit_latency(
            work_dir / "overfit_latency.png", base=base_info, moe=moe_info
        )
    except Exception as exc:  # noqa: BLE001 — plotting must not fail the run
        print(f"[overfit_latency] plot skipped: {exc}", file=sys.stderr)

    return extra


def _finalize_run(
    work_dir: Path,
    pack_dir: Path,
    *,
    base: DualHiddenNNUE | None = None,
) -> None:
    """Strip a completed run folder down to weights, metadata, and plots.

    Deletes the cached data pack (``train_pack.pt``, ``gradients.npy``,
    ``gradient_norms.npy``, ``slice_ids.npy``, ``local_rows.npy``) and any
    per-run intermediate arrays (labels, centroids, map tables, pickled
    quantizers, L1 activations). Leaves only the trained weights (``moe.pt`` /
    ``base.pt``), JSON metadata, and plot images in ``work_dir``.
    """
    for name in (
        "labels.npy",
        "centroids.npy",
        "labels_dispatcher.npy",
        "map_table.npy",
        "labels_micro.npy",
        "micro_kmeans.pkl",
        "l1_activations.npy",
    ):
        p = work_dir / name
        if p.is_file():
            p.unlink(missing_ok=True)
    if base is not None:
        base.save(work_dir / "base.pt")
    if pack_dir.is_dir():
        shutil.rmtree(pack_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Top-level pipeline
# --------------------------------------------------------------------------- #
class _StageTimer:
    """Wall-clock timing of each pipeline stage, keyed by stage name.

    ``mark(name)`` records the seconds elapsed since the previous mark (or since
    construction) under ``name``. ``as_dict()`` returns the per-stage times plus
    a ``total`` wall-clock entry, all rounded to milliseconds.
    """

    def __init__(self) -> None:
        self._begin = time.perf_counter()
        self._stage_start = self._begin
        self.times: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.times[name] = now - self._stage_start
        self._stage_start = now

    def as_dict(self) -> dict[str, float]:
        return {
            **{k: round(v, 3) for k, v in self.times.items()},
            "total": round(time.perf_counter() - self._begin, 3),
        }


def run_moe_training(cfg: TrainingConfig, emit: EmitFn, cancel: CancelToken) -> dict:
    device = _resolve_device(cfg.device)
    configure_torch(device)
    emit({"stage": "init", "event": "start", "config": cfg.to_dict(), "device": str(device)})
    timer = _StageTimer()

    slices_dir = DEFAULT_SLICES
    folders = slice_folders(slices_dir)
    if not folders:
        raise RuntimeError(f"no slice folders in {slices_dir}")

    # Bound the row budgets to the data that actually exists. Train and test
    # come from a fixed 99/1 split of every slice, so each budget is clamped to
    # its own pool; this guarantees train + test never overshoot the dataset.
    try:
        n_available = total_rows(folders)
    except Exception:
        n_available = 0
    if n_available > 0:
        test_pool = max(1, int(n_available * 0.01))
        train_pool = n_available - test_pool
        if cfg.max_rows > train_pool:
            emit(
                {
                    "stage": "init",
                    "event": "note",
                    "msg": f"train rows {cfg.max_rows:,} exceed the {train_pool:,}-row "
                    f"train split; clamping to {train_pool:,}",
                }
            )
            cfg.max_rows = train_pool
        if cfg.max_test > test_pool:
            emit(
                {
                    "stage": "init",
                    "event": "note",
                    "msg": f"test rows {cfg.max_test:,} exceed the {test_pool:,}-row "
                    f"test split; clamping to {test_pool:,}",
                }
            )
            cfg.max_test = test_pool

    # -- resolve the base model up front so the gradient-cache directory can be
    # keyed by the actual architecture + row count. Otherwise a new arch / row
    # count would reuse (and clash with) a stale cache from a previous run.
    use_world_model = cfg.encoder == "world_model"
    if use_world_model and cfg.world_model_source == "load":
        if not cfg.wm_checkpoint:
            raise ValueError(
                "world model source is 'load' but no checkpoint was selected "
                "(the checkpoint list is empty)"
            )
        wm_ckpt = Path(cfg.wm_checkpoint)
        if not wm_ckpt.is_file():
            raise FileNotFoundError(f"world model checkpoint not found: {wm_ckpt}")
        base = load_dual_hidden_checkpoint(wm_ckpt, device=device)
        if not base.is_world_model:
            raise ValueError(
                f"{wm_ckpt.name} is not a world-model checkpoint (encoder={base.encoder!r})"
            )
        h_actual, H_actual = base.hidden_dim, base.hidden2_dim
    elif cfg.base_source == "load":
        if not cfg.base_checkpoint:
            raise ValueError(
                "base model source is 'load' but no checkpoint was selected "
                "(the checkpoint list is empty)"
            )
        ckpt = Path(cfg.base_checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"base checkpoint not found: {ckpt}")
        base = load_dual_hidden_checkpoint(ckpt, device=device)
        h_actual, H_actual = base.hidden_dim, base.hidden2_dim
    else:
        base = DualHiddenNNUE(hidden_dim=cfg.h, hidden2_dim=cfg.H).to(device)
        h_actual, H_actual = cfg.h, cfg.H
    timer.mark("base_resolve")

    pack_dir = UI_PACK_ROOT / f"moe_ui_{cfg.run_name}_h{h_actual}_H{H_actual}_n{cfg.max_rows}"
    pack_dir.mkdir(parents=True, exist_ok=True)
    work_dir = MOE_RESULTS_ROOT / cfg.run_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # -- loading: pack the training rows ------------------------------------
    # A pack is only needed to (re)train the base or to run any MoE stage; a
    # base-only run that loads a checkpoint can skip it entirely.
    needs_pack = (
        cfg.base_source == "new"
        or (use_world_model and cfg.world_model_source == "train")
        or int(cfg.k) > 1
    )
    if needs_pack:
        emit({"stage": "loading", "event": "start", "msg": f"packing {cfg.max_rows:,} rows"})
        _build_pack(folders, pack_dir, cfg, emit, cancel)
    else:
        emit({"stage": "loading", "event": "skip", "msg": "base-only (no pack)"})
    timer.mark("loading")

    # -- base model ----------------------------------------------------------
    emit({"stage": "base", "event": "start"})
    if use_world_model and cfg.world_model_source == "load":
        emit(
            {
                "stage": "base",
                "event": "done",
                "source": "world_model",
                "h": base.hidden_dim,
                "H": base.hidden2_dim,
                "params": int(sum(p.numel() for p in base.parameters())),
                "msg": f"loaded world model {wm_ckpt.name} (W{base.hidden_dim}_H{base.hidden2_dim})",
            }
        )
    elif use_world_model and cfg.world_model_source == "train":
        emit(
            {
                "stage": "base",
                "event": "start_train",
                "h": h_actual,
                "H": H_actual,
                "epochs": cfg.base_epochs,
                "params": int(sum(p.numel() for p in base.parameters())),
                "world_model": True,
            }
        )
        pack, _s, _l = load_train_pack(pack_dir)
        train_world_model_encoder(base, pack, cfg, device, emit, cancel)
        _train_new_base(base, pack, cfg, device, emit, cancel, freeze_l1=True)
        del pack
        # persist the world-model base so it can be loaded again later
        wm_dir = NNUE_CHECKPOINTS_DIR / f"worldmodel_h{h_actual}_H{H_actual}"
        base.save(wm_dir / "best.pt")
        (wm_dir / "config.json").write_text(
            json.dumps(
                {
                    "hidden_dim": h_actual,
                    "hidden2_dim": H_actual,
                    "encoder": "world_model",
                    "normalize_l1": base.normalize_l1,
                    "wm_loss": cfg.wm_loss,
                    "wm_tau": cfg.wm_tau,
                    "wm_epochs": cfg.wm_epochs,
                    "wm_lr": cfg.wm_lr,
                    "base_epochs": cfg.base_epochs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        emit(
            {
                "stage": "base",
                "event": "done",
                "source": "world_model",
                "h": h_actual,
                "H": H_actual,
                "saved": str(wm_dir / "best.pt"),
                "msg": f"trained world model → {wm_dir.name} (W{h_actual}_H{H_actual})",
            }
        )
    elif cfg.base_source == "load":
        emit(
            {
                "stage": "base",
                "event": "done",
                "source": "load",
                "h": base.hidden_dim,
                "H": base.hidden2_dim,
                "params": int(sum(p.numel() for p in base.parameters())),
                "world_model": base.is_world_model,
                "msg": (
                    f"loaded {ckpt.name} (W{base.hidden_dim}_H{base.hidden2_dim}"
                    f"{', world model' if base.is_world_model else ''})"
                ),
            }
        )
    else:
        emit(
            {
                "stage": "base",
                "event": "start_train",
                "h": cfg.h,
                "H": cfg.H,
                "epochs": cfg.base_epochs,
                "params": int(sum(p.numel() for p in base.parameters())),
            }
        )
        pack, _s, _l = load_train_pack(pack_dir)
        _train_new_base(base, pack, cfg, device, emit, cancel)
        del pack
        # persist the freshly-trained base so it can be loaded again later
        ckpt_name = f"h{h_actual}_H{H_actual}_be{cfg.base_epochs}_{cfg.max_rows // 1_000_000}M"
        ckpt_dir = NNUE_CHECKPOINTS_DIR / ckpt_name
        base.save(ckpt_dir / "best.pt")
        (ckpt_dir / "config.json").write_text(
            json.dumps(
                {
                    "hidden_dim": h_actual,
                    "hidden2_dim": H_actual,
                    "encoder": "standard",
                    "normalize_l1": base.normalize_l1,
                    "base_epochs": cfg.base_epochs,
                    "max_rows": cfg.max_rows,
                    "run_name": cfg.run_name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        emit(
            {
                "stage": "base",
                "event": "done",
                "source": "new",
                "h": h_actual,
                "H": H_actual,
                "saved": str(ckpt_dir / "best.pt"),
                "msg": f"trained base → {ckpt_name} (W{h_actual}_H{H_actual})",
            }
        )

    timer.mark("base")

    # -- base-only (K=1, no MoE) --------------------------------------------
    if int(cfg.k) <= 1:
        emit({"stage": "eval", "event": "start"})
        metrics = _evaluate_base(base, folders, cfg, device, emit, cancel)
        metrics.update(
            _measure_overfit_and_latency(base, None, None, metrics, work_dir, device, cancel)
        )
        (work_dir / "eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        emit({"stage": "eval", "event": "done", **metrics})
        timer.mark("eval")
        summary = {
            "run_name": cfg.run_name,
            "technique": "base",
            "config": cfg.to_dict(),
            "metrics": metrics,
            "stage_times": timer.as_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        emit({"stage": "done", "summary": summary})
        _finalize_run(work_dir, pack_dir, base=base)
        return metrics

    # -- switch (end-to-end sparse top-1 MoE) --------------------------------
    if cfg.technique == "switch":
        pack, _s, _l = load_train_pack(pack_dir)
        pack = pack.to(device)
        emit(
            {
                "stage": "switch",
                "event": "start",
                "k": cfg.k,
                "alpha": cfg.switch_alpha,
                "epochs": cfg.switch_epochs,
            }
        )
        moe, best_test = _train_switch(base, pack, folders, cfg, device, emit, cancel)
        moe.save(work_dir / "moe.pt")
        emit({"stage": "switch", "event": "done"})
        timer.mark("switch")

        emit({"stage": "eval", "event": "start"})
        metrics = _switch_eval(base, moe, folders, cfg, device, cancel, full=True, emit=emit)
        metrics.update(
            _measure_overfit_and_latency(base, moe, pack, metrics, work_dir, device, cancel)
        )
        (work_dir / "eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        emit({"stage": "eval", "event": "done", **metrics})
        timer.mark("eval")

        summary = {
            "run_name": cfg.run_name,
            "technique": cfg.technique,
            "config": cfg.to_dict(),
            "metrics": metrics,
            "best_test_ce": best_test,
            "stage_times": timer.as_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        emit({"stage": "done", "summary": summary})
        _finalize_run(work_dir, pack_dir, base=base)
        return metrics

    # -- cluster -------------------------------------------------------------
    emit({"stage": "cluster", "event": "start", "clustering": cfg.clustering, "k": cfg.k})
    map_table: np.ndarray | None = None
    micro_labels: np.ndarray | None = None
    b_prime: int = 0
    if cfg.clustering == "gradient":
        compute_gradients(
            base, folders, pack_dir, device=device, max_rows=cfg.max_rows,
            resume=True, log=_loading_forwarder(emit, cancel),
        )
        gradients = np.load(pack_dir / "gradients.npy", mmap_mode="r")
        labels, centroids, diag, _km = _kmeans_labels(gradients, cfg.k, cancel, emit)
        diag["representation"] = "head_gradient_48d"
        del gradients
    elif cfg.clustering == "l1":
        pack, _s, _l = load_train_pack(pack_dir)
        l1 = build_l1(pack, base, device, cancel, emit, path=work_dir / "l1_activations.npy")
        labels, centroids, diag, _km = _kmeans_labels(l1, cfg.k, cancel, emit)
        diag["representation"] = "l1_256d"
        del l1
        (work_dir / "l1_activations.npy").unlink(missing_ok=True)
        del pack
    elif cfg.clustering == "kitsune":
        # Bound the clustering to a fixed representative subset so this stage is
        # O(1) in the dataset size. The subset is deterministic (seed 0) and
        # shared by the L1 build, gradient computation, and micro-labels so they
        # stay aligned with one another.
        pack, _s, _l = load_train_pack(pack_dir)
        n_pack = len(pack)
        b_prime = max(int(cfg.k) * int(cfg.micro_bucket_factor), int(cfg.k) + 1, 2)
        cluster_n = min(int(CLUSTER_MAX_ROWS), n_pack)
        cluster_idx = np.sort(
            np.random.RandomState(CLUSTER_SUBSET_SEED)
            .choice(n_pack, size=cluster_n, replace=False)
            .astype(np.int64)
        )
        # Stage 1: micro-quantize L1 into B' = factor * B micro-buckets (subset).
        l1 = build_l1(
            pack, base, device, cancel, emit,
            path=work_dir / "l1_activations.npy", rows=cluster_idx,
        )
        micro_labels, centroids, micro_diag, micro_km = _kmeans_labels(l1, b_prime, cancel, emit)
        del l1
        (work_dir / "l1_activations.npy").unlink(missing_ok=True)
        del pack
        # Persist the micro-quantizer so the dispatcher can label the full dataset.
        with open(work_dir / "micro_kmeans.pkl", "wb") as fh:
            pickle.dump(micro_km, fh)
        # Stage 2: mean sample gradient per micro-bucket (subset only).
        compute_gradients(
            base, folders, pack_dir, device=device, max_rows=cfg.max_rows,
            row_idx=cluster_idx, resume=False,
            log=_loading_forwarder(emit, cancel),
        )
        gradients = np.load(pack_dir / "gradients.npy", mmap_mode="r")
        mean_grads, bucket_sizes = _micro_bucket_means(
            gradients, micro_labels, b_prime, cancel
        )
        del gradients
        # Stage 3: agglomerative (cosine) macro-clustering of gradient prototypes.
        map_table, map_diag = _kitsune_macro_cluster(
            mean_grads, bucket_sizes, cfg.k, alpha=cfg.kitsune_alpha
        )
        labels = map_table[micro_labels.astype(np.int64)].astype(np.int16)
        diag = {
            "representation": "kitsune_l1_gradient",
            "sizes": np.bincount(labels.astype(np.int64), minlength=cfg.k).tolist(),
            "inertia": micro_diag.get("inertia"),
            "cluster_subset_rows": cluster_n,
            **map_diag,
        }
        np.save(work_dir / "map_table.npy", map_table.astype(np.int64))
        np.save(work_dir / "labels_micro.npy", micro_labels.astype(np.int16))
    else:  # piece_count
        pack, _s, _l = load_train_pack(pack_dir)
        labels = _piece_labels(pack, cancel, emit)
        centroids = np.zeros((PIECE_COUNT_BUCKETS_K, 1), dtype=np.float32)
        diag = {"representation": "piece_count", "sizes": np.bincount(
            labels.astype(np.int64), minlength=PIECE_COUNT_BUCKETS_K).tolist()}
        del pack
    save_cluster_run(work_dir, labels=labels, centroids=centroids, diagnostics=diag)
    if cfg.clustering in ("gradient", "kitsune"):
        _gradient_magnitude_by_partition(
            pack_dir,
            labels,
            int(cfg.k),
            work_dir,
            emit,
            cancel,
            representation=str(diag.get("representation", cfg.clustering)),
        )
    emit(
        {
            "stage": "cluster",
            "event": "done",
            "sizes": diag.get("sizes", []),
            "inertia": diag.get("inertia"),
            "n_rows": int(labels.shape[0]),
        }
    )
    timer.mark("cluster")

    # -- dispatcher ----------------------------------------------------------
    if cfg.clustering == "piece_count":
        disp_labels = labels
        best_val_acc = None
        best_macro_val_acc = None
        disp = None
        emit({"stage": "dispatcher", "event": "skip", "msg": "piece-count rule (no dispatcher)"})
    elif cfg.clustering == "kitsune":
        pack, _s, _l = load_train_pack(pack_dir)
        in_dim = int(base.hidden_dim) * 2
        emit({"stage": "dispatcher", "event": "start", "type": cfg.dispatcher_type})
        # Train the micro-dispatcher on a bounded subset (``dispatcher_max_train``);
        # the full dataset is labeled afterward by a streamed forward pass.
        train_rows = _dispatcher_train_rows(int(len(pack)), cfg.dispatcher_max_train)
        l1_train = build_l1(
            pack, base, device, cancel, emit,
            path=work_dir / "l1_activations.npy", rows=train_rows,
        )
        with open(work_dir / "micro_kmeans.pkl", "rb") as fh:
            micro_km = pickle.load(fh)
        micro_labels_train = _predict_kmeans(micro_km, l1_train, cancel, emit, stage="dispatcher")
        micro_disp, info = _train_dispatcher(
            l1_train, micro_labels_train, cfg, device, emit, cancel,
            n_classes=b_prime, map_table=map_table,
        )
        best_val_acc = info["best_val_acc"]
        best_macro_val_acc = info.get("macro_val_acc")
        del l1_train
        (work_dir / "l1_activations.npy").unlink(missing_ok=True)
        disp = KitsuneDispatcher(in_dim, cfg.k, map_table, micro_disp).to(device)
        disp_labels = _predict_labels_from_pack(disp, pack, base, device, cancel)
        save_dispatcher(
            micro_disp, DISPATCHER_CHECKPOINTS_DIR / cfg.run_name / "dispatcher.pt"
        )
        del pack
        emit(
            {
                "stage": "dispatcher",
                "event": "done",
                "best_val_acc": best_val_acc,
                "macro_val_acc": best_macro_val_acc,
                "label_sizes": np.bincount(
                    disp_labels.astype(np.int64), minlength=cfg.k
                ).tolist(),
            }
        )
    else:
        pack, _s, _l = load_train_pack(pack_dir)
        if cfg.dispatcher_source == "load":
            # A dispatcher is independent of the clustering algorithm: it only
            # needs its input dim and bucket count to match the current base and
            # K. The experts then train on whatever partitions it produces.
            disp = load_dispatcher(cfg.dispatcher_checkpoint, device=device)
            in_dim = int(base.hidden_dim) * 2
            if disp.in_dim != in_dim:
                raise ValueError(
                    f"dispatcher input dim {disp.in_dim} != base L1 concat {in_dim} "
                    f"(load a dispatcher trained on W={base.hidden_dim})"
                )
            if disp.n_clusters != cfg.k:
                raise ValueError(
                    f"dispatcher buckets {disp.n_clusters} != Heads K {cfg.k} "
                    f"(set Heads K to {disp.n_clusters})"
                )
            best_val_acc = None
            best_macro_val_acc = None
            emit(
                {
                    "stage": "dispatcher",
                    "event": "start",
                    "type": "load",
                    "checkpoint": Path(cfg.dispatcher_checkpoint).parent.name,
                }
            )
        else:
            emit({"stage": "dispatcher", "event": "start", "type": cfg.dispatcher_type})
            # Train on a bounded subset (``dispatcher_max_train``); the full
            # dataset is labeled afterward by a streamed forward pass.
            train_rows = _dispatcher_train_rows(int(len(pack)), cfg.dispatcher_max_train)
            l1_train = build_l1(
                pack, base, device, cancel, emit,
                path=work_dir / "l1_activations.npy", rows=train_rows,
            )
            disp, info = _train_dispatcher(l1_train, labels[train_rows], cfg, device, emit, cancel)
            best_val_acc = info["best_val_acc"]
            best_macro_val_acc = best_val_acc
            del l1_train
            (work_dir / "l1_activations.npy").unlink(missing_ok=True)
            # persist the trained dispatcher so it can be loaded in later runs
            save_dispatcher(disp, DISPATCHER_CHECKPOINTS_DIR / cfg.run_name / "dispatcher.pt")
        disp_labels = _predict_labels_from_pack(disp, pack, base, device, cancel)
        del pack
        emit(
            {
                "stage": "dispatcher",
                "event": "done",
                "best_val_acc": best_val_acc,
                "macro_val_acc": best_macro_val_acc,
                "label_sizes": np.bincount(
                    disp_labels.astype(np.int64), minlength=cfg.k
                ).tolist(),
            }
        )
    np.save(work_dir / "labels_dispatcher.npy", disp_labels.astype(np.int16))
    timer.mark("dispatcher")

    # -- experts -------------------------------------------------------------
    pack, _s, _l = load_train_pack(pack_dir)
    pack = pack.to(device)
    emit({"stage": "experts", "event": "start", "k": cfg.k})
    moe = _fine_tune_experts(base, disp_labels, pack, cfg, device, emit, cancel, dispatcher=disp)
    moe.save(work_dir / "moe.pt")
    emit({"stage": "experts", "event": "done"})
    timer.mark("experts")

    # -- eval ----------------------------------------------------------------
    emit({"stage": "eval", "event": "start"})
    metrics = _evaluate(
        base, moe, folders, cfg, device, emit, cancel,
        centroids=centroids, map_table=map_table,
    )
    metrics.update(
        _measure_overfit_and_latency(base, moe, pack, metrics, work_dir, device, cancel)
    )
    (work_dir / "eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    emit({"stage": "eval", "event": "done", **metrics})
    timer.mark("eval")

    summary = {
        "run_name": cfg.run_name,
        "technique": cfg.technique,
        "config": cfg.to_dict(),
        "device": str(device),
        "metrics": metrics,
        "dispatcher_val_acc": best_val_acc,
        "dispatcher_macro_val_acc": best_macro_val_acc,
        "cluster_sizes": diag.get("sizes", []),
        "stage_times": timer.as_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit({"stage": "done", "summary": summary})
    _finalize_run(work_dir, pack_dir, base=base)
    return metrics


def iter_checkpoints() -> list[Path]:
    """List available base checkpoints for the UI dropdown."""
    root = NNUE_CHECKPOINTS_DIR
    if not root.is_dir():
        return []
    out: list[Path] = []
    for best in sorted(root.glob("*/best.pt")):
        out.append(best)
    return out


def iter_checkpoints_info() -> list[dict]:
    """List base checkpoints with their encoder type and widths.

    Each entry: ``{path, name, encoder, is_world_model, hidden_dim, hidden2_dim}``.
    ``encoder`` is read from the checkpoint payload (defaults to ``"standard"``
    for legacy checkpoints without the flag).
    """
    root = NNUE_CHECKPOINTS_DIR
    if not root.is_dir():
        return []
    out: list[dict] = []
    for best in sorted(root.glob("*/best.pt")):
        info = checkpoint_encoder_info(best) or {}
        out.append(
            {
                "path": str(best),
                "name": best.parent.name,
                "encoder": info.get("encoder", "standard"),
                "is_world_model": info.get("encoder", "standard") == "world_model",
                "hidden_dim": info.get("hidden_dim", -1),
                "hidden2_dim": info.get("hidden2_dim", -1),
            }
        )
    return out


def iter_dispatchers() -> list[dict]:
    """List saved dispatcher checkpoints for the UI dropdown.

    Each entry: ``{path, name, dispatcher_type, in_dim, n_clusters}``.
    """
    root = DISPATCHER_CHECKPOINTS_DIR
    if not root.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(root.glob("*/dispatcher.pt")):
        info = dispatcher_checkpoint_info(p) or {}
        out.append(
            {
                "path": str(p),
                "name": p.parent.name,
                "dispatcher_type": info.get("dispatcher_type", "mlp"),
                "in_dim": info.get("in_dim", -1),
                "n_clusters": info.get("n_clusters", -1),
            }
        )
    return out


def iter_history() -> list[dict]:
    """List completed MoE runs for the History tab.

    Scans ``RESULTS/moe/*/summary.json`` and returns one entry per run, newest
    first. Each entry: ``{run_name, path, timestamp, technique, config,
    metrics}``. ``timestamp`` is the ISO string saved at write time (falls back
    to the directory mtime for legacy runs).
    """
    root = MOE_RESULTS_ROOT
    if not root.is_dir():
        return []
    out: list[dict] = []
    for summary in root.glob("*/summary.json"):
        run_name = summary.parent.name
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        timestamp = data.get("timestamp")
        if not timestamp:
            timestamp = datetime.fromtimestamp(
                summary.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")
        out.append(
            {
                "run_name": run_name,
                "path": str(summary.parent),
                "timestamp": timestamp,
                "technique": data.get("technique", ""),
                "config": data.get("config") or {},
                "metrics": data.get("metrics") or {},
            }
        )
    out.sort(key=lambda r: r["timestamp"], reverse=True)
    return out


def delete_run(run_name: str) -> tuple[bool, str]:
    """Delete a run's results folder, its cached data pack, and any ``.zip``.

    Returns ``(ok, message)``. Removes ``RESULTS/moe/<run_name>`` plus the
    matching cached pack dir under ``data/processed/<dataset>/moe/`` so deleting
    a (failed) run also frees the intermediate tensors it left behind.
    """
    root = MOE_RESULTS_ROOT
    run_dir = root / run_name
    zip_path = root / f"{run_name}.zip"
    if not run_dir.is_dir() and not zip_path.exists():
        return False, f"run '{run_name}' not found"
    try:
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        if zip_path.exists():
            zip_path.unlink()
        for pack_dir in UI_PACK_ROOT.glob(f"moe_ui_{run_name}_*"):
            shutil.rmtree(pack_dir, ignore_errors=True)
    except OSError as exc:
        return False, f"failed to delete '{run_name}': {exc}"
    return True, f"deleted '{run_name}'"
