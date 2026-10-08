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
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, SARDINE_MODELS_DIR
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
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
)
from tinymlinternship.nnue.moe_pipeline import (
    DEFAULT_REDUCE_DIM,
    ce_and_mae,
    compute_gradients,
    configure_torch,
)
from tinymlinternship.nnue.optimizers import Rprop
from tinymlinternship.nnue.sample_gradients import (
    head_parameter_dim,
    make_projection_matrix,
    sample_head_gradients,
)

DEFAULT_SLICES = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
UI_PACK_ROOT = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe"
DISPATCHER_CHECKPOINTS_DIR = SARDINE_MODELS_DIR / "checkpoints" / "dispatchers"

L1_CHUNK = 16_384
CLUSTER_CHUNK = 1_000_000
DISPATCHER_BATCH = 4096
EXPERT_BATCH = 2048
EVAL_BATCH = 4096
SWITCH_BATCH = 2048
PIECE_COUNT_BUCKETS_K = 8


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
    technique: str = "hard_moe"  # "hard_moe" | "switch"
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
    clustering: str = "gradient"  # "gradient" | "l1" | "piece_count"
    k: int = 8
    # dispatcher (hard MoE)
    dispatcher_type: str = "mlp"  # "mlp" | "linear"
    dispatcher_hidden: int = 64
    dispatcher_epochs: int = 8
    dispatcher_lr: float = 1e-2
    dispatcher_source: str = "train"  # "train" | "load"
    dispatcher_checkpoint: str = ""
    # experts (hard MoE)
    expert_epochs: int = 2
    expert_lr: float = 1e-3
    expert_lr_end: float = 1e-4
    expert_optimizer: str = "adam"  # "adam" | "sgd" | "rprop"
    l1_frozen: bool = True
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
    pack, base: DualHiddenNNUE, device: torch.device, cancel: CancelToken, emit: EmitFn
) -> np.ndarray:
    n = len(pack)
    hidden = int(base.hidden_dim)
    out = np.empty((n, 2 * hidden), dtype=np.float32)
    base.eval()
    with torch.inference_mode():
        for start in range(0, n, L1_CHUNK):
            cancel.check()
            end = min(start + L1_CHUNK, n)
            batch = pack.gather(torch.arange(start, end, dtype=torch.long))
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
def _kmeans_labels(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    km = fit_minibatch_kmeans(matrix, k, seed=0)
    n = int(matrix.shape[0])
    labels = np.empty(n, dtype=np.int16)
    for start in range(0, n, CLUSTER_CHUNK):
        end = min(start + CLUSTER_CHUNK, n)
        labels[start:end] = km.predict(
            np.ascontiguousarray(matrix[start:end], dtype=np.float32)
        ).astype(np.int16)
    centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
    diag = cluster_diagnostics(labels, centroids, inertia=float(km.inertia_))
    return labels, centroids, diag


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
# Dispatcher stage
# --------------------------------------------------------------------------- #
def _train_dispatcher(
    l1: np.ndarray,
    labels: np.ndarray,
    cfg: TrainingConfig,
    device: torch.device,
    emit: EmitFn,
    cancel: CancelToken,
) -> tuple[object, dict]:
    n = int(l1.shape[0])
    in_dim = int(l1.shape[1])
    if cfg.dispatcher_type == "mlp":
        disp = MLPDispatcher(in_dim, cfg.k, hidden_dim=cfg.dispatcher_hidden).to(device)
    else:
        disp = LinearDispatcher(in_dim, cfg.k).to(device)

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.10))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    idx_train = torch.from_numpy(train_idx.astype(np.int64)).to(device)
    idx_val = torch.from_numpy(val_idx.astype(np.int64)).to(device)
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

    disp.eval()
    train_correct0 = 0
    with torch.no_grad():
        for start in range(0, n_train, DISPATCHER_BATCH):
            end = min(start + DISPATCHER_BATCH, n_train)
            x = torch.from_numpy(
                np.ascontiguousarray(l1[train_idx[start:end]], dtype=np.float32)
            ).to(device)
            y = y_train[start:end]
            train_correct0 += int((disp(x).argmax(dim=-1) == y).sum().item())
    val_correct0 = 0
    with torch.no_grad():
        for start in range(0, n_val_v, DISPATCHER_BATCH):
            end = min(start + DISPATCHER_BATCH, n_val_v)
            x = torch.from_numpy(
                np.ascontiguousarray(l1[val_idx[start:end]], dtype=np.float32)
            ).to(device)
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
            x = torch.from_numpy(
                np.ascontiguousarray(l1[train_idx[start:end]], dtype=np.float32)
            ).to(device)
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
                x = torch.from_numpy(
                    np.ascontiguousarray(l1[val_idx[start:end]], dtype=np.float32)
                ).to(device)
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
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in disp.state_dict().items()}

    disp.load_state_dict(best_state)
    disp.eval()
    return disp, {"best_val_acc": best_acc}


def _predict_labels(disp, l1: np.ndarray, device: torch.device, cancel: CancelToken) -> np.ndarray:
    n = int(l1.shape[0])
    out = np.empty(n, dtype=np.int16)
    with torch.no_grad():
        for start in range(0, n, L1_CHUNK):
            cancel.check()
            end = min(start + L1_CHUNK, n)
            x = torch.from_numpy(np.ascontiguousarray(l1[start:end], dtype=np.float32)).to(device)
            out[start:end] = disp(x).argmax(dim=-1).cpu().numpy().astype(np.int16)
    return out


# --------------------------------------------------------------------------- #
# Expert stage
# --------------------------------------------------------------------------- #
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
        hold_pos = np.sort(idx_all[perm[:n_hold]])
        train_pos = np.sort(idx_all[perm[n_hold:]])

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
        for epoch in range(1, cfg.expert_epochs + 1):
            cancel.check()
            expert.train()
            if full_batch:
                # snapshot weights before the step for loss-based backtracking
                prev_state = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}
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
                opt.step()
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
            else:
                prev_hold = hold_ce
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
                    "progress": epoch / cfg.expert_epochs,
                }
            )
            if hold_ce < best_ce:
                best_ce = hold_ce
                best_state = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}

        expert.load_state_dict(best_state)
        moe.experts_l2[expert_id].load_state_dict(expert.l2.state_dict())
        moe.experts_head[expert_id].load_state_dict(expert.head.state_dict())
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


# --------------------------------------------------------------------------- #
# Top-level pipeline
# --------------------------------------------------------------------------- #
def run_moe_training(cfg: TrainingConfig, emit: EmitFn, cancel: CancelToken) -> dict:
    device = _resolve_device(cfg.device)
    configure_torch(device)
    emit({"stage": "init", "event": "start", "config": cfg.to_dict(), "device": str(device)})

    slices_dir = DEFAULT_SLICES
    folders = slice_folders(slices_dir)
    if not folders:
        raise RuntimeError(f"no slice folders in {slices_dir}")

    # -- resolve the base model up front so the gradient-cache directory can be
    # keyed by the actual architecture + row count. Otherwise a new arch / row
    # count would reuse (and clash with) a stale cache from a previous run.
    use_world_model = cfg.encoder == "world_model"
    if use_world_model and cfg.world_model_source == "load":
        wm_ckpt = Path(cfg.wm_checkpoint)
        base = load_dual_hidden_checkpoint(wm_ckpt, device=device)
        if not base.is_world_model:
            raise ValueError(
                f"{wm_ckpt.name} is not a world-model checkpoint (encoder={base.encoder!r})"
            )
        h_actual, H_actual = base.hidden_dim, base.hidden2_dim
    elif cfg.base_source == "load":
        ckpt = Path(cfg.base_checkpoint)
        base = load_dual_hidden_checkpoint(ckpt, device=device)
        h_actual, H_actual = base.hidden_dim, base.hidden2_dim
    else:
        base = DualHiddenNNUE(hidden_dim=cfg.h, hidden2_dim=cfg.H).to(device)
        h_actual, H_actual = cfg.h, cfg.H

    pack_dir = UI_PACK_ROOT / f"moe_ui_{cfg.run_name}_h{h_actual}_H{H_actual}_n{cfg.max_rows}"
    pack_dir.mkdir(parents=True, exist_ok=True)
    work_dir = PROJECT_ROOT / "RESULTS" / "moe" / cfg.run_name
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
        emit({"stage": "base", "event": "done", "source": "new", "h": cfg.h, "H": cfg.H})

    # -- base-only (K=1, no MoE) --------------------------------------------
    if int(cfg.k) <= 1:
        emit({"stage": "eval", "event": "start"})
        metrics = _evaluate_base(base, folders, cfg, device, emit, cancel)
        (work_dir / "eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        emit({"stage": "eval", "event": "done", **metrics})
        summary = {
            "run_name": cfg.run_name,
            "technique": "base",
            "config": cfg.to_dict(),
            "metrics": metrics,
        }
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        emit({"stage": "done", "summary": summary})
        return metrics

    # -- switch (end-to-end sparse top-1 MoE) --------------------------------
    if cfg.technique == "switch":
        pack, _s, _l = load_train_pack(pack_dir)
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
        del pack
        emit({"stage": "switch", "event": "done"})

        emit({"stage": "eval", "event": "start"})
        metrics = _switch_eval(base, moe, folders, cfg, device, cancel, full=True, emit=emit)
        (work_dir / "eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        emit({"stage": "eval", "event": "done", **metrics})

        summary = {
            "run_name": cfg.run_name,
            "technique": cfg.technique,
            "config": cfg.to_dict(),
            "metrics": metrics,
            "best_test_ce": best_test,
        }
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        emit({"stage": "done", "summary": summary})
        return metrics

    # -- cluster -------------------------------------------------------------
    emit({"stage": "cluster", "event": "start", "clustering": cfg.clustering, "k": cfg.k})
    if cfg.clustering == "gradient":
        compute_gradients(
            base, folders, pack_dir, device=device, max_rows=cfg.max_rows,
            resume=True, log=_loading_forwarder(emit, cancel),
        )
        gradients = np.load(pack_dir / "gradients.npy", mmap_mode="r")
        labels, centroids, diag = _kmeans_labels(gradients, cfg.k)
        diag["representation"] = "head_gradient_48d"
        del gradients
    elif cfg.clustering == "l1":
        pack, _s, _l = load_train_pack(pack_dir)
        l1 = build_l1(pack, base, device, cancel, emit)
        labels, centroids, diag = _kmeans_labels(l1, cfg.k)
        diag["representation"] = "l1_256d"
        del l1
        del pack
    else:  # piece_count
        pack, _s, _l = load_train_pack(pack_dir)
        labels = _piece_labels(pack, cancel, emit)
        centroids = np.zeros((PIECE_COUNT_BUCKETS_K, 1), dtype=np.float32)
        diag = {"representation": "piece_count", "sizes": np.bincount(
            labels.astype(np.int64), minlength=PIECE_COUNT_BUCKETS_K).tolist()}
        del pack
    save_cluster_run(work_dir, labels=labels, centroids=centroids, diagnostics=diag)
    emit(
        {
            "stage": "cluster",
            "event": "done",
            "sizes": diag.get("sizes", []),
            "inertia": diag.get("inertia"),
            "n_rows": int(labels.shape[0]),
        }
    )

    # -- dispatcher ----------------------------------------------------------
    if cfg.clustering == "piece_count":
        disp_labels = labels
        best_val_acc = None
        disp = None
        emit({"stage": "dispatcher", "event": "skip", "msg": "piece-count rule (no dispatcher)"})
    else:
        pack, _s, _l = load_train_pack(pack_dir)
        l1 = build_l1(pack, base, device, cancel, emit)
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
            disp_labels = _predict_labels(disp, l1, device, cancel)
            best_val_acc = None
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
            disp, info = _train_dispatcher(l1, labels, cfg, device, emit, cancel)
            best_val_acc = info["best_val_acc"]
            disp_labels = _predict_labels(disp, l1, device, cancel)
            # persist the trained dispatcher so it can be loaded in later runs
            save_dispatcher(disp, DISPATCHER_CHECKPOINTS_DIR / cfg.run_name / "dispatcher.pt")
        del l1
        del pack
        emit(
            {
                "stage": "dispatcher",
                "event": "done",
                "best_val_acc": best_val_acc,
                "label_sizes": np.bincount(
                    disp_labels.astype(np.int64), minlength=cfg.k
                ).tolist(),
            }
        )
    np.save(work_dir / "labels_dispatcher.npy", disp_labels.astype(np.int16))

    # -- experts -------------------------------------------------------------
    pack, _s, _l = load_train_pack(pack_dir)
    emit({"stage": "experts", "event": "start", "k": cfg.k})
    moe = _fine_tune_experts(base, disp_labels, pack, cfg, device, emit, cancel, dispatcher=disp)
    moe.save(work_dir / "moe.pt")
    emit({"stage": "experts", "event": "done"})

    # -- eval ----------------------------------------------------------------
    emit({"stage": "eval", "event": "start"})
    metrics = _evaluate(base, moe, folders, cfg, device, emit, cancel, centroids=centroids)
    (work_dir / "eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    emit({"stage": "eval", "event": "done", **metrics})

    summary = {
        "run_name": cfg.run_name,
        "technique": cfg.technique,
        "config": cfg.to_dict(),
        "metrics": metrics,
        "dispatcher_val_acc": best_val_acc,
        "cluster_sizes": diag.get("sizes", []),
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit({"stage": "done", "summary": summary})
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
