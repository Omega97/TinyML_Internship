"""End-to-end sample-gradient MoE steps used by the CLI scripts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from tinymlinternship.nnue.cluster import (
    assign_to_centroids,
    cluster_diagnostics,
    cluster_size_histogram,
    fit_minibatch_kmeans,
    save_cluster_run,
)
from tinymlinternship.nnue.dataset import FenValueVisitsDataset
from tinymlinternship.nnue.model import DualHiddenNNUE
from tinymlinternship.nnue.moe import DualHiddenMoE, LinearDispatcher, load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import (
    batches_to_device,
    count_rows,
    load_train_pack,
    pack_parts,
    plan_split_indices,
    save_train_pack,
    slice_folders,
    subsample_parts,
)
from tinymlinternship.nnue.moe_plots import (
    plot_centroid_cosine,
    plot_cluster_pca,
    plot_cluster_sizes,
    plot_cluster_tsne,
    plot_confusion,
    plot_dispatcher_acc,
    plot_expert_ce_by_bucket,
    plot_moe_vs_base,
)
from tinymlinternship.nnue.sample_gradients import (
    bytes_per_million,
    head_parameter_dim,
    make_projection_matrix,
    open_float16_store,
    read_grad_meta,
    sample_head_gradients,
    write_grad_meta,
)

DEFAULT_REDUCE_DIM = 48
DEFAULT_GRAD_BATCH = 2048
DEFAULT_CLUSTER_BATCH = 10_000
EPS = 1e-8


def configure_torch(device: torch.device) -> None:
    import os

    torch.set_num_threads(int(os.cpu_count() or 1))
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def ce_and_mae(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = target.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_p = F.log_softmax(logits.float(), dim=-1)
    nll = -(target * log_p).sum(dim=-1)
    probs = torch.softmax(logits.float(), dim=-1)
    pred_v = probs[:, 0] - probs[:, 2]
    tgt_v = target[:, 0] - target[:, 2]
    mae = (pred_v - tgt_v).abs()
    if weight is None:
        w = torch.ones_like(nll)
    else:
        w = weight.float().clamp(min=0.0)
    return (nll * w).sum(), (mae * w).sum(), w.sum()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_gradients(
    model: DualHiddenNNUE,
    folders: list[Path],
    work_dir: Path,
    *,
    device: torch.device,
    test_fraction: float = 0.01,
    split_seed: int = 0,
    subset_seed: int = 0,
    max_rows: int = 1_000_000,
    batch_size: int = DEFAULT_GRAD_BATCH,
    reduce_dim: int = DEFAULT_REDUCE_DIM,
    proj_seed: int = 0,
    resume: bool = False,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    planned = plan_split_indices(folders, test_fraction, seed=split_seed)
    folder_list = [p[0] for p in planned]
    train_parts = subsample_parts([p[1] for p in planned], max_rows, seed=subset_seed)
    n = count_rows(train_parts)
    hidden_dim = model.hidden_dim
    hidden2_dim = model.hidden2_dim
    p_head = head_parameter_dim(hidden_dim, hidden2_dim)
    budget = bytes_per_million(int(reduce_dim))
    if log:
        log(
            f"sample grads: {n:,} rows | head {p_head:,}d → {reduce_dim}d "
            f"| {budget / (1024 ** 2):.1f} MiB / million"
        )
        if budget > 100 * 1024 * 1024:
            log("WARNING: reduce_dim exceeds 100 MiB per million positions")

    grad_path = work_dir / "gradients.npy"
    meta_path = work_dir / "meta.json"
    pack_path = work_dir / "train_pack.pt"

    if resume and pack_path.is_file():
        pack, slice_ids, local_rows = load_train_pack(work_dir)
        if len(pack) != n:
            raise ValueError(f"resume pack has {len(pack)} rows, expected {n}")
    else:
        if log:
            log("packing selected rows into a compact CPU table")
        pack, slice_ids, local_rows = pack_parts(folder_list, train_parts, log=log)
        save_train_pack(work_dir, pack, slice_ids, local_rows)

    n_done = 0
    if resume and meta_path.is_file() and grad_path.is_file():
        prev = read_grad_meta(meta_path)
        n_done = int(prev.get("n_done", 0))
        if int(prev.get("n_rows", -1)) != n or int(prev.get("reduce_dim", -1)) != int(reduce_dim):
            raise ValueError("resume meta does not match this run (n_rows / reduce_dim)")

    store = open_float16_store(grad_path, n, int(reduce_dim), resume=resume and n_done > 0)
    projection = make_projection_matrix(
        p_head, int(reduce_dim), seed=proj_seed, device=device, dtype=torch.float32
    )
    model.eval()
    written = int(n_done)
    t0 = time.perf_counter()
    last_report = t0
    bs = int(batch_size)
    while written < n:
        end = min(written + bs, n)
        batch = batches_to_device(
            pack.gather(torch.arange(written, end, dtype=torch.long)),
            device,
        )
        grads = sample_head_gradients(model, batch, projection)
        k = end - written
        store[written:end] = grads.detach().cpu().numpy().astype(np.float16, copy=False)
        written = end
        now = time.perf_counter()
        if log and (written == n or now - last_report >= 5.0):
            elapsed = now - t0
            rate = (written - n_done) / max(elapsed, 1e-9)
            log(
                f"  grads {written:,}/{n:,} | {rate:,.0f} pos/s "
                f"| {rate * 60:,.0f} / min"
            )
            last_report = now
            write_grad_meta(
                meta_path,
                {
                    "n_rows": n,
                    "n_done": written,
                    "reduce_dim": int(reduce_dim),
                    "head_dim": p_head,
                    "hidden_dim": hidden_dim,
                    "hidden2_dim": hidden2_dim,
                    "proj_seed": int(proj_seed),
                    "bytes_per_million": budget,
                },
            )

    store.flush()
    elapsed = time.perf_counter() - t0
    rate = (written - n_done) / max(elapsed, 1e-9)
    meta = {
        "n_rows": n,
        "n_done": written,
        "reduce_dim": int(reduce_dim),
        "head_dim": p_head,
        "hidden_dim": hidden_dim,
        "hidden2_dim": hidden2_dim,
        "proj_seed": int(proj_seed),
        "bytes_per_million": budget,
        "pos_per_s": rate,
        "seconds": elapsed,
        "folders": [str(p) for p in folder_list],
        "test_fraction": float(test_fraction),
        "split_seed": int(split_seed),
        "subset_seed": int(subset_seed),
        "max_rows": int(max_rows),
    }
    write_grad_meta(meta_path, meta)
    if log:
        log(f"wrote {written:,} grads in {elapsed:.1f}s ({rate:,.0f} pos/s)")
    return meta


def cluster_gradients(
    work_dir: Path,
    *,
    n_clusters: int,
    batch_size: int = DEFAULT_CLUSTER_BATCH,
    seed: int = 0,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    grads = np.load(work_dir / "gradients.npy", mmap_mode="r")
    if log:
        log(f"k-means B={n_clusters} on {grads.shape[0]:,} × {grads.shape[1]}")
    km = fit_minibatch_kmeans(
        grads, int(n_clusters), batch_size=int(batch_size), seed=int(seed)
    )
    labels = km.predict(np.ascontiguousarray(grads, dtype=np.float32)).astype(np.int16)
    diag = cluster_diagnostics(labels, km.cluster_centers_, inertia=float(km.inertia_))
    save_cluster_run(work_dir, labels=labels, centroids=km.cluster_centers_, diagnostics=diag)
    if log:
        log(f"cluster sizes {diag['sizes']} | empty={diag['empty']}")
    return diag


@torch.inference_mode()
def collect_dispatcher_labels(
    model: DualHiddenNNUE | DualHiddenMoE,
    dispatcher: LinearDispatcher,
    pack,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n = len(pack)
    out = np.empty(n, dtype=np.int16)
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        batch = batches_to_device(
            pack.gather(torch.arange(start, end, dtype=torch.long)),
            device,
        )
        h = model.l1_concat(
            batch["white_idx"],
            batch["black_idx"],
            batch["stm_white"],
            batch["white_mask"],
            batch["black_mask"],
        )
        out[start:end] = (
            dispatcher.predict(h).detach().cpu().numpy().astype(np.int16, copy=False)
        )
    return out


def train_dispatcher(
    model: DualHiddenNNUE,
    work_dir: Path,
    folders: list[Path],
    *,
    device: torch.device,
    n_clusters: int,
    epochs: int = 8,
    batch_size: int = 1024,
    lr: float = 1e-2,
    lr_end: float = 1e-3,
    val_fraction: float = 0.10,
    seed: int = 0,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    labels = np.load(work_dir / "labels.npy")
    pack, _slice_ids, _local_rows = load_train_pack(work_dir)
    n = int(labels.shape[0])
    if len(pack) != n:
        raise ValueError(f"pack {len(pack)} vs labels {n}")
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n)
    n_val = max(1, int(n * float(val_fraction)))
    is_val = np.zeros(n, dtype=np.bool_)
    is_val[perm[:n_val]] = True

    in_dim = model.hidden_dim * 2
    dispatcher = LinearDispatcher(in_dim, int(n_clusters)).to(device)
    opt = torch.optim.Adam(dispatcher.parameters(), lr=float(lr))
    sched = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1.0,
        end_factor=float(lr_end) / float(lr) if lr > 0 else 1.0,
        total_iters=max(int(epochs), 1),
    )
    history: list[dict[str, Any]] = []
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in dispatcher.state_dict().items()}

    def _split_stream(want_val: bool):
        positions = np.flatnonzero(is_val if want_val else ~is_val).astype(np.int64)
        for start in range(0, int(positions.size), int(batch_size)):
            idx = positions[start : start + int(batch_size)]
            batch = batches_to_device(pack.gather(torch.from_numpy(idx)), device)
            y = torch.from_numpy(np.asarray(labels[idx], dtype=np.int64)).to(device)
            yield batch, y

    model.eval()
    for epoch in range(1, int(epochs) + 1):
        dispatcher.train()
        t0 = time.perf_counter()
        train_correct = 0
        train_n = 0
        train_loss = 0.0
        for batch, y in _split_stream(False):
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                h = model.l1_concat(
                    batch["white_idx"],
                    batch["black_idx"],
                    batch["stm_white"],
                    batch["white_mask"],
                    batch["black_mask"],
                )
            logits = dispatcher(h)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * int(y.shape[0])
            train_correct += int((logits.argmax(-1) == y).sum().item())
            train_n += int(y.shape[0])
        dispatcher.eval()
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for batch, y in _split_stream(True):
                h = model.l1_concat(
                    batch["white_idx"],
                    batch["black_idx"],
                    batch["stm_white"],
                    batch["white_mask"],
                    batch["black_mask"],
                )
                pred = dispatcher(h).argmax(dim=-1)
                val_correct += int((pred == y).sum().item())
                val_n += int(y.shape[0])
        sched.step()
        train_acc = train_correct / max(train_n, 1)
        val_acc = val_correct / max(val_n, 1)
        row = {
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_loss": train_loss / max(train_n, 1),
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        if log:
            log(
                f"dispatcher epoch {epoch:02d} | train_acc={train_acc:.3f} | "
                f"val_acc={val_acc:.3f}"
            )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in dispatcher.state_dict().items()}

    dispatcher.load_state_dict(best_state)
    dispatcher.to(device)
    torch.save(
        {
            "state_dict": dispatcher.state_dict(),
            "in_dim": in_dim,
            "n_clusters": int(n_clusters),
            "val_acc": best_acc,
        },
        work_dir / "dispatcher.pt",
    )
    _write_json(work_dir / "dispatcher_history.json", {"best_val_acc": best_acc, "history": history})
    if log:
        log("re-assigning buckets with dispatcher argmax")
    disp_labels = collect_dispatcher_labels(
        model,
        dispatcher,
        pack,
        device=device,
        batch_size=int(batch_size),
    )
    np.save(work_dir / "labels_dispatcher.npy", disp_labels)
    return {"best_val_acc": best_acc, "history": history, "n_train": n - n_val, "n_val": n_val}


def fine_tune_experts(
    base: DualHiddenNNUE,
    work_dir: Path,
    folders: list[Path],
    *,
    device: torch.device,
    n_clusters: int,
    expert_labels: str = "dispatcher",
    epochs: int = 2,
    batch_size: int = 2048,
    lr: float = 1e-3,
    holdout_fraction: float = 0.10,
    seed: int = 0,
    log: Callable[[str], None] | None = print,
) -> DualHiddenMoE:
    work_dir = Path(work_dir)
    name = "labels_dispatcher.npy" if expert_labels == "dispatcher" else "labels.npy"
    labels = np.load(work_dir / name)
    pack, _slice_ids, _local_rows = load_train_pack(work_dir)
    moe = DualHiddenMoE.from_base(base, int(n_clusters)).to(device)
    moe.freeze_l1()
    disp_path = work_dir / "dispatcher.pt"
    if disp_path.is_file():
        payload = torch.load(disp_path, map_location="cpu", weights_only=False)
        moe.dispatcher.load_state_dict(payload["state_dict"])
    per_bucket: list[dict[str, float]] = []
    rng = np.random.RandomState(int(seed))

    for expert_id in range(int(n_clusters)):
        mask = labels.astype(np.int64) == int(expert_id)
        idx_all = np.flatnonzero(mask)
        if idx_all.size < 4:
            if log:
                log(f"expert {expert_id}: {int(idx_all.size)} rows — keep base head")
            per_bucket.append({"expert": expert_id, "n": int(idx_all.size), "skipped": 1})
            continue
        perm = rng.permutation(idx_all.size)
        n_hold = max(1, int(idx_all.size * float(holdout_fraction)))
        hold_pos = np.sort(idx_all[perm[:n_hold]])
        train_pos = np.sort(idx_all[perm[n_hold:]])
        if train_pos.size < 1:
            train_pos, hold_pos = hold_pos, train_pos

        expert = DualHiddenNNUE(
            hidden_dim=base.hidden_dim,
            hidden2_dim=base.hidden2_dim,
            crelu_clip=base.crelu_clip,
        )
        expert.load_state_dict(base.state_dict())
        expert.to(device)
        for param in expert.l1.parameters():
            param.requires_grad = False
        opt = torch.optim.Adam(
            [p for p in expert.parameters() if p.requires_grad],
            lr=float(lr),
        )

        def _batches(positions: np.ndarray):
            if positions.size == 0:
                return
            pos = np.asarray(positions, dtype=np.int64)
            for start in range(0, int(pos.size), int(batch_size)):
                idx = pos[start : start + int(batch_size)]
                yield batches_to_device(pack.gather(torch.from_numpy(idx)), device)

        @torch.inference_mode()
        def _eval(positions: np.ndarray) -> float:
            expert.eval()
            ce_sum = torch.zeros((), device=device)
            w_sum = torch.zeros((), device=device)
            for batch in _batches(positions):
                logits = expert(
                    batch["white_idx"],
                    batch["black_idx"],
                    batch["stm_white"],
                    batch["white_mask"],
                    batch["black_mask"],
                )
                ce, _mae, w = ce_and_mae(logits, batch["target"], batch["weight"])
                ce_sum = ce_sum + ce
                w_sum = w_sum + w
            return float((ce_sum / w_sum.clamp_min(1e-8)).item()) if float(w_sum) > 0 else float("nan")

        base_hold = _eval(hold_pos)
        best_ce = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}
        for epoch in range(1, int(epochs) + 1):
            expert.train()
            for batch in _batches(train_pos):
                opt.zero_grad(set_to_none=True)
                logits = expert(
                    batch["white_idx"],
                    batch["black_idx"],
                    batch["stm_white"],
                    batch["white_mask"],
                    batch["black_mask"],
                )
                ce, _mae, w = ce_and_mae(logits, batch["target"], batch["weight"])
                loss = ce / w.clamp_min(1e-8)
                loss.backward()
                opt.step()
            hold_ce = _eval(hold_pos)
            if log:
                log(
                    f"expert {expert_id} epoch {epoch} | hold_ce={hold_ce:.5f} "
                    f"(base {base_hold:.5f}) n={int(train_pos.size)}"
                )
            if hold_ce < best_ce:
                best_ce = hold_ce
                best_state = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}
        expert.load_state_dict(best_state)
        moe.experts_l2[expert_id].load_state_dict(expert.l2.state_dict())
        moe.experts_head[expert_id].load_state_dict(expert.head.state_dict())
        per_bucket.append(
            {
                "expert": expert_id,
                "n_train": int(train_pos.size),
                "n_hold": int(hold_pos.size),
                "base_hold_ce": base_hold,
                "expert_hold_ce": best_ce if best_ce < float("inf") else base_hold,
                "skipped": 0,
            }
        )

    moe.save(work_dir / "moe.pt")
    _write_json(work_dir / "expert_metrics.json", {"expert_labels": expert_labels, "buckets": per_bucket})
    return moe


@torch.inference_mode()
def evaluate_moe(
    base: DualHiddenNNUE,
    moe: DualHiddenMoE,
    folders: list[Path],
    work_dir: Path,
    *,
    device: torch.device,
    test_fraction: float = 0.01,
    split_seed: int = 0,
    max_test: int = 50_000,
    subset_seed: int = 1,
    batch_size: int = 2048,
    oracle: bool = True,
    log: Callable[[str], None] | None = print,
) -> dict[str, float]:
    work_dir = Path(work_dir)
    planned = plan_split_indices(folders, test_fraction, seed=split_seed)
    test_parts = subsample_parts([p[2] for p in planned], max_test, seed=subset_seed)
    centroids = None
    projection = None
    if oracle and (work_dir / "centroids.npy").is_file() and (work_dir / "meta.json").is_file():
        centroids = np.load(work_dir / "centroids.npy")
        meta = read_grad_meta(work_dir / "meta.json")
        projection = make_projection_matrix(
            int(meta["head_dim"]),
            int(meta["reduce_dim"]),
            seed=int(meta["proj_seed"]),
            device=device,
            dtype=torch.float32,
        )

    def _acc() -> dict[str, torch.Tensor]:
        zeros = {
            "base_ce": torch.zeros((), device=device),
            "base_mae": torch.zeros((), device=device),
            "moe_ce": torch.zeros((), device=device),
            "moe_mae": torch.zeros((), device=device),
            "oracle_ce": torch.zeros((), device=device),
            "oracle_mae": torch.zeros((), device=device),
            "w": torch.zeros((), device=device),
        }
        return zeros

    stats = _acc()
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
            base_logits = base(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            moe_logits = moe(
                batch["white_idx"],
                batch["black_idx"],
                batch["stm_white"],
                batch["white_mask"],
                batch["black_mask"],
            )
            bce, bmae, w = ce_and_mae(base_logits, batch["target"], batch["weight"])
            mce, mmae, _ = ce_and_mae(moe_logits, batch["target"], batch["weight"])
            stats["base_ce"] += bce
            stats["base_mae"] += bmae
            stats["moe_ce"] += mce
            stats["moe_mae"] += mmae
            stats["w"] += w
            n_rows += int(idx.size)
            if projection is not None and centroids is not None:
                grads = sample_head_gradients(base, batch, projection)
                oracle_ids = assign_to_centroids(
                    grads.detach().cpu().numpy(), centroids
                )
                oracle_logits, _ = moe.route_from_h(
                    moe.l1_concat(
                        batch["white_idx"],
                        batch["black_idx"],
                        batch["stm_white"],
                        batch["white_mask"],
                        batch["black_mask"],
                    ),
                    expert_ids=torch.from_numpy(oracle_ids.astype(np.int64)).to(device),
                )
                oce, omae, _ = ce_and_mae(oracle_logits, batch["target"], batch["weight"])
                stats["oracle_ce"] += oce
                stats["oracle_mae"] += omae
        del ds

    denom = stats["w"].clamp_min(1e-8)
    metrics = {
        "n_test": float(n_rows),
        "base_ce": float((stats["base_ce"] / denom).item()),
        "base_mae": float((stats["base_mae"] / denom).item()),
        "moe_ce": float((stats["moe_ce"] / denom).item()),
        "moe_mae": float((stats["moe_mae"] / denom).item()),
    }
    if projection is not None:
        metrics["oracle_ce"] = float((stats["oracle_ce"] / denom).item())
        metrics["oracle_mae"] = float((stats["oracle_mae"] / denom).item())
    _write_json(work_dir / "eval.json", metrics)
    if log:
        log(
            f"eval n={n_rows:,} | base_ce={metrics['base_ce']:.5f} | "
            f"moe_ce={metrics['moe_ce']:.5f}"
            + (f" | oracle_ce={metrics['oracle_ce']:.5f}" if "oracle_ce" in metrics else "")
        )
    return metrics


def write_plots(work_dir: Path, plots_dir: Path) -> list[Path]:
    work_dir = Path(work_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    grads = np.load(work_dir / "gradients.npy", mmap_mode="r")
    km_labels = np.load(work_dir / "labels.npy")
    disp_path = work_dir / "labels_dispatcher.npy"
    disp_labels = np.load(disp_path) if disp_path.is_file() else None
    centroids = np.load(work_dir / "centroids.npy")
    diag = json.loads((work_dir / "diagnostics.json").read_text(encoding="utf-8"))
    written: list[Path] = []
    written.append(plot_cluster_pca(grads, km_labels, plots_dir / "cluster_pca.png"))
    written.append(plot_cluster_tsne(grads, km_labels, plots_dir / "cluster_tsne.png"))
    km_sizes = cluster_size_histogram(km_labels, int(centroids.shape[0]))
    disp_sizes = (
        cluster_size_histogram(disp_labels, int(centroids.shape[0])) if disp_labels is not None else None
    )
    written.append(plot_cluster_sizes(km_sizes, disp_sizes, plots_dir / "cluster_sizes.png"))
    written.append(
        plot_centroid_cosine(np.asarray(diag["centroid_cosine"]), plots_dir / "centroid_cosine.png")
    )
    hist_path = work_dir / "dispatcher_history.json"
    if hist_path.is_file():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        if hist.get("history"):
            written.append(plot_dispatcher_acc(hist["history"], plots_dir / "dispatcher_acc.png"))
    if disp_labels is not None:
        written.append(
            plot_confusion(
                km_labels,
                disp_labels,
                int(centroids.shape[0]),
                plots_dir / "dispatcher_confusion.png",
            )
        )
    expert_path = work_dir / "expert_metrics.json"
    if expert_path.is_file():
        payload = json.loads(expert_path.read_text(encoding="utf-8"))
        buckets = [b for b in payload.get("buckets", []) if not b.get("skipped")]
        if buckets:
            written.append(
                plot_expert_ce_by_bucket(
                    [float(b["base_hold_ce"]) for b in buckets],
                    [float(b["expert_hold_ce"]) for b in buckets],
                    plots_dir / "expert_ce_by_bucket.png",
                )
            )
    eval_path = work_dir / "eval.json"
    if eval_path.is_file():
        metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        written.append(plot_moe_vs_base(metrics, plots_dir / "moe_vs_base_ce.png"))
    return written


def load_base_on_device(checkpoint: Path, device: torch.device) -> DualHiddenNNUE:
    model = load_dual_hidden_checkpoint(checkpoint, device=device)
    model.eval()
    return model
