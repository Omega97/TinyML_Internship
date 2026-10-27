#!/usr/bin/env python3
"""Loss landscapes of the shared NNUE head on three partitions.

Same checkpoint, same pack. For each bucket of the random partition, the L1
k-means partition, and KITSUNE (L1 micro-buckets stitched by mean head
gradient), draw the mean soft cross-entropy on a plane spanned by the
bucket gradient and a random orthogonal direction.

The x axis is that gradient. The y axis is not shared across panels.

Each panel grid is saved under ``grids/<figure stem>/``. A later run with the
same checkpoint, pack, labels, and numeric arguments loads those files and
skips the loss evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.cm import ScalarMappable
from matplotlib.colors import CenteredNorm

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME
from tinymlinternship.nnue.cluster import fit_minibatch_kmeans
from tinymlinternship.nnue.moe import load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import load_train_pack
from tinymlinternship.nnue.moe_pipeline import ce_and_mae

CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
PACK_DIR = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / "moe" / "moe_b4_2m"
L1_LABELS = PROJECT_ROOT / "RESULTS" / "moe" / "l1_clustered_k4" / "labels.npy"
OUT_DIR = PROJECT_ROOT / "RESULTS" / "loss_plots"

METHODS = ("random", "l1", "kitsune")
ROW_TITLES = {
    "random": "Random",
    "l1": "L1 clustering",
    "kitsune": "KITSUNE",
}
HEAD_NAMES = ("l2.weight", "l2.bias", "head.weight", "head.bias")
L1_CHUNK = 16_384
GRAD_CHUNK = 8_192


def log(message: str) -> None:
    print(message, flush=True)


def direction_seed(method: str, bucket: int) -> int:
    digest = hashlib.sha256(f"{method}:{bucket}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def sample_indices(pool: np.ndarray, n: int, method: str, bucket: int) -> np.ndarray:
    """Draw ``n`` rows from one bucket. Seed 0, independent of loop order."""
    seed = int.from_bytes(
        hashlib.sha256(f"sample:0:{method}:{bucket}".encode()).digest()[:8],
        "little",
    ) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    take = min(int(n), int(pool.shape[0]))
    choice = rng.choice(pool, size=take, replace=False)
    return np.sort(choice.astype(np.int64))


def head_tensors(model: torch.nn.Module) -> list[torch.Tensor]:
    return [model.l2.weight, model.l2.bias, model.head.weight, model.head.bias]


def flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1).float() for tensor in tensors])


def split_flat(flat: torch.Tensor, shapes: list[tuple[int, ...]]) -> list[torch.Tensor]:
    pieces: list[torch.Tensor] = []
    offset = 0
    for shape in shapes:
        count = int(np.prod(shape))
        pieces.append(flat[offset : offset + count].view(shape))
        offset += count
    return pieces


def head_logits(
    h: torch.Tensor,
    pieces: list[torch.Tensor],
    clip: float,
) -> torch.Tensor:
    pre = F.linear(h, pieces[0], pieces[1])
    h2 = torch.clamp(pre, min=0.0, max=float(clip))
    return F.linear(h2, pieces[2], pieces[3])


def weighted_mean_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training soft CE as a weighted mean, plus the sum that backprops onto it."""
    nll_sum, _mae_sum, w_sum = ce_and_mae(logits, target, weight)
    return nll_sum / w_sum.clamp_min(1e-8), nll_sum


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
            if end == n or (start // L1_CHUNK) % 20 == 0:
                log(f"  l1 {end:,}/{n:,}")
    return out


def accumulate_micro_gradients(
    pack,
    model,
    micro: np.ndarray,
    device: torch.device,
    n_micro: int,
) -> np.ndarray:
    """Mean head gradient of each micro-bucket, shape ``(B', P)``."""
    shapes = [tuple(tensor.shape) for tensor in head_tensors(model)]
    base = [tensor.detach().float().to(device) for tensor in head_tensors(model)]
    totals = [
        [torch.zeros(shape, dtype=torch.float64, device=device) for shape in shapes]
        for _ in range(n_micro)
    ]
    weight_sums = np.zeros(n_micro, dtype=np.float64)
    clip = float(model.crelu_clip)
    n = len(pack)
    log("KITSUNE: mean head gradient per micro-bucket")
    for start in range(0, n, GRAD_CHUNK):
        end = min(start + GRAD_CHUNK, n)
        batch = pack.gather(torch.arange(start, end, dtype=torch.long))
        with torch.no_grad():
            h = model.l1_concat(
                batch["white_idx"].to(device),
                batch["black_idx"].to(device),
                batch["stm_white"].to(device),
                batch["white_mask"].to(device),
                batch["black_mask"].to(device),
            ).detach().clone()
        target = batch["target"].to(device)
        weight = batch["weight"].to(device)
        chunk_labels = micro[start:end]
        for k in np.unique(chunk_labels):
            sel = torch.from_numpy(chunk_labels == int(k)).to(device)
            leaves = [piece.detach().clone().requires_grad_(True) for piece in base]
            _mean, nll_sum = weighted_mean_ce(
                head_logits(h[sel], leaves, clip),
                target[sel],
                weight[sel],
            )
            grads = torch.autograd.grad(nll_sum, leaves)
            bucket = totals[int(k)]
            for slot, grad in zip(bucket, grads):
                slot.add_(grad.detach().double())
            weight_sums[int(k)] += float(weight[sel].sum().detach().cpu())
        if end == n or (start // GRAD_CHUNK) % 20 == 0:
            log(f"  micro grad {end:,}/{n:,}")

    rows = []
    for k in range(n_micro):
        if weight_sums[k] <= 0:
            raise RuntimeError(f"KITSUNE micro-bucket {k} has no weight")
        flat = torch.cat([(slot / weight_sums[k]).reshape(-1) for slot in totals[k]])
        rows.append(flat.cpu().numpy().astype(np.float64))
    return np.stack(rows, axis=0)


def stitch_kitsune(prototypes: np.ndarray, n_buckets: int) -> np.ndarray:
    """Cosine k-means: Euclidean k-means on L2-normalized mean gradients."""
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    hat = prototypes / np.clip(norms, 1e-12, None)
    km = fit_minibatch_kmeans(hat.astype(np.float32), n_buckets, seed=0)
    return np.asarray(km.labels_, dtype=np.int64)


def subset_activations(pack, model, indices: np.ndarray, device: torch.device):
    batch = pack.gather(torch.from_numpy(indices.astype(np.int64)))
    with torch.no_grad():
        h = model.l1_concat(
            batch["white_idx"].to(device),
            batch["black_idx"].to(device),
            batch["stm_white"].to(device),
            batch["white_mask"].to(device),
            batch["black_mask"].to(device),
        ).detach().clone()
    return h, batch["target"].to(device).detach().clone(), batch["weight"].to(device).detach().clone()


def bucket_gradient(
    h: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    base_pieces: list[torch.Tensor],
    clip: float,
) -> torch.Tensor:
    leaves = [piece.detach().clone().requires_grad_(True) for piece in base_pieces]
    _mean, nll_sum = weighted_mean_ce(head_logits(h, leaves, clip), target, weight)
    grads = torch.autograd.grad(nll_sum, leaves)
    flat = torch.cat([grad.detach().reshape(-1).double() for grad in grads])
    w_sum = weight.detach().double().sum().clamp_min(1e-8)
    return (flat / w_sum).float()


def orthogonal_directions(grad: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    """Unit gradient and a unit Gaussian direction orthogonal to it."""
    g = grad.detach().double().cpu()
    norm = float(torch.linalg.vector_norm(g))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    degenerate = norm < 1e-12
    if degenerate:
        u = torch.randn(g.shape, generator=gen, dtype=torch.float64)
        u = u / torch.linalg.vector_norm(u)
    else:
        u = g / norm
    r = torch.randn(u.shape, generator=gen, dtype=torch.float64)
    r = r - torch.dot(r, u) * u
    v = r / torch.linalg.vector_norm(r)
    return u.float(), v.float(), norm, degenerate


def loss_grid(
    h: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    base_flat: torch.Tensor,
    shapes: list[tuple[int, ...]],
    u: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    grid: int,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    alpha = np.linspace(-1.0, 1.0, grid)
    beta = np.linspace(-1.0, 1.0, grid)
    values = np.empty((grid, grid), dtype=np.float64)
    with torch.no_grad():
        for bi, b in enumerate(beta):
            for ai, a in enumerate(alpha):
                step = (float(a) * scale) * u + (float(b) * scale) * v
                pieces = split_flat(base_flat + step.to(base_flat.device), shapes)
                mean, _sum = weighted_mean_ce(head_logits(h, pieces, clip), target, weight)
                values[bi, ai] = float(mean.detach().cpu())
    center = grid // 2
    origin = float(values[center, center])
    return alpha, beta, values - origin, origin


def plot_figure(
    panels: list[dict],
    *,
    n_buckets: int,
    n_micro: int,
    out_png: Path,
    note: str,
) -> None:
    stacked = np.concatenate([panel["delta_loss"].ravel() for panel in panels])
    # Same |ΔCE| must be the same colour. TwoSlopeNorm would spend the whole
    # blue half of bwr on the small negative percentile and the whole red half
    # on the larger positive one, so the colour bar would not match.
    span = max(
        abs(float(np.percentile(stacked, 2))),
        abs(float(np.percentile(stacked, 98))),
        1e-6,
    )
    norm = CenteredNorm(vcenter=0.0, halfrange=span)
    levels = np.linspace(-span, span, 7)[1:-1]

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    by_method: dict[str, list[dict]] = {method: [] for method in METHODS}
    for panel in panels:
        by_method[panel["method"]].append(panel)
    for method in METHODS:
        by_method[method].sort(key=lambda panel: (panel["grad_norm"], panel["bucket"]))

    fig, axes = plt.subplots(
        len(METHODS),
        n_buckets,
        figsize=(3.6 * n_buckets, 3.15 * len(METHODS)),
        squeeze=False,
        constrained_layout=True,
    )
    last = None
    for row, method in enumerate(METHODS):
        for col, panel in enumerate(by_method[method]):
            ax = axes[row, col]
            ax.set_title(f"bucket {panel['bucket']}")
            last = ax.imshow(
                panel["delta_loss"],
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap="bwr",
                interpolation="bilinear",
                norm=norm,
                aspect="auto",
            )
            if np.ptp(panel["delta_loss"]) > 1e-8:
                ax.contour(
                    panel["alpha"],
                    panel["beta"],
                    panel["delta_loss"],
                    levels=levels,
                    colors="#1a1a1a",
                    linewidths=0.6,
                    alpha=0.55,
                )
            ax.plot(0, 0, marker="+", color="black", markersize=9, markeredgewidth=1.6)
            ax.set_xticks((-1, 0, 1))
            ax.set_yticks((-1, 0, 1))
            ax.text(
                0.03,
                0.97,
                f"n={panel['n']}\n||g||={panel['grad_norm']:.2e}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.5, "edgecolor": "none"},
            )
    for col in range(n_buckets):
        axes[-1, col].set_xlabel("gradient direction")
    for row, method in enumerate(METHODS):
        axes[row, 0].set_ylabel(f"{ROW_TITLES[method]}\nrandom orthogonal direction")
    fig.colorbar(
        ScalarMappable(norm=norm, cmap="bwr") if last is None else last,
        ax=axes,
        fraction=0.03,
        pad=0.02,
        label="Δ cross-entropy",
    )
    fig.suptitle(
        f"Head loss landscape    B={n_buckets}    KITSUNE B'={n_micro}\n"
        f"each row ordered left to right by increasing ||g||\n{note}",
        fontsize=15,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    log(f"wrote {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, default=4)
    parser.add_argument("--micro-factor", type=int, default=8)
    parser.add_argument("--n-per-bucket", type=int, default=8192)
    parser.add_argument("--grid", type=int, default=31)
    parser.add_argument("--scale-fraction", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    parser.add_argument("--l1-labels", type=Path, default=None)
    parser.add_argument("--kitsune-labels", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def zoom_code(scale_fraction: float) -> int:
    return int(round(float(scale_fraction) * 10_000))


def figure_stem(args: argparse.Namespace) -> str:
    return (
        f"landscape_b{args.buckets}_m{args.micro_factor}"
        f"_n{args.n_per_bucket}_g{args.grid}_z{zoom_code(args.scale_fraction)}"
        f"_seed{args.random_seed}"
    )


def grid_path(out_dir: Path, args: argparse.Namespace, method: str, bucket: int) -> Path:
    """One file per panel. Random and L1 omit the micro-factor, which they do not use."""
    zoom = zoom_code(args.scale_fraction)
    shared = f"_n{args.n_per_bucket}_g{args.grid}_z{zoom}_seed{args.random_seed}"
    if method == "kitsune":
        folder = f"b{args.buckets}_m{args.micro_factor}{shared}"
    else:
        folder = f"b{args.buckets}{shared}"
    return out_dir / "grids" / folder / f"{method}_bucket{bucket}.npz"


def file_stamp(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def panel_identity(
    args: argparse.Namespace,
    ckpt: Path,
    pack_pt: Path,
    method: str,
    l1_file: Path | None = None,
    kitsune_file: Path | None = None,
) -> dict:
    """Identity of one loss grid. A stamp change means the file is stale."""
    ident: dict = {
        "buckets": int(args.buckets),
        "n_per_bucket": int(args.n_per_bucket),
        "grid": int(args.grid),
        "zoom": zoom_code(args.scale_fraction),
        "random_seed": int(args.random_seed),
        "checkpoint": file_stamp(ckpt),
        "pack": file_stamp(pack_pt),
    }
    if method == "l1":
        if l1_file is None:
            raise ValueError("L1 grids need the label file")
        ident["l1_labels"] = file_stamp(l1_file)
    elif method == "kitsune":
        if kitsune_file is None:
            raise ValueError("KITSUNE grids need the label file")
        ident["micro_factor"] = int(args.micro_factor)
        ident["kitsune_labels"] = file_stamp(kitsune_file)
    elif method != "random":
        raise ValueError(f"unknown method {method}")
    return ident


def _stamp_matches(saved: object, current: dict) -> bool:
    if not isinstance(saved, dict):
        return False
    return (
        saved.get("path") == current["path"]
        and int(saved.get("size", -1)) == int(current["size"])
        and int(saved.get("mtime_ns", -1)) == int(current["mtime_ns"])
    )


def identity_matches(meta: dict, ident: dict, method: str, bucket: int) -> bool:
    if meta.get("method") != method or int(meta.get("bucket", -1)) != int(bucket):
        return False
    for key in ("buckets", "n_per_bucket", "grid", "zoom", "random_seed"):
        if int(meta.get(key, -1)) != int(ident[key]):
            return False
    if method == "kitsune" and int(meta.get("micro_factor", -1)) != int(ident["micro_factor"]):
        return False
    for key in ("checkpoint", "pack", "l1_labels", "kitsune_labels"):
        if key not in ident:
            continue
        if not _stamp_matches(meta.get(key), ident[key]):
            return False
    return "scale" in meta and "theta_head_norm" in meta


def save_grid(path: Path, panel: dict, ident: dict, scale: float, theta_norm: float) -> None:
    meta = {
        **ident,
        "method": panel["method"],
        "bucket": int(panel["bucket"]),
        "scale": float(scale),
        "theta_head_norm": float(theta_norm),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        alpha=np.asarray(panel["alpha"]),
        beta=np.asarray(panel["beta"]),
        delta_loss=np.asarray(panel["delta_loss"]),
        grad_norm=np.float64(panel["grad_norm"]),
        ce_origin=np.float64(panel["ce_origin"]),
        slope=np.float64(panel["slope"]),
        n=np.int64(panel["n"]),
        degenerate=np.int64(int(panel["degenerate"])),
        direction_seed=np.int64(panel["direction_seed"]),
        meta=np.asarray(json.dumps(meta)),
    )


def load_grid(path: Path, ident: dict | None, method: str, bucket: int) -> dict | None:
    if ident is None:
        return None
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            meta = json.loads(data["meta"].item())
            if not identity_matches(meta, ident, method, bucket):
                return None
            return {
                "method": method,
                "bucket": int(bucket),
                "n": int(data["n"]),
                "alpha": np.asarray(data["alpha"]),
                "beta": np.asarray(data["beta"]),
                "delta_loss": np.asarray(data["delta_loss"]),
                "grad_norm": float(data["grad_norm"]),
                "ce_origin": float(data["ce_origin"]),
                "slope": float(data["slope"]),
                "degenerate": bool(int(data["degenerate"])),
                "direction_seed": int(data["direction_seed"]),
                "scale": float(meta["scale"]),
                "theta_head_norm": float(meta["theta_head_norm"]),
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def load_cached_panels(
    out_dir: Path,
    args: argparse.Namespace,
    idents: dict[str, dict | None],
) -> list[dict] | None:
    panels: list[dict] = []
    for method in METHODS:
        for bucket in range(int(args.buckets)):
            panel = load_grid(
                grid_path(out_dir, args, method, bucket),
                idents.get(method),
                method,
                bucket,
            )
            if panel is None:
                return None
            panels.append(panel)
    return panels


def write_outputs(
    panels: list[dict],
    *,
    n_buckets: int,
    n_micro: int,
    out_dir: Path,
    stem: str,
    note: str,
    ckpt: Path,
    pack_dir: Path,
    args: argparse.Namespace,
    scale: float,
    theta_norm: float,
) -> None:
    plot_figure(
        panels,
        n_buckets=n_buckets,
        n_micro=n_micro,
        out_png=out_dir / f"{stem}.png",
        note=note,
    )
    rows = [
        {
            "method": panel["method"],
            "bucket": panel["bucket"],
            "n": panel["n"],
            "ce_origin": panel["ce_origin"],
            "grad_norm": panel["grad_norm"],
            "slope": panel["slope"],
            "degenerate": int(panel["degenerate"]),
        }
        for panel in panels
    ]
    csv_path = out_dir / "slopes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "bucket", "n", "ce_origin", "grad_norm", "slope", "degenerate"),
        )
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {csv_path}")

    medians = {}
    for method in METHODS:
        values = [row["grad_norm"] for row in rows if row["method"] == method]
        medians[method] = float(np.median(values))
        log(f"median ||g|| {method}: {medians[method]:.6e}")
    order_ok = medians["random"] <= medians["l1"] < medians["kitsune"]
    log(f"expected median order random <= l1 < kitsune: {order_ok}")

    scale_path = out_dir / "scale.json"
    scale_path.write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt),
                "pack": str(pack_dir),
                "buckets": args.buckets,
                "micro_buckets": n_micro,
                "micro_factor": args.micro_factor,
                "scale": scale,
                "scale_fraction": args.scale_fraction,
                "theta_head_norm": theta_norm,
                "random_seed": args.random_seed,
                "direction_seeds": {
                    f"{panel['method']}:{panel['bucket']}": panel["direction_seed"] for panel in panels
                },
                "grid": args.grid,
                "n_per_bucket": args.n_per_bucket,
                "interpolation": "bilinear",
                "colormap": "bwr",
                "loss": "weighted mean soft cross-entropy (ce_and_mae)",
                "median_grad_norm": medians,
                "median_order_holds": order_ok,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"wrote {scale_path}")


def main() -> int:
    args = parse_args()
    if args.buckets < 2:
        raise SystemExit("--buckets must be >= 2")
    if args.micro_factor < 1:
        raise SystemExit("--micro-factor must be >= 1")
    if args.grid < 3 or args.grid % 2 == 0:
        raise SystemExit("--grid must be an odd integer >= 3 so the origin is a node")

    ckpt = resolve(args.checkpoint)
    pack_dir = resolve(args.pack_dir)
    out_dir = resolve(args.out_dir)
    n_micro = int(args.buckets) * int(args.micro_factor)
    l1_path = resolve(args.l1_labels) if args.l1_labels else (
        PROJECT_ROOT / "RESULTS" / "moe" / f"l1_clustered_k{args.buckets}" / "labels.npy"
    )
    pack_pt = pack_dir / "train_pack.pt"
    missing = [str(path) for path in (ckpt, pack_pt) if not path.is_file()]
    if missing:
        log("Missing checkpoint or train pack.")
        for path in missing:
            log(f"  missing {path}")
        return 1

    stem = figure_stem(args)
    label_dir = out_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    slice_ids_path = pack_dir / "slice_ids.npy"
    n_hint = None
    if slice_ids_path.is_file():
        try:
            n_hint = int(np.load(slice_ids_path, mmap_mode="r").shape[0])
        except (OSError, ValueError):
            n_hint = None
    kitsune_path = (
        resolve(args.kitsune_labels)
        if args.kitsune_labels
        else label_dir / f"kitsune_b{args.buckets}_micro{n_micro}.npy"
    )
    l1_length = None
    if l1_path.is_file():
        try:
            l1_length = int(np.load(l1_path, mmap_mode="r").shape[0])
        except (OSError, ValueError):
            l1_length = None
    l1_file = l1_path if n_hint is not None and l1_length == n_hint else (
        label_dir / f"l1_b{args.buckets}.npy"
    )
    idents: dict[str, dict | None] = {
        "random": panel_identity(args, ckpt, pack_pt, "random"),
        "l1": None,
        "kitsune": None,
    }
    if l1_file.is_file():
        idents["l1"] = panel_identity(args, ckpt, pack_pt, "l1", l1_file=l1_file)
    if kitsune_path.is_file():
        idents["kitsune"] = panel_identity(
            args, ckpt, pack_pt, "kitsune", kitsune_file=kitsune_path
        )
    if idents["l1"] is not None and idents["kitsune"] is not None:
        cached = load_cached_panels(out_dir, args, idents)
        if cached is not None:
            log(f"loaded {len(cached)} cached grids under {out_dir / 'grids'}")
            write_outputs(
                cached,
                n_buckets=args.buckets,
                n_micro=n_micro,
                out_dir=out_dir,
                stem=stem,
                note=ckpt.parent.name,
                ckpt=ckpt,
                pack_dir=pack_dir,
                args=args,
                scale=float(cached[0]["scale"]),
                theta_norm=float(cached[0]["theta_head_norm"]),
            )
            return 0
        if any(
            grid_path(out_dir, args, method, bucket).is_file()
            for method in METHODS
            for bucket in range(args.buckets)
        ):
            log("cached grids are incomplete or stale; computing the missing panels")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.cpu_count() or 1))
    log(f"device={device}")
    model = load_dual_hidden_checkpoint(ckpt, device=device)
    model.eval()
    model.requires_grad_(False)
    pack, _slice_ids, _local_rows = load_train_pack(pack_dir)
    n = len(pack)
    log(f"pack rows {n:,} | checkpoint {ckpt}")

    l1_saved = np.asarray(np.load(l1_path)) if l1_path.is_file() else None
    l1_matches = l1_saved is not None and l1_saved.shape == (n,)
    if l1_saved is not None and not l1_matches:
        log(f"L1 labels at {l1_path} have length {l1_saved.shape[0]}, pack has {n}. Refitting on this pack.")

    need_kitsune = args.kitsune_labels is None and not kitsune_path.is_file()
    need_l1 = not l1_matches
    l1_act = None
    if need_l1 or need_kitsune:
        log("building L1 activations once for the partitions that are not on disk")
        l1_act = build_l1(pack, model, device)

    if l1_matches:
        l1 = l1_saved
        log(f"loaded L1 labels {l1_path}")
    else:
        assert l1_act is not None
        log(f"k-means on L1, B={args.buckets}")
        l1_km = fit_minibatch_kmeans(l1_act, args.buckets, seed=0)
        l1 = np.asarray(l1_km.labels_, dtype=np.int16)
        l1_out = label_dir / f"l1_b{args.buckets}.npy"
        np.save(l1_out, l1)
        log(f"wrote {l1_out}")
    if int(np.min(l1)) < 0 or int(np.max(l1)) >= args.buckets:
        raise SystemExit(f"L1 labels are outside 0..{args.buckets - 1}")
    rng = np.random.default_rng(args.random_seed)
    random_labels = rng.integers(0, args.buckets, size=n, dtype=np.int16)
    random_path = label_dir / f"random_b{args.buckets}.npy"
    np.save(random_path, random_labels)
    log(f"wrote {random_path}")

    if args.kitsune_labels is not None or kitsune_path.is_file():
        kitsune = np.asarray(np.load(kitsune_path))
        log(f"loaded {kitsune_path}")
    else:
        assert l1_act is not None
        log(f"KITSUNE: k-means B'={n_micro} on the same L1 activations")
        km = fit_minibatch_kmeans(l1_act, n_micro, seed=0)
        micro = np.asarray(km.labels_, dtype=np.int64)
        del l1_act
        l1_act = None
        micro_path = label_dir / f"kitsune_micro_b{n_micro}.npy"
        np.save(micro_path, micro.astype(np.int16))
        prototypes = accumulate_micro_gradients(pack, model, micro, device, n_micro)
        mapping = stitch_kitsune(prototypes, args.buckets)
        kitsune = mapping[micro].astype(np.int16)
        np.save(kitsune_path, kitsune)
        np.save(label_dir / f"kitsune_map_b{args.buckets}_micro{n_micro}.npy", mapping.astype(np.int16))
        log(f"wrote {kitsune_path}")
    if l1_act is not None:
        del l1_act
    if kitsune.shape != (n,):
        raise SystemExit(f"KITSUNE labels length {kitsune.shape[0]} != pack rows {n}")

    labels = {
        "random": random_labels.astype(np.int64),
        "l1": l1.astype(np.int64),
        "kitsune": kitsune.astype(np.int64),
    }
    base_pieces = [tensor.detach().float().to(device) for tensor in head_tensors(model)]
    shapes = [tuple(piece.shape) for piece in base_pieces]
    base_flat = flatten(base_pieces).to(device)
    theta_norm = float(torch.linalg.vector_norm(base_flat.double()))
    scale = float(args.scale_fraction) * theta_norm
    clip = float(model.crelu_clip)
    log(f"||theta_head||={theta_norm:.6f}  s={scale:.6f}")

    l1_file = l1_path if l1_matches else label_dir / f"l1_b{args.buckets}.npy"
    idents = {
        "random": panel_identity(args, ckpt, pack_pt, "random"),
        "l1": panel_identity(args, ckpt, pack_pt, "l1", l1_file=l1_file),
        "kitsune": panel_identity(args, ckpt, pack_pt, "kitsune", kitsune_file=kitsune_path),
    }
    panels: list[dict] = []
    for method in METHODS:
        for bucket in range(args.buckets):
            path = grid_path(out_dir, args, method, bucket)
            cached = load_grid(path, idents[method], method, bucket)
            if cached is not None:
                panels.append(cached)
                log(f"loaded {method} bucket {bucket} from {path}")
                continue
            if path.is_file():
                log(f"stale grid {path.name}; recomputing {method} bucket {bucket}")
            pool = np.flatnonzero(labels[method] == bucket)
            if pool.size == 0:
                raise SystemExit(f"{method} bucket {bucket} is empty")
            chosen = sample_indices(pool, args.n_per_bucket, method, bucket)
            h, target, weight = subset_activations(pack, model, chosen, device)
            grad = bucket_gradient(h, target, weight, base_pieces, clip)
            seed = direction_seed(method, bucket)
            u, v, grad_norm, degenerate = orthogonal_directions(grad, seed)
            dot = float(torch.dot(u.double(), v.double()))
            if abs(dot) > 1e-5:
                raise RuntimeError(f"directions for {method} bucket {bucket} are not orthogonal (dot={dot})")
            alpha, beta, delta, origin = loss_grid(
                h, target, weight, base_flat, shapes, u.to(device), v.to(device), scale, args.grid, clip
            )
            panel = {
                "method": method,
                "bucket": bucket,
                "n": int(chosen.shape[0]),
                "alpha": alpha,
                "beta": beta,
                "delta_loss": delta,
                "grad_norm": grad_norm,
                "ce_origin": origin,
                "slope": scale * grad_norm,
                "degenerate": degenerate,
                "direction_seed": seed,
            }
            save_grid(path, panel, idents[method], scale, theta_norm)
            panels.append(panel)
            log(
                f"{method} bucket {bucket}: n={panel['n']}  CE0={origin:.4f}  "
                f"||g||={grad_norm:.3e}  slope={panel['slope']:.3e}"
            )
            log(f"wrote {path}")

    write_outputs(
        panels,
        n_buckets=args.buckets,
        n_micro=n_micro,
        out_dir=out_dir,
        stem=stem,
        note=ckpt.parent.name,
        ckpt=ckpt,
        pack_dir=pack_dir,
        args=args,
        scale=scale,
        theta_norm=theta_norm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
