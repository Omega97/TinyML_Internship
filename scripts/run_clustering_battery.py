#!/usr/bin/env python3
"""GOAL.md §1 clustering battery on 1e6 positions.

Representations, all on the same rows of ``moe_b3_1m``:
  board      STM-ordered binary features [x_stm ‖ x_opp], 1688-d
  l1         STM-ordered L1 activations [h_stm ‖ h_opp] of the reference NNUE
  gradients  the cached 48-d head-gradient projection
  piece      handcrafted piece-count buckets

Writes summary metrics and plot coordinates only. Full assignments are not saved.

  RESULTS/clustering/stats.csv
  RESULTS/clustering/overlap.csv
  RESULTS/clustering/summary.json
  RESULTS/clustering/plots/
  RESULTS/clustering/projections/embeddings.npz
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

from tinymlinternship.config.settings import NNUE_CHECKPOINTS_DIR, PROJECT_ROOT
from tinymlinternship.features import FEATURE_DIM, piece_square_count
from tinymlinternship.nnue.cluster import DBSCAN_FIT_CAP, fit_dbscan_for_b, fit_minibatch_kmeans
from tinymlinternship.nnue.clustering_eval import (
    DBSCAN_TARGET_K,
    KMEANS_KS,
    SILHOUETTE_SAMPLES,
    VIZ_SAMPLES,
    embed_for_plots,
    fill_stm_board,
    manifold_embed,
    overlap_scores,
    piece_bucket_names,
    piece_count_labels,
    piece_counts_from_indices,
    plot_cosine_panels,
    plot_embedding,
    plot_size_panels,
    reduce_for_dbscan,
    summarize_partition,
    write_overlap_csv,
    write_stats_csv,
    write_summary_json,
)
from tinymlinternship.nnue.dataset import _slice_n_rows
from tinymlinternship.nnue.moe import load_dual_hidden_checkpoint
from tinymlinternship.nnue.moe_data import load_train_pack

SEED = 0
PACK_DIR = PROJECT_ROOT / "data" / "processed" / "board_eval" / "moe" / "moe_b3_1m"
CHECKPOINT = NNUE_CHECKPOINTS_DIR / "dual_h128_H256_e200_bpe512_bs10000" / "best.pt"
OUT_DIR = PROJECT_ROOT / "RESULTS" / "clustering"
BOARD_CHUNK = 4_096
L1_CHUNK = 16_384


def log(message: str) -> None:
    print(message, flush=True)


def corpus_rows(meta: dict) -> int:
    total = 0
    missing = 0
    for folder in meta.get("folders", []):
        try:
            total += int(_slice_n_rows(Path(folder)))
        except (OSError, FileNotFoundError, ValueError) as exc:
            missing += 1
            log(f"  skip corpus folder {folder!r}: {exc}")
    if missing:
        log(f"  corpus scan skipped {missing} folders")
    return total


def build_board_and_counts(pack) -> tuple[np.ndarray, np.ndarray]:
    n = len(pack)
    board = np.zeros((n, 2 * FEATURE_DIM), dtype=np.float32)
    counts = np.empty(n, dtype=np.int16)
    limit = piece_square_count()
    started = time.perf_counter()
    for start in range(0, n, BOARD_CHUNK):
        end = min(start + BOARD_CHUNK, n)
        index = torch.arange(start, end, dtype=torch.long)
        batch = pack.gather(index)
        white_idx = batch["white_idx"].numpy()
        black_idx = batch["black_idx"].numpy()
        white_mask = batch["white_mask"].numpy()
        black_mask = batch["black_mask"].numpy()
        stm = batch["stm_white"].numpy()
        fill_stm_board(
            board[start:end],
            white_idx,
            white_mask,
            black_idx,
            black_mask,
            stm,
        )
        counts[start:end] = piece_counts_from_indices(
            white_idx, white_mask.sum(axis=1), limit=limit
        )
        done = end
        if done == n or (start // BOARD_CHUNK) % 25 == 0:
            log(f"  board {done:,}/{n:,}  ({time.perf_counter() - started:.0f}s)")
    return board, counts


def build_l1(pack, model, device: torch.device) -> np.ndarray:
    n = len(pack)
    hidden = int(model.hidden_dim)
    out = np.empty((n, 2 * hidden), dtype=np.float32)
    model.eval()
    started = time.perf_counter()
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
                log(f"  l1 {end:,}/{n:,}  ({time.perf_counter() - started:.0f}s)")
    return out


def _row(
    representation: str,
    algorithm: str,
    requested_k: int,
    feature_dim: int,
    summary: dict,
    extra: dict | None = None,
) -> dict:
    row = {
        "representation": representation,
        "algorithm": algorithm,
        "requested_k": int(requested_k),
        "n_clusters": int(summary["n_clusters"]),
        "feature_dim": int(feature_dim),
        "cluster_feature_dim": int(feature_dim),
        "n_rows": int(summary["n_rows"]),
        "sizes": summary["sizes"],
        "proportions": summary["proportions"],
        "inertia": summary["inertia"],
        "silhouette": summary["silhouette"],
        "silhouette_samples": summary["silhouette_samples"],
        "cosine_distance_mean": summary["cosine_distance_mean"],
        "cosine_distance_min": summary["cosine_distance_min"],
        "empty_clusters": summary["empty_clusters"],
        "n_noise": "",
        "dbscan_epsilon": "",
        "dbscan_eps": "",
        "dbscan_min_samples": "",
        "dbscan_fit_rows": "",
        "dbscan_pca_var": "",
        "cosine_distance": summary["cosine_distance"],
        "names": summary["names"],
    }
    if extra:
        row.update(extra)
    return row


def run_kmeans(name: str, x: np.ndarray) -> tuple[list[dict], dict[tuple, np.ndarray]]:
    rows: list[dict] = []
    labels: dict[tuple, np.ndarray] = {}
    for k in KMEANS_KS:
        started = time.perf_counter()
        model = fit_minibatch_kmeans(x, k, seed=SEED)
        lab = np.asarray(model.labels_, dtype=np.int16)
        # Native 1688-d / 256-d pairwise distances at 10k points dominate the run.
        # 2k is enough for a stable average and stays in the space k-means used.
        sil_n = SILHOUETTE_SAMPLES if int(x.shape[1]) <= 64 else 2_000
        summary = summarize_partition(
            x,
            lab,
            model.cluster_centers_,
            inertia=float(model.inertia_),
            silhouette_samples=sil_n,
            seed=SEED + 1,
        )
        rows.append(_row(name, "minibatch_kmeans", k, int(x.shape[1]), summary))
        labels[(name, "minibatch_kmeans", k)] = lab
        log(
            f"  kmeans {name} k={k}: inertia={summary['inertia']:.3g} "
            f"silhouette={summary['silhouette']} "
            f"empty={summary['empty_clusters']} ({time.perf_counter() - started:.0f}s)"
        )
    return rows, labels


def run_dbscan(name: str, x: np.ndarray) -> tuple[dict, np.ndarray, dict]:
    started = time.perf_counter()
    reduced, note = reduce_for_dbscan(x, seed=SEED, fit_cap=DBSCAN_FIT_CAP)
    labels, centroids, diag = fit_dbscan_for_b(reduced, DBSCAN_TARGET_K, seed=SEED)
    lab = np.asarray(labels, dtype=np.int16)
    summary = summarize_partition(
        reduced,
        lab,
        centroids,
        inertia=float(diag["inertia"]) if diag.get("inertia") is not None else None,
        silhouette_samples=SILHOUETTE_SAMPLES,
        seed=SEED + 1,
    )
    extra = {
        "cluster_feature_dim": int(note["cluster_feature_dim"]),
        "n_noise": int(diag.get("n_noise", 0)),
        "dbscan_epsilon": diag.get("dbscan_epsilon", ""),
        "dbscan_eps": diag.get("dbscan_eps", ""),
        "dbscan_min_samples": diag.get("dbscan_min_samples", ""),
        "dbscan_fit_rows": diag.get("dbscan_fit_rows", ""),
        "dbscan_pca_var": note.get("dbscan_pca_var", ""),
    }
    row = _row(name, "dbscan", DBSCAN_TARGET_K, int(x.shape[1]), summary, extra)
    log(
        f"  dbscan {name}: B={summary['n_clusters']} ε-quantile={extra['dbscan_epsilon']} "
        f"noise={extra['n_noise']} silhouette={summary['silhouette']} "
        f"({time.perf_counter() - started:.0f}s)"
    )
    trials = {
        "trials": diag.get("dbscan_trials", []),
        "chosen_epsilon": extra["dbscan_epsilon"],
        "n_clusters": int(summary["n_clusters"]),
        **note,
    }
    return row, lab, trials


def run_piece(counts: np.ndarray) -> tuple[dict, np.ndarray]:
    labels = piece_count_labels(counts)
    x = counts.astype(np.float32).reshape(-1, 1)
    from tinymlinternship.nnue.clustering_eval import centroids_from_labels

    centers = centroids_from_labels(x, labels, len(piece_bucket_names()))
    summary = summarize_partition(
        x,
        labels,
        centers,
        silhouette_samples=SILHOUETTE_SAMPLES,
        seed=SEED + 1,
        names=piece_bucket_names(),
    )
    row = _row("piece_count", "piece_count", len(piece_bucket_names()), 1, summary)
    log(
        f"  piece-count: sizes={summary['sizes']} silhouette={summary['silhouette']}"
    )
    return row, labels


def build_overlap(store: dict[tuple, np.ndarray]) -> list[dict]:
    reps = ("board", "l1", "gradients")
    rows: list[dict] = []

    def add(key_a, key_b) -> None:
        scores = overlap_scores(store[key_a], store[key_b])
        rows.append(
            {
                "representation_a": key_a[0],
                "algorithm_a": key_a[1],
                "k_a": int(key_a[2]),
                "representation_b": key_b[0],
                "algorithm_b": key_b[1],
                "k_b": int(key_b[2]),
                "ari": scores["ari"],
                "nmi": scores["nmi"],
            }
        )

    for k in KMEANS_KS:
        keys = [(rep, "minibatch_kmeans", k) for rep in reps]
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                add(left, right)
            add(left, ("piece_count", "piece_count", len(piece_bucket_names())))
    db_keys = [(rep, "dbscan", DBSCAN_TARGET_K) for rep in reps]
    for i, left in enumerate(db_keys):
        for right in db_keys[i + 1 :]:
            add(left, right)
        add(left, ("piece_count", "piece_count", len(piece_bucket_names())))
    return rows


def draw_plots(
    out_dir: Path,
    embeddings: dict[str, dict[str, np.ndarray]],
    variances: dict[str, np.ndarray],
    viz_labels: dict[str, np.ndarray],
    stat_rows: list[dict],
) -> None:
    plots = out_dir / "plots"
    pretty = {"board": "Board state", "l1": "L1 activations", "gradients": "Sample gradients"}
    for rep, bundle in embeddings.items():
        labels = viz_labels[rep]
        var = variances[rep]
        var2 = float(np.sum(var[:2]))
        var3 = float(np.sum(var[: min(3, var.size)]))
        specs = (
            ("pca_2d", bundle["pca"][:, :2], f"PCA 2D ({var2:.0%} var)", ("PC1", "PC2")),
            ("pca_3d", bundle["pca"][:, :3], f"PCA 3D ({var3:.0%} var)", ("PC1", "PC2", "PC3")),
            ("tsne_2d", bundle["tsne2"], "t-SNE 2D", ("t-SNE 1", "t-SNE 2")),
            ("tsne_3d", bundle["tsne3"], "t-SNE 3D", ("t-SNE 1", "t-SNE 2", "t-SNE 3")),
            ("umap_2d", bundle["umap2"], "UMAP 2D", ("UMAP 1", "UMAP 2")),
            ("umap_3d", bundle["umap3"], "UMAP 3D", ("UMAP 1", "UMAP 2", "UMAP 3")),
        )
        for stem, coords, kind, axes in specs:
            plot_embedding(
                plots / f"{stem}_{rep}.png",
                coords,
                labels,
                title=f"{pretty[rep]} · {kind} · k-means k=4 · {coords.shape[0]:,} pts",
                axis_names=axes,
            )
            log(f"  plot {stem}_{rep}")

    by_rep: dict[str, list] = {rep: [] for rep in pretty}
    for row in stat_rows:
        if row["algorithm"] == "minibatch_kmeans" and row["representation"] in by_rep:
            by_rep[row["representation"]].append(
                (f"k={row['requested_k']}", row["sizes"], None)
            )
    for rep, panels in by_rep.items():
        plot_size_panels(plots / f"sizes_kmeans_{rep}.png", panels)

    db_panels = []
    for row in stat_rows:
        if row["algorithm"] == "dbscan":
            db_panels.append(
                (
                    f"{row['representation']} B={row['n_clusters']}",
                    row["sizes"],
                    None,
                )
            )
    plot_size_panels(plots / "sizes_dbscan.png", db_panels)
    piece = next(row for row in stat_rows if row["algorithm"] == "piece_count")
    plot_size_panels(
        plots / "sizes_piece_count.png",
        [("piece count", piece["sizes"], piece["names"])],
    )
    cosine = []
    for rep in pretty:
        match = next(
            row
            for row in stat_rows
            if row["representation"] == rep
            and row["algorithm"] == "minibatch_kmeans"
            and int(row["requested_k"]) == 4
        )
        cosine.append((pretty[rep], np.asarray(match["cosine_distance"], dtype=np.float64)))
    plot_cosine_panels(plots / "centroid_cosine_k4.png", cosine)


def refresh_piece_count_artifacts(piece_lab: np.ndarray, piece_row: dict) -> None:
    """Rewrite the piece-count size bar and the 10k projection labels."""
    plots = OUT_DIR / "plots"
    plot_size_panels(
        plots / "sizes_piece_count.png",
        [("piece count", piece_row["sizes"], piece_row["names"])],
    )
    npz_path = OUT_DIR / "projections" / "embeddings.npz"
    if not npz_path.is_file():
        return
    payload = dict(np.load(npz_path))
    viz_idx = payload["viz_idx"]
    payload["piece_count"] = np.asarray(piece_lab)[viz_idx].astype(np.int16)
    np.savez_compressed(npz_path, **payload)


def main(*, skip_embeddings: bool = False) -> None:
    log(f"loading pack {PACK_DIR}")
    pack, slice_ids, local_rows = load_train_pack(PACK_DIR)
    gradients = np.asarray(np.load(PACK_DIR / "gradients.npy"), dtype=np.float32)
    meta = json.loads((PACK_DIR / "meta.json").read_text(encoding="utf-8"))
    n = len(pack)
    if gradients.shape[0] != n or slice_ids.shape[0] != n or local_rows.shape[0] != n:
        raise RuntimeError(
            f"row mismatch pack={n} gradients={gradients.shape[0]} "
            f"slice_ids={slice_ids.shape[0]}"
        )
    if int(meta["hidden_dim"]) != 128 or int(meta["reduce_dim"]) != gradients.shape[1]:
        raise RuntimeError(f"unexpected gradient meta {meta['hidden_dim']} / {meta['reduce_dim']}")
    log(f"rows {n:,}  gradients {gradients.shape}  corpus scan...")
    n_corpus = corpus_rows(meta)
    log(f"corpus rows {n_corpus:,}  (sample excludes test_fraction={meta.get('test_fraction')})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"L1 checkpoint {CHECKPOINT.name} on {device}")
    model = load_dual_hidden_checkpoint(CHECKPOINT, device=device)
    log("building board features and piece counts")
    board, counts = build_board_and_counts(pack)
    log("building L1 activations")
    l1 = build_l1(pack, model, device)
    del pack, model
    spaces = {"board": board, "l1": l1, "gradients": gradients}
    log(
        f"features ready  board {board.shape}  l1 {l1.shape}  "
        f"piece mean {float(counts.mean()):.2f}"
    )

    stat_rows: list[dict] = []
    store: dict[tuple, np.ndarray] = {}
    db_trials: dict[str, dict] = {}
    for name, matrix in spaces.items():
        log(f"k-means {name}")
        rows, labels = run_kmeans(name, matrix)
        stat_rows.extend(rows)
        store.update(labels)
        log(f"dbscan {name}")
        row, lab, trials = run_dbscan(name, matrix)
        stat_rows.append(row)
        store[(name, "dbscan", DBSCAN_TARGET_K)] = lab
        db_trials[name] = trials
    piece_row, piece_lab = run_piece(counts)
    stat_rows.append(piece_row)
    store[("piece_count", "piece_count", len(piece_bucket_names()))] = piece_lab

    overlap_rows = build_overlap(store)
    write_stats_csv(OUT_DIR / "stats.csv", stat_rows)
    write_overlap_csv(OUT_DIR / "overlap.csv", overlap_rows)
    summary = {
        "n_rows": n,
        "corpus_rows": n_corpus,
        "source": str(PACK_DIR),
        "checkpoint": str(CHECKPOINT),
        "hidden_dim": int(meta["hidden_dim"]),
        "hidden2_dim": int(meta["hidden2_dim"]),
        "test_fraction_excluded": meta.get("test_fraction"),
        "split_seed": meta.get("split_seed"),
        "subset_seed": meta.get("subset_seed"),
        "gradient_reduce_dim": int(meta["reduce_dim"]),
        "gradient_proj_seed": int(meta["proj_seed"]),
        "seed": SEED,
        "silhouette_samples": SILHOUETTE_SAMPLES,
        "viz_samples": VIZ_SAMPLES,
        "dbscan_target_k": DBSCAN_TARGET_K,
        "dbscan_trials": db_trials,
        "piece_count_histogram": {
            str(int(v)): int(c)
            for v, c in zip(*np.unique(counts, return_counts=True))
        },
        "note": (
            "1e6 rows are the seed-0 training-split subsample cached in moe_b3_1m "
            "(the 1% base-model test split is not in this sample). "
            "Board features are the STM-ordered concatenation of the two 844-d "
            "binary views. DBSCAN on board and L1 runs in a 48-d PCA of that "
            "representation because 8-NN density in the native width is not usable. "
            "Full cluster assignments are not stored. "
            "Piece-count inertia and silhouette are computed on the scalar piece count, "
            "so centroid cosine distance is 0: every bucket mean is a positive 1-d number. "
            "DBSCAN min_samples is the explorer schedule (80 on the 100k fit); "
            "the ε quantile is the grid point whose dense-cluster count is closest to 8."
        ),
        "experiments": [
            {key: value for key, value in row.items() if key != "names"}
            for row in stat_rows
        ],
    }
    write_summary_json(OUT_DIR / "summary.json", summary)
    log(f"wrote {OUT_DIR / 'stats.csv'} and overlap.csv")

    if skip_embeddings:
        refresh_piece_count_artifacts(piece_lab, piece_row)
        log("CLUSTERING_BATTERY_OK")
        return

    rng = np.random.RandomState(SEED)
    viz_idx = np.sort(rng.choice(n, size=min(VIZ_SAMPLES, n), replace=False))
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    variances: dict[str, np.ndarray] = {}
    viz_labels = {
        rep: store[(rep, "minibatch_kmeans", 4)][viz_idx] for rep in spaces
    }
    for name, matrix in spaces.items():
        log(f"embedding {name}")
        started = time.perf_counter()
        z, variance = embed_for_plots(matrix, viz_idx, seed=SEED)
        embeddings[name] = {"pca": z}
        variances[name] = variance
        log(
            f"  pca {name} var3={float(variance[:3].sum()):.3f} "
            f"({time.perf_counter() - started:.0f}s)"
        )
    # Manifolds only need the reduced plot sample.
    del board, l1, gradients, spaces
    for name, bundle in embeddings.items():
        z = bundle["pca"]
        for method, key, dim in (
            ("tsne", "tsne2", 2),
            ("tsne", "tsne3", 3),
            ("umap", "umap2", 2),
            ("umap", "umap3", 3),
        ):
            started = time.perf_counter()
            bundle[key] = manifold_embed(z, n_components=dim, seed=SEED, method=method)
            log(f"  {key} {name} ({time.perf_counter() - started:.0f}s)")

    proj_dir = OUT_DIR / "projections"
    proj_dir.mkdir(parents=True, exist_ok=True)
    payload = {"viz_idx": viz_idx.astype(np.int64)}
    for name, bundle in embeddings.items():
        for key, value in bundle.items():
            payload[f"{name}_{key}"] = value
        payload[f"{name}_kmeans4"] = viz_labels[name].astype(np.int16)
    payload["piece_count"] = piece_lab[viz_idx].astype(np.int16)
    np.savez_compressed(proj_dir / "embeddings.npz", **payload)
    log("drawing plots")
    draw_plots(OUT_DIR, embeddings, variances, viz_labels, stat_rows)
    log("CLUSTERING_BATTERY_OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip PCA/t-SNE/UMAP and only refresh tables plus sizes_piece_count.png",
    )
    args = parser.parse_args()
    main(skip_embeddings=bool(args.skip_embeddings))
