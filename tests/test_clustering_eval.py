"""GOAL.md clustering metrics, piece-count buckets, and plot writers."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from tinymlinternship.features import (
    FEATURE_DIM,
    encode_dual,
    piece_count_bucket,
    piece_square_count,
)
from tinymlinternship.nnue.cluster import assign_to_centroids_cosine
from tinymlinternship.nnue.clustering_eval import (
    cosine_distance_matrix,
    fill_stm_board,
    overlap_scores,
    piece_count_labels,
    piece_counts_from_indices,
    plot_embedding,
    plot_size_panels,
    summarize_partition,
)


def test_piece_count_bucket_intervals():
    assert piece_count_bucket(32) == 0
    assert piece_count_bucket(28) == 1
    assert piece_count_bucket(29) == 1
    assert piece_count_bucket(30) == 2
    assert piece_count_bucket(31) == 2
    assert piece_count_bucket(27) == 3
    assert piece_count_bucket(24) == 3
    assert piece_count_bucket(23) == 4
    assert piece_count_bucket(20) == 4
    assert piece_count_bucket(19) == 5
    assert piece_count_bucket(16) == 5
    assert piece_count_bucket(15) == 6
    assert piece_count_bucket(10) == 6
    assert piece_count_bucket(9) == 7
    assert piece_count_bucket(2) == 7
    with pytest.raises(ValueError):
        piece_count_bucket(1)
    with pytest.raises(ValueError):
        piece_count_bucket(33)


def test_startpos_feature_count_is_32_pieces():
    white, _black = encode_dual(chess.Board())
    counts = piece_counts_from_indices(
        np.asarray([white], dtype=np.int16),
        np.asarray([len(white)]),
        limit=piece_square_count(),
    )
    assert int(counts[0]) == 32
    assert int(piece_count_labels(counts)[0]) == 0


def test_fill_stm_board_swaps_with_side_to_move():
    dest = np.zeros((2, 2 * FEATURE_DIM), dtype=np.float32)
    white_idx = np.array([[0, 2], [1, 0]], dtype=np.int64)
    black_idx = np.array([[4, 0], [5, 0]], dtype=np.int64)
    white_mask = np.array([[True, True], [True, False]])
    black_mask = np.array([[True, False], [True, False]])
    fill_stm_board(dest, white_idx, white_mask, black_idx, black_mask, np.array([True, False]))
    assert dest[0, 0] == 1.0 and dest[0, 2] == 1.0
    assert dest[0, FEATURE_DIM + 4] == 1.0
    assert dest[1, 5] == 1.0
    assert dest[1, FEATURE_DIM + 1] == 1.0
    assert dest.sum() == 5.0


def test_cosine_distance_of_opposite_centroids():
    dist = cosine_distance_matrix(np.array([[1.0, 0.0], [-1.0, 0.0]]))
    assert dist[0, 0] == pytest.approx(0.0)
    assert dist[0, 1] == pytest.approx(2.0)
    dist_nan = cosine_distance_matrix(np.array([[1.0, 0.0], [np.nan, np.nan], [-1.0, 0.0]]))
    assert dist_nan[0, 2] == pytest.approx(2.0)
    assert np.isnan(dist_nan[0, 1])


def test_cosine_assignment_picks_nearest_direction():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    x = np.array(
        [[10.0, 0.0], [0.0, 5.0], [1.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    labels = assign_to_centroids_cosine(x, centroids)
    assert labels.tolist() == [0, 1, 0, 1]


def test_cosine_assignment_ignores_magnitude():
    centroids = np.array([[3.0, 4.0]], dtype=np.float32)
    x = np.array([[6.0, 8.0], [0.3, 0.4]], dtype=np.float32)
    labels = assign_to_centroids_cosine(x, centroids)
    assert np.all(labels == 0)


def test_summarize_separated_blobs():
    rng = np.random.RandomState(0)
    left = rng.normal(loc=0.0, scale=0.05, size=(30, 3))
    right = rng.normal(loc=4.0, scale=0.05, size=(30, 3))
    x = np.vstack([left, right]).astype(np.float32)
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int16)
    centers = np.vstack([left.mean(axis=0), right.mean(axis=0)])
    summary = summarize_partition(x, labels, centers, silhouette_samples=60, seed=0)
    assert summary["sizes"] == [30, 30]
    assert summary["empty_clusters"] == 0
    assert summary["proportions"][0] == pytest.approx(0.5)
    assert summary["silhouette"] is not None and summary["silhouette"] > 0.9
    assert summary["cosine_distance_min"] > 0.5


def test_overlap_identical_labels():
    labels = np.array([0, 0, 1, 1, 2, 2])
    scores = overlap_scores(labels, labels)
    assert scores["ari"] == pytest.approx(1.0)
    assert scores["nmi"] == pytest.approx(1.0)


def test_plot_embedding_and_sizes(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.RandomState(0)
    coords = rng.normal(size=(40, 3)).astype(np.float32)
    labels = np.array([0] * 20 + [1] * 20, dtype=np.int16)
    path = plot_embedding(
        tmp_path / "pca.png",
        coords[:, :2],
        labels,
        title="pca",
        axis_names=("PC1", "PC2"),
    )
    path3 = plot_embedding(
        tmp_path / "pca3.png",
        coords,
        labels,
        title="pca3",
        axis_names=("PC1", "PC2", "PC3"),
    )
    bars = plot_size_panels(tmp_path / "sizes.png", [("k=2", [20, 20], ["a", "b"])])
    assert path.is_file() and path.stat().st_size > 0
    assert path3.is_file() and path3.stat().st_size > 0
    assert bars.is_file() and bars.stat().st_size > 0
