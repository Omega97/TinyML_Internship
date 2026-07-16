"""Unit tests for Lc0 preprocessing helpers."""

from tinymlinternship.data.lc0_preprocess import (
    DEFAULT_MIN_PLY,
    FilterConfig,
    ParsedPosition,
    SampleConfig,
    passes_filter,
    stratified_sample,
)


def _pos(bucket: int, ply: int) -> ParsedPosition:
    return ParsedPosition(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        bucket=bucket,
        ply=ply,
        game_id="g:0",
        chunk_path="c.gz",
        piece_count=32,
        has_queen=True,
        stm_white=True,
        source="lc0",
        best_q=0.0,
        root_q=0.0,
        plies_left=100.0,
        visits=200,
        adjudicated=False,
    )


def test_default_min_ply_is_32_half_moves():
    assert DEFAULT_MIN_PLY == 32


def test_bucket7_relaxed_min_ply():
    cfg = FilterConfig(min_ply=32, bucket_min_ply={7: 8})
    assert passes_filter(_pos(7, 16), cfg)
    assert not passes_filter(_pos(3, 16), cfg)
    assert passes_filter(_pos(3, 32), cfg)


def test_stratified_sample_respects_bucket_cap():
    pool = [_pos(b, 40) for b in range(8) for _ in range(50)]
    train, val = stratified_sample(pool, SampleConfig(total_positions=800, val_fraction=0.1, seed=1))
    assert len(train) + len(val) <= 800
    train_buckets = {p.bucket for p in train}
    assert len(train_buckets) >= 4