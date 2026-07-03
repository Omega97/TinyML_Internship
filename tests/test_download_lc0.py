"""Unit tests for Lc0 download shard selection (no network)."""

from tinymlinternship.data.lc0_shards import (
    DEFAULT_SHARDS,
    EXTRA_SHARDS,
    Lc0Shard,
    select_shards,
)


def test_default_subset_within_two_gb_budget():
    chosen = select_shards(DEFAULT_SHARDS, max_gb=2.0)
    names = [s.name for s in chosen]
    assert "training-run1--20240819-1917.tar" in names
    assert "training-run1--20250209-1017.tar" in names
    total = sum(s.size_bytes for s in chosen)
    assert total <= 2 * 1024**3
    assert total >= 1 * 1024**3


def test_max_gb_stops_after_first_shard():
    chosen = select_shards(DEFAULT_SHARDS, max_gb=0.8)
    assert len(chosen) == 1
    assert chosen[0].name == "training-run1--20240819-1917.tar"


def test_explicit_shard_names():
    name = "training-run1--20230505-0917.tar"
    chosen = select_shards(DEFAULT_SHARDS, max_gb=2.0, names=[name])
    assert len(chosen) == 1
    assert chosen[0].name == name


def test_shard_url_and_size_gb():
    shard = Lc0Shard("training-run1--20250209-1017.tar", 439_726_080)
    assert shard.url.endswith(shard.name)
    assert 0.4 < shard.size_gb < 0.5


def test_catalog_has_no_tiny_placeholder_tars():
    tiny = [s for s in (*DEFAULT_SHARDS, *EXTRA_SHARDS) if s.size_bytes <= 20_480]
    assert tiny == []