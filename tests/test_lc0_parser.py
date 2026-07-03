"""Lc0 parser smoke tests (uses one real chunk if present)."""

from pathlib import Path

import chess
import pytest

from tinymlinternship.data.lc0_parser import V6_RECORD_SIZE, iter_positions, read_chunk
from tinymlinternship.features.bucket import bucket_id

CHUNKS_DIR = Path(__file__).parent.parent / "data" / "raw" / "lc0" / "chunks"


def _smoke_chunk() -> Path | None:
    """Prefer a chunk with enough V6 records for parser validation."""
    chunks = sorted(CHUNKS_DIR.rglob("*.gz"), key=lambda p: p.stat().st_size, reverse=True)
    for path in chunks[:200]:
        raw = read_chunk(path)
        if len(raw) // V6_RECORD_SIZE >= 20:
            return path
    return chunks[0] if chunks else None


@pytest.mark.skipif(_smoke_chunk() is None, reason="Lc0 chunks not downloaded")
def test_chunk_decompresses_to_v6_records():
    chunk = _smoke_chunk()
    assert chunk is not None
    raw = read_chunk(chunk)
    assert len(raw) % V6_RECORD_SIZE == 0
    assert len(raw) // V6_RECORD_SIZE >= 1


@pytest.mark.skipif(_smoke_chunk() is None, reason="Lc0 chunks not downloaded")
def test_decode_positions_are_legal():
    chunk = _smoke_chunk()
    assert chunk is not None
    valid = 0
    for pos in iter_positions(chunk, limit=20):
        board = chess.Board(pos.fen)
        assert board.is_valid()
        assert 0 <= bucket_id(board) <= 7
        valid += 1
    assert valid >= 10