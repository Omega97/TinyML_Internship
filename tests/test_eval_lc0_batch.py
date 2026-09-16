"""Batched Lc0 fenbatch client (no GPU required)."""

from __future__ import annotations

import pytest

from tinymlinternship.engine.eval_lc0_batch import (
    Lc0FenBatch,
    parse_fenbatch_line,
    wdl_from_q_d,
)


def test_wdl_from_q_d_startpos_like():
    w, d, l = wdl_from_q_d(0.1, 0.4)
    assert pytest.approx(w + d + l, abs=1e-12) == 1.0
    assert pytest.approx(w - l, abs=1e-12) == 0.1
    assert pytest.approx(d, abs=1e-12) == 0.4


def test_wdl_from_q_d_clamps_draw():
    w, d, l = wdl_from_q_d(0.0, 1.0)
    assert (w, d, l) == (0.0, 1.0, 0.0)


def test_parse_fenbatch_ok_line():
    assert parse_fenbatch_line("ok 0.312 0.469 0.219") == pytest.approx(
        (0.312, 0.469, 0.219)
    )


def test_parse_fenbatch_err_line():
    with pytest.raises(RuntimeError, match="Bad fen"):
        parse_fenbatch_line("err Bad fen string")


def test_parse_fenbatch_rejects_garbage():
    with pytest.raises(RuntimeError, match="unexpected"):
        parse_fenbatch_line("READY")


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        Lc0FenBatch(batch_size=0)
