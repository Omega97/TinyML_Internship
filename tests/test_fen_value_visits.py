"""Aggregation for the slim fen / value / visits table."""

import importlib.util
import json
from pathlib import Path

import pytest

from tinymlinternship.data.board_store import (
    add_teacher_value,
    bump_visits,
    fen_value_visits_slice_path,
    fen_value_visits_source_filename,
    slim_fen_value_visits,
)
from tinymlinternship.data.wdl import stm_value, value_to_wdl

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_LATER = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 3"
E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
UNLABELED = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"


def test_source_filename_slug():
    assert fen_value_visits_source_filename("lichess") == "fen_value_visits_lichess.parquet"
    assert fen_value_visits_source_filename("Lc0") == "fen_value_visits_lc0.parquet"
    assert fen_value_visits_source_filename("kaggle games") == "fen_value_visits_kaggle_games.parquet"
    assert fen_value_visits_source_filename("lc0_large_25k") == "fen_value_visits_lc0_large_25k.parquet"
    assert fen_value_visits_source_filename("lichess_kaggle_10k") == "fen_value_visits_lichess_kaggle_10k.parquet"
    nested = fen_value_visits_slice_path(Path("slices"), "fen_value_visits_lc0.json")
    assert nested == Path("slices") / "fen_value_visits_lc0" / "fen_value_visits_lc0.json"


def test_visits_use_epd_hash_and_skip_unlabeled():
    store: dict = {}
    bump_visits(store, START)
    bump_visits(store, START_LATER)  # same EPD as START
    bump_visits(store, E4)
    bump_visits(store, UNLABELED)

    add_teacher_value(store, START, 0.2)
    add_teacher_value(store, START_LATER, 0.4)  # mean 0.3
    add_teacher_value(store, E4, -0.5)

    rows = {r["fen"]: r for r in slim_fen_value_visits(store, labeled_only=True)}
    assert UNLABELED not in rows
    assert len(rows) == 2

    start_row = next(r for r in rows.values() if r["fen"].startswith("rnbqkbnr/pppppppp/8/8/8/8"))
    assert start_row["visits"] == 2
    assert abs(start_row["value"] - 0.3) < 1e-9
    assert len(start_row["wdl"]) == 3
    assert abs(sum(start_row["wdl"]) - 1.0) < 1e-6

    e4_row = next(r for r in rows.values() if "4P3" in r["fen"])
    assert e4_row["visits"] == 1
    assert e4_row["value"] == -0.5


def test_labeled_only_position_gets_visit_from_label_count():
    store: dict = {}
    add_teacher_value(store, START, 0.1)
    add_teacher_value(store, START, 0.3)
    rows = slim_fen_value_visits(store, labeled_only=True)
    assert len(rows) == 1
    assert rows[0]["visits"] == 2
    assert abs(rows[0]["value"] - 0.2) < 1e-9


def test_parse_wdl_cell_accepts_numpy_array():
    import numpy as np

    from tinymlinternship.data.wdl import parse_wdl_cell, wdl_from_row

    cell = np.array([0.312, 0.469, 0.219], dtype=np.float64)
    assert parse_wdl_cell(cell) == pytest.approx((0.312, 0.469, 0.219), abs=1e-9)
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "wdl": cell,
        "value": 0.093,
    }
    assert wdl_from_row(row) == pytest.approx((0.312, 0.469, 0.219), abs=1e-9)


def test_dumps_labeled_json_three_decimals():
    from tinymlinternship.data.wdl import dumps_labeled_json, labeled_payload, round_wdl

    w, d, l = round_wdl(0.312, 0.469, 0.219)
    assert (w, d, l) == (0.312, 0.469, 0.219)
    text = dumps_labeled_json(
        labeled_payload(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            (0.31200000000000006, 0.46900000000000003, 0.21900000000000003),
            10,
        )
    )
    assert "0.31200000000000006" not in text
    assert '"wdl": [\n    0.312,\n    0.469,\n    0.219\n  ]' in text
    assert '"value": 0.093' in text


def test_value_to_wdl_recovers_expected_reward():
    for v in (-1.0, -0.7, -0.2, 0.0, 0.401, 0.5, 1.0):
        w, d, l = value_to_wdl(v)
        assert w + d + l == pytest.approx(1.0, abs=1e-9)
        assert stm_value(w, d, l) == pytest.approx(v, abs=1e-6)
        assert min(w, d, l) >= -1e-12


JOIN_SCRIPT = Path(__file__).parent.parent / "scripts" / "join_fen_value_visits.py"


def _load_join():
    spec = importlib.util.spec_from_file_location("join_fen_value_visits", JOIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_epd_key_drops_clocks():
    join = _load_join()
    assert join.epd_key(START) == join.epd_key(START_LATER)
    assert join.epd_key(START) != join.epd_key(E4)


def test_discover_slices_prefers_parquet(tmp_path: Path):
    join = _load_join()
    (tmp_path / "fen_value_visits_a.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "fen_value_visits_a.parquet").write_bytes(b"not-a-real-parquet")
    (tmp_path / "fen_value_visits_b.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    found = join.discover_slices(tmp_path)
    names = [p.name for p in found]
    assert names == ["fen_value_visits_a.parquet", "fen_value_visits_b.json"]


def test_discover_slices_nested_folders(tmp_path: Path):
    join = _load_join()
    nested = tmp_path / "fen_value_visits_a"
    nested.mkdir()
    (nested / "fen_value_visits_a.json").write_text("[]\n", encoding="utf-8")
    (nested / "fen_value_visits_a.parquet").write_bytes(b"not-a-real-parquet")
    other = tmp_path / "fen_value_visits_b"
    other.mkdir()
    (other / "fen_value_visits_b.json").write_text("[]\n", encoding="utf-8")
    found = join.discover_slices(tmp_path)
    names = [p.name for p in found]
    assert names == ["fen_value_visits_a.parquet", "fen_value_visits_b.json"]
    assert found[0].parent.name == "fen_value_visits_a"


def test_join_sums_visits_weights_value_and_sorts(tmp_path: Path):
    join = _load_join()
    src = tmp_path / "slices"
    src.mkdir()
    a = [
        {"fen": START, "value": 0.2, "visits": 3},
        {"fen": E4, "value": -0.5, "visits": 1},
    ]
    b = [
        {"fen": START_LATER, "value": 0.4, "visits": 1},
        {"fen": E4, "value": -0.1, "visits": 4},
        {"fen": UNLABELED, "value": 0.123456, "visits": 2},
    ]
    (src / "fen_value_visits_a.json").write_text(json.dumps(a) + "\n", encoding="utf-8")
    (src / "fen_value_visits_b.json").write_text(json.dumps(b) + "\n", encoding="utf-8")
    assert join.main(["--input-dir", str(src), "--output", str(tmp_path / "fen_value_visits.parquet")]) == 0
    out = tmp_path / "fen_value_visits.parquet"
    js = tmp_path / "fen_value_visits.json"
    assert out.is_file() and js.is_file()
    rows = json.loads(js.read_text(encoding="utf-8"))
    assert [r["visits"] for r in rows] == sorted((r["visits"] for r in rows), reverse=True)
    by_epd = {join.epd_key(r["fen"]): r for r in rows}
    assert len(rows) == 3
    start = by_epd[join.epd_key(START)]
    assert start["visits"] == 4
    assert abs(start["value"] - round((0.2 * 3 + 0.4 * 1) / 4, 3)) < 5e-2
    e4 = by_epd[join.epd_key(E4)]
    assert e4["visits"] == 5
    assert abs(e4["value"] - round((-0.5 * 1 + -0.1 * 4) / 5, 3)) < 5e-2
    unlabeled = by_epd[join.epd_key(UNLABELED)]
    assert abs(unlabeled["value"] - 0.123) < 5e-3
    assert len(unlabeled["wdl"]) == 3
    assert abs(sum(unlabeled["wdl"]) - 1.0) < 1e-6
    assert rows[0]["visits"] >= rows[1]["visits"] >= rows[2]["visits"]


COUNT_SCRIPT = Path(__file__).parent.parent / "scripts" / "count_fen_value_visits.py"


def _load_count():
    spec = importlib.util.spec_from_file_location("count_fen_value_visits", COUNT_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_count_unique_and_total_rows(tmp_path: Path):
    count = _load_count()
    a_dir = tmp_path / "fen_value_visits_a"
    b_dir = tmp_path / "fen_value_visits_b"
    a_dir.mkdir()
    b_dir.mkdir()
    a = [
        {"fen": START, "value": 0.2, "visits": 3},
        {"fen": E4, "value": -0.5, "visits": 1},
    ]
    b = [
        {"fen": START_LATER, "value": 0.4, "visits": 1},
        {"fen": UNLABELED, "value": 0.1, "visits": 2},
    ]
    (a_dir / "fen_value_visits_a.json").write_text(json.dumps(a) + "\n", encoding="utf-8")
    (b_dir / "fen_value_visits_b.json").write_text(json.dumps(b) + "\n", encoding="utf-8")
    assert count.main(["--input-dir", str(tmp_path), "--quiet"]) == 0
    stats = count.count_slices(count.discover_slices(tmp_path))
    assert stats["n_slices"] == 2
    assert stats["total_rows"] == 4
    assert stats["unique_epds"] == 3  # START and START_LATER share an EPD
    assert stats["visits_sum"] == 7
    assert stats["cross_slice_overlap_rows"] == 1
