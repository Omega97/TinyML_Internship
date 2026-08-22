"""Aggregation for the slim fen / value / visits table."""

import importlib.util
import json
from pathlib import Path

from tinymlinternship.data.board_store import (
    add_teacher_value,
    bump_visits,
    fen_value_visits_source_filename,
    slim_fen_value_visits,
)

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
        {"fen": UNLABELED, "value": 0.0, "visits": 2},
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
    assert abs(start["value"] - (0.2 * 3 + 0.4 * 1) / 4) < 1e-9
    e4 = by_epd[join.epd_key(E4)]
    assert e4["visits"] == 5
    assert abs(e4["value"] - (-0.5 * 1 + -0.1 * 4) / 5) < 1e-9
    assert rows[0]["visits"] >= rows[1]["visits"] >= rows[2]["visits"]
