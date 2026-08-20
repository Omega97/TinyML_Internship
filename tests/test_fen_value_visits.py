"""Aggregation for the slim fen / value / visits table."""

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
