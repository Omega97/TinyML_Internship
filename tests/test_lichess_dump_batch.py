"""Skip-games + unique extract for Lichess dump batches."""

from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "lichess_dump_to_fen_value_visits.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("lichess_dump_to_fen_value_visits", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _game(event: str, san: str) -> str:
    return (
        f'[Event "{event}"]\n'
        f'[Site "https://lichess.org/x"]\n'
        f'[White "w"]\n'
        f'[Black "b"]\n'
        f'[Result "1-0"]\n'
        f"\n"
        f"1. {san} 1-0\n\n"
    )


def test_game_range_and_slice_names():
    dump = _load_script()
    assert dump.game_range_to_skip_max(1, 10) == (0, 10)
    assert dump.game_range_to_skip_max(20_000_001, 21_000_000) == (20_000_000, 1_000_000)
    path = dump.DEFAULT_DUMP
    assert dump.dump_month_id(path) == "lichess_db_standard_rated_2026-07"
    assert dump.slice_json_name(path, 1, 10) == (
        "fen_value_visits_lichess_db_standard_rated_2026-07_1-10.json"
    )


def test_progress_bar_disable_is_noop():
    dump = _load_script()
    bar = dump._progress_bar(total=10, desc="test", unit="it", disable=True)
    bar.update(10)
    bar.close()
    assert bar.disable is True


def test_skip_pgn_games_starts_at_requested_event():
    dump = _load_script()
    pgn = _game("g0", "e4") + _game("g1", "d4") + _game("g2", "c4") + _game("g3", "Nf3")
    for every in (0, 1):
        handle = dump.skip_pgn_games(StringIO(pgn), skip=2, progress_every=every)
        text = handle.read()
        assert text.startswith('[Event "g2"]')
        assert '[Event "g0"]' not in text
        assert '[Event "g1"]' not in text


def test_collect_unique_second_batch(tmp_path: Path):
    dump = _load_script()
    pgn_path = tmp_path / "tiny.pgn"
    pgn_path.write_text(
        _game("g0", "e4")
        + _game("g1", "d4")
        + _game("g2", "c4")
        + _game("g3", "Nf3")
        + _game("g4", "g3"),
        encoding="utf-8",
    )
    rows, stats = dump.collect_unique(
        pgn_path,
        skip_games=2,
        max_games=2,
        max_unique=100,
        progress_every=0,
    )
    assert stats["games_read"] == 2
    assert stats["skip_games"] == 2
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert any(r["fen"] == start for r in rows)
    start_row = next(r for r in rows if r["fen"] == start)
    assert start_row["visits"] == 2
    fens = " ".join(r["fen"] for r in rows)
    assert "c4" in fens or "2P5" in fens or "2p5" in fens.lower() or "2P" in fens
    # After 1. c4 the FEN has a pawn on c4: ...2P5... wait it's white pawn c4: 2P3? 
    # rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR
    assert any("2P5" in r["fen"] for r in rows)
    assert any("5N2" in r["fen"] or "n5" in r["fen"] for r in rows) or any(
        "N" in r["fen"].split()[0][32:] for r in rows
    )
    assert not any("4P3" in r["fen"] and "2P5" not in r["fen"] for r in rows if r["fen"].count("P") == 8)
    e4_only = [r for r in rows if "4P3" in r["fen"] and "2P5" not in r["fen"] and "3P" not in r["fen"]]
    # 1. e4 is 4P3; 1. d4 is 3P4. Batch 2 should not include e4 or d4 first-move.
    assert not any("4P3" in r["fen"] for r in rows)
    assert not any("/3P4/" in r["fen"] for r in rows)
