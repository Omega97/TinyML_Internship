"""Skip-games + unique extract for Lichess dump batches."""

from __future__ import annotations

import importlib.util
import json
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
    assert dump.game_range_to_skip_max(0, 10) == (0, 10)
    assert dump.game_range_to_skip_max(1, 11) == (0, 10)
    assert dump.game_range_to_skip_max(1, 10) == (0, 9)
    assert dump.slice_json_name(dump.DEFAULT_DUMP, 0, 10) == (
        "fen_value_visits_lichess_db_standard_rated_2026-07_0-10.json"
    )
    assert dump.game_range_to_skip_max(1000, 2000) == (999, 1000)
    assert dump.game_range_to_skip_max(20_000_001, 21_000_001) == (20_000_000, 1_000_000)
    try:
        dump.game_range_to_skip_max(1, 1)
    except ValueError as exc:
        assert "exclusive" in str(exc).lower() or "m must be" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty [n, m)")
    path = dump.DEFAULT_DUMP
    assert dump.dump_month_id(path) == "lichess_db_standard_rated_2026-07"
    assert dump.slice_json_name(path, 1, 10) == (
        "fen_value_visits_lichess_db_standard_rated_2026-07_1-10.json"
    )
    assert dump.dropout_suffix(0.0) == ""
    assert dump.dropout_suffix(0.90) == "_d90"
    assert dump.slice_json_name(path, 0, 10, dropout=0.90) == (
        "fen_value_visits_lichess_db_standard_rated_2026-07_0-10_d90.json"
    )
    assert dump.slice_extract_name(path, 0, 10, dropout=0.90) == (
        "lichess_db_standard_rated_2026-07_0-10_d90_extract.parquet"
    )
    assert dump.max_draw_suffix(None) == ""
    assert dump.max_draw_suffix(0.30) == "_draw30"
    assert dump.slice_json_name(path, 0, 10, dropout=0.90, max_draw=0.30) == (
        "fen_value_visits_lichess_db_standard_rated_2026-07_0-10_d90_draw30.json"
    )
    assert dump.max_moves_suffix(0) == ""
    assert dump.max_moves_suffix(10) == "_m10"
    assert dump.slice_json_name(path, 100000, 105000, dropout=0.0, max_moves=10) == (
        "fen_value_visits_lichess_db_standard_rated_2026-07_100000-105000_m10.json"
    )


def test_write_json_max_draw_keeps_decisive(tmp_path: Path):
    dump = _load_script()
    labeled = tmp_path / "lab.parquet"
    import pandas as pd

    pd.DataFrame(
        [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "wdl_win": 0.31,
                "wdl_draw": 0.47,
                "wdl_loss": 0.22,
                "expected_reward": 0.09,
                "visits": 3,
            },
            {
                "fen": "8/8/8/8/8/8/4K3/4k3 w - - 0 1",
                "wdl_win": 0.80,
                "wdl_draw": 0.10,
                "wdl_loss": 0.10,
                "expected_reward": 0.70,
                "visits": 1,
            },
        ]
    ).to_parquet(labeled, index=False)
    out = tmp_path / "out.json"
    n = dump.write_json(labeled, out, max_draw=0.30)
    assert n == 1
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["wdl"][1] < 0.30


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


def test_max_moves_keeps_opening_only(tmp_path: Path):
    dump = _load_script()
    pgn_path = tmp_path / "opening.pgn"
    pgn_path.write_text(
        '[Event "g0"]\n[Site "x"]\n[White "w"]\n[Black "b"]\n[Result "1-0"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0\n\n",
        encoding="utf-8",
    )
    rows, stats = dump.collect_unique(
        pgn_path,
        skip_games=0,
        max_games=1,
        max_unique=0,
        progress_every=0,
        dropout=0.0,
        max_moves=1,
    )
    assert stats["max_moves"] == 1
    fens = [r["fen"] for r in rows]
    assert any(r["fen"].startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w") for r in rows)
    assert any("4P3" in fen for fen in fens)
    assert not any("5N2" in fen.split()[0] for fen in fens)


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


def test_encode_slice_folder_writes_npz(tmp_path: Path):
    import json

    spec = importlib.util.spec_from_file_location(
        "encode_slice_features", SCRIPT.parent / "encode_slice_features.py"
    )
    assert spec is not None and spec.loader is not None
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)

    folder = tmp_path / "fen_value_visits_tiny"
    folder.mkdir()
    rows = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "wdl": [0.312, 0.469, 0.219],
            "value": 0.093,
            "visits": 2,
        }
    ]
    (folder / "fen_value_visits_tiny.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    npz = enc.encode_slice_folder(folder, rebuild=True, progress=False)
    assert npz.is_file() and npz.name == "features.npz"


def test_dropout_drops_plies_and_is_seeded(tmp_path: Path):
    dump = _load_script()
    pgn_path = tmp_path / "tiny.pgn"
    pgn_path.write_text(
        _game("g0", "e4") + _game("g1", "d4") + _game("g2", "c4") + _game("g3", "Nf3"),
        encoding="utf-8",
    )
    kwargs = dict(
        skip_games=0,
        max_games=4,
        max_unique=0,
        progress_every=0,
    )
    full, full_stats = dump.collect_unique(pgn_path, **kwargs, dropout=0.0)
    dropped, drop_stats = dump.collect_unique(pgn_path, **kwargs, dropout=0.90, seed=0)
    again, _ = dump.collect_unique(pgn_path, **kwargs, dropout=0.90, seed=0)
    other, _ = dump.collect_unique(pgn_path, **kwargs, dropout=0.90, seed=1)
    assert drop_stats["plies_kept"] + drop_stats["plies_dropped"] == full_stats["plies_considered"]
    assert drop_stats["plies_kept"] < full_stats["plies_considered"]
    assert len(dropped) <= len(full)
    assert [r["fen"] for r in dropped] == [r["fen"] for r in again]
    assert [r["fen"] for r in dropped] != [r["fen"] for r in other] or len(dropped) < 2
