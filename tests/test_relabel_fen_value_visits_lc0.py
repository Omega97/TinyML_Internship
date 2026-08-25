"""Helpers for re-labeling fen-value-visits with Lc0 go depth N."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tinymlinternship.engine.eval_lc0 import (
    go_command,
    mate_to_wdl_permille,
    wdl_from_uci_lines,
)

SCRIPT = Path(__file__).parent.parent / "scripts" / "relabel_fen_value_visits_lc0.py"


def _load():
    spec = importlib.util.spec_from_file_location("relabel_fen_value_visits_lc0", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_depth_zero_is_one_eval():
    relabel = _load()
    assert relabel.uci_go(depth=0, nodes=None) == "go nodes 1"
    assert relabel.uci_go(depth=0, nodes=8) == "go nodes 8"
    assert relabel.uci_go(depth=1, nodes=None) == "go depth 1"


def test_go_command_depth_and_nodes():
    assert go_command() == "go nodes 1"
    assert go_command(depth=1) == "go depth 1"
    assert go_command(depth=2, nodes=256) == "go depth 2 nodes 256"


def test_wdl_from_uci_prefers_last_wdl_over_mate():
    lines = [
        "info depth 1 nodes 1 score cp -138 wdl 16 324 660 pv h7h5",
        "info depth 2 nodes 18 score mate 2 wdl 1000 0 0 pv h7h5 g4h4 e7g6",
        "bestmove h7h5",
    ]
    assert wdl_from_uci_lines(lines) == (1000, 0, 0)


def test_wdl_from_mate_if_no_wdl():
    lines = ["info depth 2 score mate 2 pv h7h5 g4h4 e7g6", "bestmove h7h5"]
    assert wdl_from_uci_lines(lines) == (1000, 0, 0)
    assert mate_to_wdl_permille(-3) == (0, 0, 1000)


def test_default_output_suffix():
    relabel = _load()
    src = Path("data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_puzzles.json")
    assert relabel.default_output(src, depth=1, nodes=None).name == (
        "fen_value_visits_lichess_puzzles_depth1.json"
    )
    assert relabel.default_output(src, depth=2, nodes=256).name == (
        "fen_value_visits_lichess_puzzles_depth2_nodes256.json"
    )


def test_discover_targets_file_folder_and_parent(tmp_path: Path):
    relabel = _load()
    parent = tmp_path / "fen_value_visits"
    a = parent / "fen_value_visits_a"
    b = parent / "fen_value_visits_b"
    a.mkdir(parents=True)
    b.mkdir()
    a_json = a / "fen_value_visits_a.json"
    b_json = b / "fen_value_visits_b.json"
    a_json.write_text("[]\n", encoding="utf-8")
    b_json.write_text("[]\n", encoding="utf-8")
    (a / "features.meta.json").write_text("{}\n", encoding="utf-8")
    (a / "fen_value_visits_a_depth2.json").write_text("[]\n", encoding="utf-8")
    assert relabel.discover_targets(a_json) == [a_json.resolve()]
    assert relabel.discover_targets(a) == [a_json.resolve()]
    found = relabel.discover_targets(parent)
    assert found == [a_json.resolve(), b_json.resolve()]


def test_depth_defaults_to_one():
    relabel = _load()
    src = Path("slice.json")
    assert relabel.default_output(src, depth=1, nodes=None).name == "slice_depth1.json"


def test_slim_row_rounds_value_keeps_visits():
    relabel = _load()
    row = relabel.slim_row({"fen": "x", "value": 0.644, "visits": 3}, (0.0, 0.0, 1.0))
    assert row["fen"] == "x"
    assert row["visits"] == 3
    assert row["wdl"] == [0.0, 0.0, 1.0]
    assert row["value"] == -1.0


def test_progress_bar_disable_is_noop():
    relabel = _load()
    bar = relabel._progress_bar(total=10, desc="test", disable=True)
    bar.update(10)
    bar.close()
    assert bar.disable is True


def test_load_rows_recovers_missing_brace(tmp_path: Path):
    relabel = _load()
    broken = tmp_path / "test.json"
    broken.write_text(
        """[
  {
    "fen": "1B3k2/5P2/5K2/7p/P4P1P/8/8/2q5 w - - 0 48",
    "value": 0.9,
    "visits": 1
  },
    "fen": "1Q5R/5pk1/5rpp/8/8/4P1PK/4qPQP/8 b - - 0 41",
    "value": -0.9,
    "visits": 1
  }
]
""",
        encoding="utf-8",
    )
    rows, how = relabel.load_rows(broken)
    assert how == "recovered"
    assert len(rows) == 2
    assert rows[0]["fen"].startswith("1B3k2/")
    assert rows[1]["visits"] == 1
    assert rows[1]["value"] == -0.9


def test_load_rows_valid_json_list(tmp_path: Path):
    relabel = _load()
    path = tmp_path / "ok.json"
    path.write_text(
        json.dumps(
            [
                {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "visits": 4},
            ]
        ),
        encoding="utf-8",
    )
    rows, how = relabel.load_rows(path)
    assert how == "json"
    assert rows[0]["visits"] == 4
