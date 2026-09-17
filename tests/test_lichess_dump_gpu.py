"""GPU dump script: batched labels + extract overlap hook (no Lc0 process)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parent.parent / "scripts" / "lichess_dump_to_fen_value_visits-gpu.py"


def _load_gpu():
    spec = importlib.util.spec_from_file_location("lichess_dump_to_fen_value_visits_gpu", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeTeacher:
    batch_size = 2

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.seen: list[str] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def evaluate_batch(self, fens: list[str]) -> list[tuple[float, float, float]]:
        self.seen.extend(fens)
        return [(0.5, 0.3, 0.2) for _ in fens]


def test_on_new_unique_fires_once_per_epd(tmp_path: Path):
    dump = _load_gpu()
    pgn_path = tmp_path / "tiny.pgn"
    pgn_path.write_text(
        '[Event "g0"]\n[Site "x"]\n[White "w"]\n[Black "b"]\n[Result "1-0"]\n\n'
        "1. e4 e5 1-0\n\n"
        '[Event "g1"]\n[Site "x"]\n[White "w"]\n[Black "b"]\n[Result "1-0"]\n\n'
        "1. e4 e5 1-0\n\n",
        encoding="utf-8",
    )
    seen: list[str] = []

    def on_new(_key: str, fen: str, _board) -> None:
        seen.append(fen)

    rows, stats = dump.collect_unique(
        pgn_path,
        skip_games=0,
        max_games=2,
        max_unique=0,
        progress_every=0,
        dropout=0.0,
        on_new_unique=on_new,
    )
    assert stats["unique"] == len(rows)
    assert len(seen) == len(rows)
    assert len(seen) == len(set(seen))


def test_label_extract_batches_through_teacher(tmp_path: Path):
    dump = _load_gpu()
    extract = tmp_path / "extract.parquet"
    labeled = tmp_path / "labeled.parquet"
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    pd.DataFrame(
        [
            {"fen": start, "visits": 2},
            {"fen": e4, "visits": 1},
        ]
    ).to_parquet(extract, index=False)
    fake = _FakeTeacher()
    n = dump.label_extract(
        extract,
        labeled,
        batch=20_000,
        network=Path("791556.pb.gz"),
        progress=False,
        nn_batch=2,
        teacher=fake,
    )
    assert n == 2
    assert fake.started is True
    assert fake.closed is False
    assert fake.seen == [start, e4]
    out = pd.read_parquet(labeled)
    assert list(out["wdl_win"]) == [0.5, 0.5]
    assert list(out["wdl_draw"]) == [0.3, 0.3]
    assert list(out["expected_reward"]) == [0.3, -0.3]


def test_labeled_frame_joins_visits():
    dump = _load_gpu()
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    df = dump.labeled_frame(
        [{"fen": start, "visits": 4}],
        {start: (0.4, 0.4, 0.2)},
        teacher_name="net.pb.gz",
    )
    assert int(df.iloc[0]["visits"]) == 4
    assert df.iloc[0]["teacher_network"] == "net.pb.gz"
    assert abs(float(df.iloc[0]["expected_reward"]) - 0.2) < 1e-9
