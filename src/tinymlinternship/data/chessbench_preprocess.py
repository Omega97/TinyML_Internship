"""ChessBench state_value .bag → SARDINE NNUE training rows."""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
from apache_beam import coders

from tinymlinternship.features.bucket import bucket_id, piece_count
from tinymlinternship.features.encoder import encode_dual, validate_features

STATE_VALUE_CODER = coders.TupleCoder((
    coders.StrUtf8Coder(),
    coders.FloatCoder(),
))


def win_prob_to_expected_reward(win_prob: float) -> float:
    """
    Map ChessBench ``win_prob`` (STM POV, [0, 1]) to SARDINE expected reward.

    ChessBench stores Stockfish 16 win percentage from side to move; blueprint
    target is expected reward in [-1, +1] from STM POV.
    """
    reward = 2.0 * win_prob - 1.0
    return max(-1.0, min(1.0, reward))


@dataclass(frozen=True)
class ChessbenchRow:
    fen: str
    bucket_id: int
    piece_count: int
    white_features: list[int]
    black_features: list[int]
    expected_reward: float
    win_prob: float
    stm_white: bool


class BagFileReader:
    def __init__(self, path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            self._records = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            file_size = self._records.size()
        except ValueError:
            self._records = b""
            file_size = 0
        finally:
            os.close(fd)
        if 0 < file_size < 8:
            raise ValueError(f"bag file too small: {path}")
        if file_size:
            (index_start,) = struct.unpack("<Q", self._records[-8:])
        else:
            index_start = 0
        self._limits_start = index_start
        self._num_records = (file_size - index_start) // 8

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, index: int) -> bytes:
        if not 0 <= index < self._num_records:
            raise IndexError("bag index out of range")
        end = index * 8 + self._limits_start
        if index:
            rec_range = struct.unpack("<2q", self._records[end - 8 : end + 8])
        else:
            rec_range = (0, *struct.unpack("<q", self._records[end : end + 8]))
        return self._records[slice(*rec_range)]


def iter_state_value_records(path: Path) -> Iterator[tuple[str, float]]:
    reader = BagFileReader(path)
    for i in range(len(reader)):
        fen, win_prob = STATE_VALUE_CODER.decode(reader[i])
        yield fen, win_prob


def parse_chessbench_row(fen: str, win_prob: float) -> ChessbenchRow | None:
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    white_features, black_features = encode_dual(board)
    validate_features(white_features)
    validate_features(black_features)

    return ChessbenchRow(
        fen=fen,
        bucket_id=bucket_id(board),
        piece_count=piece_count(board),
        white_features=white_features,
        black_features=black_features,
        expected_reward=win_prob_to_expected_reward(win_prob),
        win_prob=win_prob,
        stm_white=board.turn == chess.WHITE,
    )


def row_to_dict(row: ChessbenchRow) -> dict:
    return {
        "fen": row.fen,
        "bucket_id": row.bucket_id,
        "piece_count": row.piece_count,
        "white_features": row.white_features,
        "black_features": row.black_features,
        "expected_reward": row.expected_reward,
        "win_prob": row.win_prob,
        "stm_white": row.stm_white,
    }