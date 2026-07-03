"""Parse Lc0 training chunks (V3–V6) into chess positions."""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import numpy as np

V6_STRUCT = struct.Struct("4si7432s832sBBBBBBBbfffffffffffffffIHHfI")
V6_RECORD_SIZE = V6_STRUCT.size  # 8356

@dataclass(frozen=True)
class Lc0Position:
    """One decoded training record (input_format 1 / classical)."""

    fen: str
    version: int
    input_format: int
    best_q: float
    root_q: float
    plies_left: float
    rule50: int
    visits: int
    played_idx: int
    best_idx: int
    invariance_info: int


def _iter_records(decompressed: bytes) -> Iterator[bytes]:
    version = struct.unpack_from("<i", decompressed, 0)[0]
    if version == 6:
        record_size = V6_RECORD_SIZE
    elif version == 5:
        record_size = 8308
    elif version == 4:
        record_size = 8292
    elif version == 3:
        record_size = 8276
    else:
        raise ValueError(f"Unsupported training-data version: {version}")
    for offset in range(0, len(decompressed) - record_size + 1, record_size):
        yield decompressed[offset : offset + record_size]


def _plane_bit(plane: int, square: int) -> bool:
    return bool((plane >> square) & 1)


def _planes_to_fen_classical(
    planes_u64: np.ndarray,
    *,
    us_ooo: int,
    us_oo: int,
    them_ooo: int,
    them_oo: int,
    stm: int,
    rule50: int,
) -> str:
    """Rebuild FEN from classical (input_format=1) bit planes."""
    board = chess.Board(None)
    board.clear_board()

    white_to_move = stm == 0
    stm_color = chess.WHITE if white_to_move else chess.BLACK
    opp_color = not stm_color
    for plane_base, color in ((0, stm_color), (6, opp_color)):
        # Lc0 classical planes 4/5 are king then queen (see encoder.cc).
        for offset, piece_type in enumerate(
            (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.KING, chess.QUEEN)
        ):
            plane = int(planes_u64[plane_base + offset])
            for square in chess.SQUARES:
                if _plane_bit(plane, square):
                    board.set_piece_at(square, chess.Piece(piece_type, color))

    fen_board = _board_to_fen_board(board)
    castling = _castling_fen(
        white_to_move=white_to_move,
        us_oo=bool(us_oo),
        us_ooo=bool(us_ooo),
        them_oo=bool(them_oo),
        them_ooo=bool(them_ooo),
    )
    fen = f"{fen_board} {'w' if white_to_move else 'b'} {castling} - {min(rule50, 255)} 1"
    return _sanitize_fen(fen)


def _sanitize_fen(fen: str) -> str:
    """Drop inconsistent castling flags (common in Lc0 classical records)."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return fen
    if board.is_valid():
        return board.fen()
    if board.status() == chess.STATUS_BAD_CASTLING_RIGHTS:
        parts = fen.split()
        parts[2] = "-"
        board = chess.Board(" ".join(parts))
        if board.is_valid():
            return board.fen()
    return fen


def _castling_fen(
    *,
    white_to_move: bool,
    us_oo: bool,
    us_ooo: bool,
    them_oo: bool,
    them_ooo: bool,
) -> str:
    """Map Lc0 us/them castling flags to FEN castling field."""
    if white_to_move:
        white_oo, white_ooo = us_oo, us_ooo
        black_oo, black_ooo = them_oo, them_ooo
    else:
        black_oo, black_ooo = us_oo, us_ooo
        white_oo, white_ooo = them_oo, them_ooo

    castling = ""
    if white_oo:
        castling += "K"
    if white_ooo:
        castling += "Q"
    if black_oo:
        castling += "k"
    if black_ooo:
        castling += "q"
    return castling or "-"


def _board_to_fen_board(board: chess.Board) -> str:
    rows: list[str] = []
    for rank in range(7, -1, -1):
        empty = 0
        row = ""
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is None:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                sym = piece.symbol()
                row += sym
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


def decode_v6_record(record: bytes) -> Lc0Position:
    """Decode one V6 record with classical input (format 1)."""
    if len(record) != V6_RECORD_SIZE:
        raise ValueError(f"Expected {V6_RECORD_SIZE} bytes, got {len(record)}")

    (
        ver_bytes,
        input_format,
        _probs,
        planes_bytes,
        us_ooo,
        us_oo,
        them_ooo,
        them_oo,
        stm,
        rule50,
        invariance_info,
        _dummy,
        root_q,
        best_q,
        _root_d,
        _best_d,
        _root_m,
        _best_m,
        plies_left,
        _result_q,
        _result_d,
        _played_q,
        _played_d,
        _played_m,
        _orig_q,
        _orig_d,
        _orig_m,
        visits,
        played_idx,
        best_idx,
        _policy_kld,
        _reserved,
    ) = V6_STRUCT.unpack(record)

    ver = struct.unpack("<i", ver_bytes)[0]
    if ver != 6:
        raise ValueError(f"Expected V6 record, got version {ver}")
    if input_format != 1:
        raise ValueError(f"Smoke parser supports input_format=1 only, got {input_format}")

    planes_u64 = np.frombuffer(planes_bytes, dtype=np.uint64)
    fen = _planes_to_fen_classical(
        planes_u64,
        us_ooo=us_ooo,
        us_oo=us_oo,
        them_ooo=them_ooo,
        them_oo=them_oo,
        stm=stm,
        rule50=rule50,
    )

    return Lc0Position(
        fen=fen,
        version=ver,
        input_format=input_format,
        best_q=best_q,
        root_q=root_q,
        plies_left=plies_left,
        rule50=rule50,
        visits=visits,
        played_idx=played_idx,
        best_idx=best_idx,
        invariance_info=invariance_info,
    )


def read_chunk(path: Path | str) -> bytes:
    """Decompress one training .gz chunk."""
    with gzip.open(path, "rb") as handle:
        return handle.read()


def iter_positions(path: Path | str, *, limit: int | None = None) -> Iterator[Lc0Position]:
    """Yield decoded positions from one chunk file."""
    data = read_chunk(path)
    count = 0
    for record in _iter_records(data):
        yield decode_v6_record(record)
        count += 1
        if limit is not None and count >= limit:
            return