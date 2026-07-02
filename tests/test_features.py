"""Tests for SARDINE step-1 feature index map and king mirroring."""

import chess

from tinymlinternship.features import (
    FEATURE_DIM,
    bucket_id,
    encode_dual,
    encode_perspective,
    is_pawn_rank_inactive,
    is_valid_index,
    king_slot_index,
    meta_base,
    mirror_for_perspective,
    needs_horizontal_mirror,
    piece_square_count,
    piece_square_index,
    validate_features,
)


def test_feature_dimension():
    assert FEATURE_DIM == 716
    assert piece_square_count() == 704
    assert meta_base() == 704


def test_all_indices_in_range():
    board = chess.Board()
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        idx = piece_square_index(piece.color, piece.piece_type, sq)
        if idx is not None:
            assert is_valid_index(idx)


def test_pawn_inactive_ranks():
    assert is_pawn_rank_inactive(0)
    assert is_pawn_rank_inactive(7)
    assert not is_pawn_rank_inactive(3)
    assert piece_square_index(chess.WHITE, chess.PAWN, chess.A1) is None
    assert piece_square_index(chess.WHITE, chess.PAWN, chess.E4) is not None


def test_king_slots_alias_under_mirror():
    """g1 and b1 (horizontal mirrors) map to the same king feature index."""
    g1 = chess.parse_square("g1")
    b1 = chess.parse_square("b1")
    idx_g = piece_square_index(chess.WHITE, chess.KING, g1)
    idx_b = piece_square_index(chess.WHITE, chess.KING, b1)
    assert idx_g == idx_b
    assert idx_g == king_slot_index(chess.WHITE, 0, 1)


def test_needs_mirror_when_king_on_right_half():
    board = chess.Board("4k3/8/8/8/8/8/8/K7 w - - 0 1")  # white Ka1, black Ke8
    assert not needs_horizontal_mirror(board, chess.WHITE)
    assert needs_horizontal_mirror(board, chess.BLACK)


def test_mirror_places_king_on_left_half():
    board = chess.Board("4k3/8/8/8/8/8/8/K7 w - - 0 1")
    mirrored = mirror_for_perspective(board, chess.BLACK)
    sq = mirrored.king(chess.BLACK)
    assert sq is not None
    assert chess.square_file(sq) < 4


def test_mirrored_board_same_king_index():
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")  # white Ke1
    raw_sq = board.king(chess.WHITE)
    mirrored = mirror_for_perspective(board, chess.WHITE)
    mir_sq = mirrored.king(chess.WHITE)
    assert piece_square_index(chess.WHITE, chess.KING, raw_sq) == piece_square_index(
        chess.WHITE, chess.KING, mir_sq
    )


def test_encode_startpos():
    features = encode_perspective(chess.Board(), chess.WHITE)
    validate_features(features)
    assert len(features) == 36  # 32 pieces + 4 castling rights


def test_encode_en_passant():
    fen = "8/8/4p3/3Pp2p/8/8/8/8 w - e6 0 1"
    features = encode_perspective(fen, chess.WHITE)
    validate_features(features)
    assert len(features) == 5  # 4 pieces + 1 ep file


def test_encode_dual_asymmetric():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    white_feat, black_feat = encode_dual(board)
    validate_features(white_feat)
    validate_features(black_feat)
    assert white_feat != black_feat


def _board_with_pieces(target: int, *, with_queen: bool) -> chess.Board:
    """Build a simple board with exactly ``target`` pieces (both kings always present)."""
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    count = 2
    if with_queen:
        board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
        count += 1
    for square in chess.SQUARES:
        if count >= target:
            break
        if board.piece_at(square) is not None:
            continue
        color = chess.WHITE if count % 2 == 0 else chess.BLACK
        board.set_piece_at(square, chess.Piece(chess.PAWN, color))
        count += 1
    assert len(board.piece_map()) == target
    return board


def test_bucket_startpos():
    assert bucket_id(chess.Board()) == 7


def test_bucket_endgame():
    assert bucket_id(_board_with_pieces(12, with_queen=False)) == 0
    assert bucket_id(_board_with_pieces(12, with_queen=True)) == 0


def test_bucket_boundaries():
    cases = [
        (13, False, 1),
        (13, True, 2),
        (21, False, 1),
        (21, True, 2),
        (22, False, 3),
        (22, True, 4),
        (27, False, 3),
        (27, True, 4),
        (28, False, 5),
        (28, True, 6),
        (31, False, 5),
        (31, True, 6),
    ]
    for piece_total, with_queen, expected in cases:
        board = _board_with_pieces(piece_total, with_queen=with_queen)
        assert bucket_id(board) == expected, (piece_total, with_queen, bucket_id(board))