"""Tests for SARDINE step-1 feature index map and king mirroring."""

import chess
import pytest

from tinymlinternship.features import (
    FEATURE_DIM,
    bucket_id,
    castling_index,
    encode_dual,
    encode_perspective,
    is_pawn_rank_inactive,
    is_valid_index,
    king_enemy_index,
    king_self_index,
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
        for perspective in (chess.WHITE, chess.BLACK):
            idx = piece_square_index(
                piece.color, piece.piece_type, sq, perspective=perspective
            )
            if idx is not None:
                assert is_valid_index(idx)


def test_pawn_inactive_ranks():
    assert is_pawn_rank_inactive(0)
    assert is_pawn_rank_inactive(7)
    assert not is_pawn_rank_inactive(3)
    assert piece_square_index(chess.WHITE, chess.PAWN, chess.A1, perspective=chess.WHITE) is None
    assert (
        piece_square_index(chess.WHITE, chess.PAWN, chess.E4, perspective=chess.WHITE) is not None
    )


def test_perspective_king_aliases_under_mirror():
    """Perspective king: g1 and b1 map to the same self-king slot."""
    g1 = chess.parse_square("g1")
    b1 = chess.parse_square("b1")
    idx_g = piece_square_index(chess.WHITE, chess.KING, g1, perspective=chess.WHITE)
    idx_b = piece_square_index(chess.WHITE, chess.KING, b1, perspective=chess.WHITE)
    assert idx_g == idx_b
    assert idx_g == king_self_index(0, 1)


def test_enemy_king_keeps_full_file_resolution():
    """Enemy king: g8 and b8 are different slots (no horizontal fold)."""
    g8 = chess.parse_square("g8")
    b8 = chess.parse_square("b8")
    idx_g = piece_square_index(chess.BLACK, chess.KING, g8, perspective=chess.WHITE)
    idx_b = piece_square_index(chess.BLACK, chess.KING, b8, perspective=chess.WHITE)
    assert idx_g != idx_b
    assert idx_g == king_enemy_index(7, 6)
    assert idx_b == king_enemy_index(7, 1)


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


def test_mirrored_board_same_perspective_king_index():
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")  # white Ke1
    raw_sq = board.king(chess.WHITE)
    mirrored = mirror_for_perspective(board, chess.WHITE)
    mir_sq = mirrored.king(chess.WHITE)
    assert piece_square_index(chess.WHITE, chess.KING, raw_sq, perspective=chess.WHITE) == (
        piece_square_index(chess.WHITE, chess.KING, mir_sq, perspective=chess.WHITE)
    )


def test_encode_startpos():
    features = encode_perspective(chess.Board(), chess.WHITE)
    validate_features(features)
    assert len(features) == 36  # 32 pieces + 4 castling rights


def test_castling_labels_swap_when_horizontally_mirrored():
    """Startpos: white king on e1 triggers mirror — K/Q castling indices swap in view frame."""
    features = encode_perspective(chess.Board(), chess.WHITE)
    assert castling_index(chess.WHITE, is_kingside=False) in features  # was WK in base
    assert castling_index(chess.WHITE, is_kingside=True) in features  # was WQ in base


def test_castling_no_swap_when_mirror_not_needed():
    """Without horizontal mirror, base kingside maps to kingside index directly."""
    from tinymlinternship.features.encoder import _append_castling_features

    board = chess.Board()
    active: set[int] = set()
    _append_castling_features(board, active, horizontally_mirrored=False)
    assert castling_index(chess.WHITE, is_kingside=True) in active
    assert castling_index(chess.WHITE, is_kingside=False) in active


def test_encode_en_passant():
    fen = "8/8/4p3/3Pp2p/8/8/8/8 w - e6 0 1"
    features = encode_perspective(fen, chess.WHITE)
    validate_features(features)
    assert len(features) == 5  # 4 pieces + 1 ep file


def test_en_passant_file_follows_horizontal_flip():
    """EP file must be taken from the mirrored view, not the original board."""
    # White king on e1 triggers flip: e6 -> d6 in view (file 3, not 4).
    board = chess.Board("4k3/8/4p3/8/8/8/8/4K3 w - e6 0 1")
    features = encode_perspective(board, chess.WHITE)
    validate_features(features)
    assert 711 in features  # ep file d (3) in flipped view
    assert 712 not in features  # ep file e (4) if wrongly taken from original board


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


# --- Golden FEN snapshots (step-1 gate) ---
# Full active-index lists; any change here signals encoder or index-map regression.


_GOLDEN_STARTPOS_WHITE = [
    0, 1, 2, 3, 4, 5, 6, 7, 49, 54, 114, 117, 176, 183, 244, 344, 345, 346, 347, 348,
    349, 350, 351, 409, 414, 474, 477, 536, 543, 604, 611, 699, 704, 705, 706, 707,
]

_GOLDEN_EP_WHITE = [27, 332, 335, 340, 712]

_GOLDEN_EP_MIRROR_WHITE = [339, 611, 699, 711]

_GOLDEN_KA1_WHITE = [608, 700]

_GOLDEN_ENDGAME_P12 = [6, 309, 311, 611, 699]

_GOLDEN_MID_P13_QUEEN = [4, 6, 244, 309, 311, 611, 699]

_GOLDEN_MID_P22 = [
    0, 2, 4, 6, 12, 14, 305, 307, 309, 311, 315, 317, 319, 611, 699,
]

_GOLDEN_DUAL_AFTER_E4_WHITE = [
    0, 1, 2, 4, 5, 6, 7, 19, 49, 54, 114, 117, 176, 183, 244, 344, 345, 346, 347, 348,
    349, 350, 351, 409, 414, 474, 477, 536, 543, 604, 611, 699, 704, 705, 706, 707, 711,
]

_GOLDEN_DUAL_AFTER_E4_BLACK = [
    0, 1, 2, 3, 4, 5, 6, 7, 49, 54, 114, 117, 176, 183, 244, 331, 344, 345, 346, 348,
    349, 350, 351, 409, 414, 474, 477, 536, 543, 604, 611, 699, 704, 705, 706, 707, 711,
]

_GOLDEN_AFTER_E4_FEN = (
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
)


@pytest.mark.parametrize(
    "board_like,perspective,expected",
    [
        pytest.param(chess.Board(), chess.WHITE, _GOLDEN_STARTPOS_WHITE, id="startpos"),
        pytest.param(
            "8/8/4p3/3Pp2p/8/8/8/8 w - e6 0 1",
            chess.WHITE,
            _GOLDEN_EP_WHITE,
            id="en_passant",
        ),
        pytest.param(
            "4k3/8/4p3/8/8/8/8/4K3 w - e6 0 1",
            chess.WHITE,
            _GOLDEN_EP_MIRROR_WHITE,
            id="en_passant_mirror",
        ),
        pytest.param(
            "4k3/8/8/8/8/8/8/K7 w - - 0 1",
            chess.WHITE,
            _GOLDEN_KA1_WHITE,
            id="king_a1_no_mirror",
        ),
        pytest.param(
            lambda: _board_with_pieces(12, with_queen=False),
            chess.WHITE,
            _GOLDEN_ENDGAME_P12,
            id="endgame_p12",
        ),
        pytest.param(
            lambda: _board_with_pieces(13, with_queen=True),
            chess.WHITE,
            _GOLDEN_MID_P13_QUEEN,
            id="middlegame_p13_queen",
        ),
        pytest.param(
            lambda: _board_with_pieces(22, with_queen=False),
            chess.WHITE,
            _GOLDEN_MID_P22,
            id="middlegame_p22",
        ),
    ],
)
def test_golden_encode_perspective(board_like, perspective, expected):
    if callable(board_like):
        board_like = board_like()
    features = encode_perspective(board_like, perspective)
    validate_features(features)
    assert features == expected


def test_golden_encode_dual_asymmetric():
    white_feat, black_feat = encode_dual(_GOLDEN_AFTER_E4_FEN)
    validate_features(white_feat)
    validate_features(black_feat)
    assert white_feat == _GOLDEN_DUAL_AFTER_E4_WHITE
    assert black_feat == _GOLDEN_DUAL_AFTER_E4_BLACK
    assert white_feat != black_feat


@pytest.mark.parametrize(
    "board_like,expected_bucket",
    [
        pytest.param(chess.Board(), 7, id="startpos"),
        pytest.param("8/8/4p3/3Pp2p/8/8/8/8 w - e6 0 1", 0, id="endgame_ep"),
        pytest.param(lambda: _board_with_pieces(13, with_queen=True), 2, id="p13_queen"),
        pytest.param(lambda: _board_with_pieces(22, with_queen=False), 3, id="p22_no_queen"),
    ],
)
def test_golden_bucket_id(board_like, expected_bucket):
    if callable(board_like):
        board_like = board_like()
    assert bucket_id(board_like) == expected_bucket