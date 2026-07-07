"""Tests for ACPL gate helpers (no Stockfish required)."""

from __future__ import annotations

import chess
import chess.pgn

from tinymlinternship.bot_eval import (
    ELO_ACPL_INTERCEPT,
    ELO_ACPL_SLOPE,
    elo_estimate_from_acpl,
    elo_range_from_acpl,
    game_from_moves,
    merge_reports,
    report_for_side,
    split_report_by_side,
)
from tinymlinternship.bot_eval.acpl import AcplMoveResult, AcplReport, _report_from_moves


def test_elo_formula():
    assert elo_estimate_from_acpl(0) == ELO_ACPL_INTERCEPT
    assert elo_estimate_from_acpl(100) == ELO_ACPL_INTERCEPT - ELO_ACPL_SLOPE * 100


def test_elo_range_ordering():
    low, high = elo_range_from_acpl(50.0, 20.0, n_moves=40)
    assert low < elo_estimate_from_acpl(50.0) < high


def test_merge_reports_pools_moves():
    mk = lambda cpl: AcplMoveResult(
        ply=1,
        fen=chess.STARTING_FEN,
        move_uci="e2e4",
        side="white",
        best_move_uci="e2e4",
        best_cp=20,
        played_cp=20 - cpl,
        cpl=cpl,
        is_mate_best=False,
        is_mate_played=False,
    )
    r1 = _report_from_moves([mk(0), mk(100)])
    r2 = _report_from_moves([mk(50)])
    merged = merge_reports([r1, r2])
    assert merged.moves_analysed == 3
    assert merged.acpl == (0 + 100 + 50) / 3


def test_empty_report():
    rep = _report_from_moves([])
    assert rep.moves_analysed == 0
    assert rep.acpl == 0.0


def test_pgn_mainline_iter():
    from tinymlinternship.bot_eval.acpl import iter_games

    pgn = '[Event "t"]\n\n1. e4 e5 2. Nf3 *\n'
    games = list(iter_games(pgn))
    assert len(games) == 1
    assert len(list(games[0].mainline_moves())) == 3


def test_split_report_by_side():
    mk = lambda side, cpl: AcplMoveResult(
        ply=1,
        fen=chess.STARTING_FEN,
        move_uci="e2e4",
        side=side,
        best_move_uci="e2e4",
        best_cp=20,
        played_cp=20 - cpl,
        cpl=cpl,
        is_mate_best=False,
        is_mate_played=False,
    )
    full = _report_from_moves([mk("white", 0), mk("white", 100), mk("black", 50)])
    sides = split_report_by_side(full)
    assert sides["white"].moves_analysed == 2
    assert sides["white"].acpl == 50.0
    assert sides["black"].moves_analysed == 1
    assert sides["black"].acpl == 50.0
    assert report_for_side(full, "black").acpl == 50.0


def test_game_from_moves_uci():
    g = game_from_moves(["e2e4", "e7e5", "g1f3"], white="A", black="B")
    assert g.headers["White"] == "A"
    assert len(list(g.mainline_moves())) == 3