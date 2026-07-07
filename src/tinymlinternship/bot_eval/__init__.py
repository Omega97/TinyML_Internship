"""Bot strength evaluation (ACPL gate, Elo heuristics)."""

from tinymlinternship.bot_eval.acpl import (
    AcplMoveResult,
    AcplReport,
    ELO_ACPL_INTERCEPT,
    ELO_ACPL_SLOPE,
    analyse_game_acpl,
    elo_estimate_from_acpl,
    elo_range_from_acpl,
    game_from_moves,
    merge_reports,
    report_for_side,
    split_report_by_side,
)
from tinymlinternship.bot_eval.stockfish_path import resolve_stockfish

__all__ = [
    "AcplMoveResult",
    "AcplReport",
    "ELO_ACPL_INTERCEPT",
    "ELO_ACPL_SLOPE",
    "analyse_game_acpl",
    "elo_estimate_from_acpl",
    "elo_range_from_acpl",
    "game_from_moves",
    "merge_reports",
    "report_for_side",
    "resolve_stockfish",
    "split_report_by_side",
]