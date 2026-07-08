"""
ACPL-based bot evaluation (blueprint §Bot Evaluation Tool Selection — A1).

Stockfish analyses each move in a self-play (or imported) game; average centipawn
loss maps to a heuristic Elo via ``Elo ≈ 2855 - ACPL × 10``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import chess.engine
import chess.pgn

from collections.abc import Callable

if TYPE_CHECKING:
    from collections.abc import Iterator

ELO_ACPL_INTERCEPT = 2855
ELO_ACPL_SLOPE = 10
MATE_CP = 30_000


@dataclass(frozen=True)
class AcplMoveResult:
    ply: int
    fen: str
    move_uci: str
    side: str
    best_move_uci: str
    best_cp: int
    played_cp: int
    cpl: int
    is_mate_best: bool
    is_mate_played: bool


@dataclass(frozen=True)
class AcplReport:
    moves_analysed: int
    total_cpl: int
    acpl: float
    acpl_std: float
    move_results: tuple[AcplMoveResult, ...]
    elo_estimate: int
    elo_low: int
    elo_high: int

    @property
    def elo_range_str(self) -> str:
        return f"{self.elo_low}–{self.elo_high}"


def _score_cp(score: chess.engine.PovScore, *, pov: chess.Color) -> tuple[int, bool]:
    """Centipawns from ``pov``'s perspective; mate as ±MATE_CP."""
    rel = score.pov(pov)
    if rel.is_mate():
        mate = rel.mate()
        if mate is None:
            return 0, True
        return (MATE_CP if mate > 0 else -MATE_CP), True
    cp = rel.score(mate_score=MATE_CP)
    return (cp if cp is not None else 0), False


def _centipawn_loss(
    board: chess.Board,
    move: chess.Move,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
) -> AcplMoveResult:
    pov = board.turn
    info_best = engine.analyse(board, limit)
    pv = info_best.get("pv") or ()
    best_move = pv[0] if pv else move
    best_cp, mate_best = _score_cp(info_best["score"], pov=pov)

    if move == best_move:
        played_cp = best_cp
        cpl = 0
        mate_played = mate_best
    else:
        info_played = engine.analyse(board, limit, root_moves=[move])
        played_cp, mate_played = _score_cp(info_played["score"], pov=pov)
        cpl = max(0, best_cp - played_cp)

    return AcplMoveResult(
        ply=board.fullmove_number,
        fen=board.fen(),
        move_uci=move.uci(),
        side="white" if pov == chess.WHITE else "black",
        best_move_uci=best_move.uci(),
        best_cp=best_cp,
        played_cp=played_cp,
        cpl=cpl,
        is_mate_best=mate_best,
        is_mate_played=mate_played,
    )


def analyse_game_acpl(
    game: chess.pgn.Game,
    engine: chess.engine.SimpleEngine,
    *,
    limit: chess.engine.Limit,
    max_plies: int | None = None,
    on_ply: Callable[[int, int, chess.Move], None] | None = None,
) -> AcplReport:
    """Run Stockfish on every move in ``game`` mainline; return ACPL report."""
    moves = list(game.mainline_moves())
    if max_plies is not None:
        moves = moves[:max_plies]
    total = len(moves)

    board = game.board()
    results: list[AcplMoveResult] = []

    for plies, move in enumerate(moves, start=1):
        results.append(_centipawn_loss(board, move, engine, limit))
        board.push(move)
        if on_ply is not None:
            on_ply(plies, total, move)

    return _report_from_moves(results)


def _report_from_moves(results: list[AcplMoveResult]) -> AcplReport:
    if not results:
        return AcplReport(
            moves_analysed=0,
            total_cpl=0,
            acpl=0.0,
            acpl_std=0.0,
            move_results=(),
            elo_estimate=elo_estimate_from_acpl(0.0),
            elo_low=0,
            elo_high=0,
        )

    cpls = [r.cpl for r in results]
    n = len(cpls)
    total = sum(cpls)
    acpl = total / n
    if n > 1:
        var = sum((c - acpl) ** 2 for c in cpls) / (n - 1)
        acpl_std = math.sqrt(var)
    else:
        acpl_std = acpl

    elo_est = elo_estimate_from_acpl(acpl)
    elo_low, elo_high = elo_range_from_acpl(acpl, acpl_std, n_moves=n)

    return AcplReport(
        moves_analysed=n,
        total_cpl=total,
        acpl=acpl,
        acpl_std=acpl_std,
        move_results=tuple(results),
        elo_estimate=elo_est,
        elo_low=elo_low,
        elo_high=elo_high,
    )


def elo_estimate_from_acpl(acpl: float) -> int:
    raw = round(ELO_ACPL_INTERCEPT - ELO_ACPL_SLOPE * acpl)
    return max(400, min(3200, raw))


def elo_range_from_acpl(
    acpl: float,
    acpl_std: float,
    *,
    n_moves: int,
    z: float = 1.96,
) -> tuple[int, int]:
    """
    Heuristic Elo band (blueprint C2).

    Uses ACPL standard error: ``acpl ± z * acpl_std / sqrt(n)``, mapped through
    the same linear formula. Clamped to [400, 3200].
    """
    if n_moves <= 0:
        return (0, 0)
    se = acpl_std / math.sqrt(n_moves) if n_moves > 1 else acpl_std
    acpl_high = acpl + z * se
    acpl_low = max(0.0, acpl - z * se)
    elo_center = elo_estimate_from_acpl(acpl)
    elo_low = round(ELO_ACPL_INTERCEPT - ELO_ACPL_SLOPE * acpl_high)
    elo_high = round(ELO_ACPL_INTERCEPT - ELO_ACPL_SLOPE * acpl_low)
    margin = max(75, round(z * ELO_ACPL_SLOPE * se))
    elo_low = max(400, min(elo_center - margin, elo_low))
    elo_high = min(3200, max(elo_center + margin, elo_high))
    if elo_low > elo_high:
        elo_low, elo_high = elo_high, elo_low
    return elo_low, elo_high


def report_for_side(report: AcplReport, side: str) -> AcplReport:
    """ACPL / Elo for one color only (``white`` or ``black``)."""
    side = side.lower()
    if side not in ("white", "black"):
        raise ValueError(f"side must be 'white' or 'black', got {side!r}")
    moves = [r for r in report.move_results if r.side == side]
    return _report_from_moves(list(moves))


def split_report_by_side(report: AcplReport) -> dict[str, AcplReport]:
    """Per-color ACPL reports from a single analysed game."""
    return {
        "white": report_for_side(report, "white"),
        "black": report_for_side(report, "black"),
    }


def merge_reports(reports: list[AcplReport]) -> AcplReport:
    """Pool move-level CPL across multiple games."""
    all_moves: list[AcplMoveResult] = []
    for rep in reports:
        all_moves.extend(rep.move_results)
    return _report_from_moves(all_moves)


def game_from_moves(
    moves: list[str],
    *,
    fen: str | None = None,
    white: str = "?",
    black: str = "?",
) -> chess.pgn.Game:
    """Build a one-line PGN game from UCI move strings."""
    board = chess.Board(fen) if fen else chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["FEN"] = board.fen()
    game.headers["SetUp"] = "1"
    node = game
    for token in moves:
        move = board.parse_uci(token.strip())
        node = node.add_variation(move)
        board.push(move)
    outcome = board.outcome()
    game.headers["Result"] = outcome.result() if outcome else "*"
    return game


def iter_games(pgn_text: str) -> Iterator[chess.pgn.Game]:
    import io

    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        yield game