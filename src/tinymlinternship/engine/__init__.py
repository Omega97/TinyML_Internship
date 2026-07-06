"""SARDINE chess engine (HCE + alpha-beta search)."""

from tinymlinternship.engine.eval_hce import evaluate_hce
from tinymlinternship.engine.eval_lc0 import Lc0Teacher, evaluate_lc0_teacher
from tinymlinternship.engine.perft import perft
from tinymlinternship.engine.search import EvalFn, SearchResult, search, search_best_move

ENGINE_VERSION = "0.3.0"

__all__ = [
    "ENGINE_VERSION",
    "EvalFn",
    "Lc0Teacher",
    "SearchResult",
    "evaluate_hce",
    "evaluate_lc0_teacher",
    "perft",
    "search",
    "search_best_move",
]