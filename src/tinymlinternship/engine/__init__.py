"""SARDINE chess engine (HCE + alpha-beta search)."""

from tinymlinternship.engine.eval_hce import evaluate_hce
from tinymlinternship.engine.perft import perft
from tinymlinternship.engine.search import EvalFn, SearchResult, search, search_best_move

ENGINE_VERSION = "0.3.0"

__all__ = [
    "ENGINE_VERSION",
    "EvalFn",
    "SearchResult",
    "evaluate_hce",
    "perft",
    "search",
    "search_best_move",
]