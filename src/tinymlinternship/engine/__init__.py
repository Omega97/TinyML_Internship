"""SARDINE chess engine (v0.1 bring-up: HCE + 1-ply search)."""

from tinymlinternship.engine.eval_hce import evaluate_hce
from tinymlinternship.engine.search import SearchResult, search_best_move

ENGINE_VERSION = "0.1.0"

__all__ = [
    "ENGINE_VERSION",
    "SearchResult",
    "evaluate_hce",
    "search_best_move",
]