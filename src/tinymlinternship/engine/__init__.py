"""SARDINE chess engine (HCE + alpha-beta search)."""

from tinymlinternship.engine.eval_factory import EVAL_CHOICES, make_eval_fn
from tinymlinternship.engine.eval_hce import evaluate_hce
from tinymlinternship.engine.eval_lc0 import Lc0Teacher, evaluate_lc0_teacher
from tinymlinternship.engine.eval_nnue import NnueEvaluator, evaluate_nnue
from tinymlinternship.engine.perft import perft
from tinymlinternship.engine.search import EvalFn, SearchResult, search, search_best_move

ENGINE_VERSION = "0.3.0"

__all__ = [
    "ENGINE_VERSION",
    "EVAL_CHOICES",
    "EvalFn",
    "Lc0Teacher",
    "NnueEvaluator",
    "SearchResult",
    "evaluate_hce",
    "evaluate_lc0_teacher",
    "evaluate_nnue",
    "make_eval_fn",
    "perft",
    "search",
    "search_best_move",
]