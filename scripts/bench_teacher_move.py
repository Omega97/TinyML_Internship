#!/usr/bin/env python3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from tinymlinternship.engine import search, evaluate_lc0_teacher
from tinymlinternship.engine.eval_lc0 import get_lc0_teacher

get_lc0_teacher()
t0 = time.perf_counter()
r = search(chess.Board(), 1, eval_fn=evaluate_lc0_teacher, quiescence=False)
dt = time.perf_counter() - t0
print(f"move={r.move.uci()} score={r.score} nodes={r.nodes} time={dt:.1f}s")