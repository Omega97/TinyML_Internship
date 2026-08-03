# Locked decisions: which file to follow

_Session 2026-08-03. Choices: `A1, B1, C1, D1, E1, F3, G1, H1, I1, J1, K1, L1`._

When docs disagree, use this table. Do not re-open options without an explicit new decision pass.

| ID | Topic | Follow | Do not follow / ignore |
| -- | ----- | ------ | ---------------------- |
| **A1** | Product pipeline status (Cfish → data → train → hardware) | **[PROJECT.md](PROJECT.md)** §Progress Overview | TODOs as sole status; Goal as checklist |
| **B1** | Architecture & device target (844, NNUE shape, TT, search v1, memory) | **[NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md)** §Design Decisions | PROJECT change tables alone (use them only for Cfish deltas) |
| **C1** | Train label contract | **[ASSETS.md](ASSETS.md)** (+ blueprint §Training data) | Any “Stockfish centipawns as train target” wording |
| **D1** | Playing strength / game review | Stockfish **ACPL** stack: `scripts/eval_bot_acpl.py`, `bot_eval/` | Cfish as judge; match Elo only (later ship gate) |
| **E1** | Production data sources | **ASSETS** ideal set (Lichess primary + Lc0 supplement) | Kaggle / ChessBench as production train |
| **F3** | Bucket / multi-head routing | **Single expert head** until §D ablation finishes | Shipping multi-bucket MoE before D; silent 8→4 migrate |
| **G1** | PC bring-up language | **Python** PC (`tinymlinternship`) → pure **C** on Wio | Blueprint “C++ on PC first” (stale) |
| **H1** | Paths & artifacts | **PROJECT** + **ASSETS** | Old NOTES path pins that contradict them |
| **I1** | Stockfish binary | **PATH / `STOCKFISH_PATH` / `--stockfish`** only | Required in-repo `models/teacher/stockfish/` |
| **J1** | Thesis / task vectors | **Blueprint §Later** + [NOTES/Thesis.md](NOTES/Thesis.md) **after** Elo path | Thesis on critical path before hardware |
| **K1** | PROJECT title vs italic | Prefer **italic + ASSETS + blueprint design** | Bare checklist titles when they conflict |
| **L1** | Cfish inventory & fork deltas | **PROJECT** §Cfish Features + §must change | NOTES/Cfish.md as fork plan authority |

## F3 note (important)

**Decision:** no multi-expert routing for the production student until bucket ablation (**TODOs §D** / blueprint §D) locks a scheme. Train and document a **single** output head: dual-POV L1 → concat `2W` → one `2W → 1` head.

- `bucket_id` / piece-count metadata may still be **stored** for analysis and future ablation.
- Legacy pilots with **8 heads** (`pilot_W128_844`, `smoke_prod_W128_844`, `features/bucket.py` router) are **historical / experimental**, not the locked v1 production shape.
- Ablation §D compares candidate partitions (including single-head baseline, 4 piece-count, 8 queen-split, …) before any multi-head ship.

## Quick roles

| Doc | Role after lock |
| --- | --------------- |
| **PROJECT.md** | Sole **progress** map + Cfish inventory |
| **Blueprint** | **Architecture** / device design |
| **ASSETS.md** | **Labels + data** contract + paths |
| **TODOs.md** | Optional eng micro-gates (not product status) |
| **Goal.md** | Short mission only |
| **ai-feed.md** | This authority table for agents |

#core
