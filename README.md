# SARDINE

**Small Artificial RAM-restricted Deep Intelligent Neural Engine**

Tiny-hardware chess bot. Spec: **[Goal.md](Goal.md)**. Status: **[PROJECT.md](PROJECT.md)**.

Do this in order. Do not skip ahead.

| Step | What |
|------|------|
| 1 | Unique `{fen, value, visits}` table from games + an Lc0 teacher |
| 2 | Train a dual-POV 2-hidden NNUE \(f_w\) (sparse 844 → L1 accumulator → L2 → tanh \(v\)) |
| 3 | Task-vector clusters, linear dispatcher on \(h\), freeze L1, fine-tune expert heads |
| 4 | Keep **Cfish** αβ; replace only `evaluate` / `nnue_evaluate` |
| 5 | ACPL and STS supporting; BayesElo match vanilla vs MoE is the ship gate |

Old pipeline (8-bucket / single-head Python engine, HCE product path, ICTP 2026-07) lives in **[LEGACY/](LEGACY/)**.

### Live tree

- `data/raw/` — conversion sources (Lc0, Lichess, Kaggle) for fen-value-visits JSON
- `data/processed/board_eval/fen_value_visits/` — Goal §1 JSON slices; joined table will live in `data/processed/board_eval/`
- `models/teacher/lc0/` — Lc0 binary + `791556.pb.gz`; stronger nets in `models/teacher/networks/`
- `src/cfish/` — Cfish search (Goal §4). Launch: `run-cfish.bat`
- `tools/stockfish/` — ACPL judge (Goal §5)

### Current step

**§1 Building the dataset.**
