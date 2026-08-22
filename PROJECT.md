# Project: SARDINE

**SARDINE** — *Small Artificial RAM-restricted Deep Intelligent Neural Engine*

Tiny-hardware chess bot. **Spec:** [Goal.md](Goal.md). **This file** is the current repo status (not the old 8-bucket / Python-engine plan).

| Doc                        | Role                                 |
| -------------------------- | ------------------------------------ |
| [Goal.md](Goal.md)         | What to build (five steps)           |
| **PROJECT.md** (this file) | What is on disk now                  |
| [README.md](README.md)     | Short pointer + live tree            |
| [ai-feed.md](ai-feed.md)   | Cleanup log (what went to `LEGACY/`) |
| `LEGACY/`                  | Previous pipeline (gitignored)       |

---

## Current step

**Goal §1 — dataset.** Train / MoE / Cfish-eval swap are not started on the live path.

---

## Progress vs Goal.md

### 1 — Building the dataset

- [x] Downloads on disk (not yet \(\gtrsim 10^6\) unique labeled positions)
	- Lc0 chunks + FEN extract: `data/raw/lc0/` (~54 866 `.gz`, `positions.parquet`)
	- Lichess smoke PGN + FENs: `data/raw/lichess/` (`lichess_smoke50.pgn`, `positions.parquet`)
	- Lichess monthly dump (compressed, not converted): `data/raw/lichess/dumps/lichess_db_standard_rated_2026-07.pgn.zst` (27.06 GiB, SHA256 verified)
	- Kaggle games + FENs: `data/raw/kaggle/` (`games.csv`, `kaggle_games_positions.parquet`)
	- Lichess puzzle FENs: `data/raw/lichess/puzzles_sample.parquet` (4k) + `puzzles_sample_16k.parquet` (16k, teacher not run yet)
- [x] Teacher binary + nets
	- `models/teacher/lc0/lc0.exe` (v0.32.1)
	- Labels so far: **`791556.pb.gz`** (White POV \(\hat v = (W-L)/1000\))
	- Stronger nets on disk, unused for these tables: T1-256, BT4 under `models/teacher/networks/`
- [~] `{fen, value, visits}` JSON slices exist; **joined table not written yet** (see Dataset layout)

Labeled teacher parquets: `data/processed/labeled/` (`train`/`val` mini merge, `lc0_*`, `lichess_*`, `lichess_kaggle_10k` + **`lichess_kaggle_40k`**, `lc0_large_25k` + **`lc0_large_40k`**).

Extracts still have unlabeled unique EPDs (order \(10^5\): leftover Kaggle + leftover Lc0 `positions.parquet`). Goal \(\gtrsim 10^6\) unique labeled positions is **open**.

### 2 — Train dual-POV 2-hidden NNUE

- [ ] No live `DualHiddenNNUE` / `train_nnue.py`. Encoder 844 lives only in `LEGACY/`.

### 3 — Task vectors / dispatcher / expert fine-tune

- [ ] Not started.

### 4 — Inference (Cfish αβ + student eval)

- [x] Cfish search on disk: `src/cfish/` · `run-cfish.bat` · stock `nn-62ef826d1a6d.nnue`
- [ ] Student `evaluate` / `nnue_evaluate` hook not wired

### 5 — Evaluation

- [ ] ACPL / STS / BayesElo match not run on the Goal student
- Stockfish judge is local: `tools/stockfish/`

---

## Dataset layout (Goal §1)

**Per-source JSON (and parquet twins) live in**

`data/processed/board_eval/fen_value_visits/`

Each file is a list of `{fen, value, visits}` objects (unique EPD; `visits` = extract multiplicity). Slices on disk:

| File | Approx. rows |
|------|-------------:|
| `fen_value_visits_lc0.json` | 269 062 |
| `fen_value_visits_lichess.json` | 2 131 |
| `fen_value_visits_lichess_kaggle_10k.json` | 10 000 |
| `fen_value_visits_lichess_puzzles.json` | 4 000 |
| `fen_value_visits_lichess_db_standard_rated_2026-07_1-1000.json` | 62 040 |
| `fen_value_visits_lichess_db_standard_rated_2026-07_1000-2000.json` | 62 518 |
| `fen_value_visits_lichess_db_standard_rated_2026-07_2000-4000.json` | 121 770 |
| `fen_value_visits_lichess_db_standard_rated_2026-07_4000-6000.json` | 122 185 |
| `fen_value_visits_lichess_db_standard_rated_2026-07_6000-8000.json` | 122 729 |
| `fen_value_visits_lichess_db_standard_rated_2026-07_8000-10000.json` | 122 967 |

`lichess_kaggle_40k` is labeled (`data/processed/labeled/lichess_kaggle_40k.parquet`) but **not** exported to JSON yet. Puzzle batch `puzzles_sample_16k` is extract-only (no teacher \(\hat v\)).

**The joined table lives in the parent folder**

`data/processed/board_eval/fen_value_visits.parquet` (+ `.json` twin)

Join: `scripts/join_fen_value_visits.py` (sum visits on the same EPD; visit-weighted value; sort by visits descending). Current join: **882 730** unique EPDs, visits sum 1 022 530.

### Dataset Status

**Raw data folder**: `data\raw`
**Processed data folder**: `data\processed\board_eval`

| **Name**                   | **Rows**                          | **Description**                                                                                       | **Downloaded** | **Converted** |
| -------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------- | ------------- |
| **Lc0 training tars**      | 269 062 unique in JSON | Two run1 shards; 250k filter-pass extract from 7 835/54 866 chunks, all unique EPDs labeled (`fen_value_visits_lc0.json`). Remaining chunks unsampled. | ✅              | ✅             |
| **Lichess smoke PGN**      | 2,371 (2,131 unique)              | 50-game smoke; all unique FENs in fen_value_visits_lichess.json                                       | ✅              | ✅             |
| **Kaggle datasnaek/chess** | 115,500 (~112,702 unique)         | games.csv; 10k in JSON, 40k labeled but no JSON, ~63k unlabeled                                       | ✅              | ❌             |
| **Lichess puzzles (HF)**   | 20,000 sampled (4k + 16k)         | Lichess/chess-puzzles; 4k in JSON; 16k extract unlabeled                                              | ✅              | ❌             |
| **ChessBench bags**        | ~62,829 (LEGACY)                  | SF win_prob, not Lc0 v̂; not Goal JSON                                                                | ✅              | ❌             |
| **Lichess monthly dump**   | games 1–10 000 in JSON slices     | Standard rated 2026-07 `.pgn.zst` (27.06 GiB); stream via `scripts/lichess_dump_to_fen_value_visits.py n m`. Full dump not converted. | ✅              | ~              |

---

## Live tree

| Path | Role |
|------|------|
| `src/tinymlinternship/` | Goal §1 helpers: schema, board_store, Lc0 parse/preprocess, `eval_lc0` |
| `scripts/` | Download (incl. `download_lichess_dump.py`), dump range `n`–`m` → fen-value-visits, `join_fen_value_visits.py`, extract FENs, `label_positions.py`, `export_fen_value_visits.py`, `select_unlabeled_fens.py` |
| `tests/test_fen_value_visits.py` | EPD hash / visits aggregation |
| `src/cfish/` | Goal §4 search (stock eval) |
| `models/teacher/` | Lc0 teacher |
| `data/` | Raw + processed (see above) |
| `LEGACY/` | Old product (8-head / F3 / Python αβ / HCE / ICTP 2026-07) |

---

## Next

1. Stream-convert more of `lichess_db_standard_rated_2026-07.pgn.zst` with `scripts/lichess_dump_to_fen_value_visits.py n m` (do not inflate the full PGN on disk). Games 1–10 000 are in JSON slices (adjacent ranges share a boundary game).
2. Export `lichess_kaggle_40k` → `fen_value_visits/fen_value_visits_lichess_kaggle_40k.json`.
3. Label `puzzles_sample_16k` if that slice should enter the JSON folder.
4. Re-run `scripts/join_fen_value_visits.py` after new slices.
5. Grow unique labeled EPDs toward \(10^6\).
6. Only then Goal §2 (844 encoder + DualHidden train).
