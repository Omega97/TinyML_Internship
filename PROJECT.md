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

**Goal §2 — dual-POV 2-hidden NNUE.** Dataset gate \(\gtrsim 10^6\) is met. Smoke trainer is live (`scripts/train_nnue.py`). MoE / Cfish-eval swap not started.

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

- [x] Live `DualHiddenNNUE` (L1 64×2, L2 128) + 844 encoder under `src/tinymlinternship/`
- [x] `scripts/train_nnue.py` — joined parquet train, `test_set_2026-07_100000-100100` holdout, visit-weighted MSE
- [~] 10-epoch **smoke** (20k random train rows, 6 381 test): train MSE ↓, test MSE ↑ (overfit). Full-table train not run. Plot: `plots/nnue_smoke_mse.png`

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

`data/processed/board_eval/fen_value_visits/<slice>/`

Each slice folder is named after the file stem (e.g. `fen_value_visits_lc0/fen_value_visits_lc0.json`). Each file is a list of `{fen, value, visits}` objects (unique EPD; `visits` = extract multiplicity). Slices on disk:

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
| `scripts/` | Download (incl. `download_lichess_dump.py`), dump range `n`–`m` → fen-value-visits, `count_fen_value_visits.py` (unique EPD vs total rows), `join_fen_value_visits.py`, extract FENs, `label_positions.py`, `export_fen_value_visits.py`, `select_unlabeled_fens.py` |
| `tests/test_fen_value_visits.py` | EPD hash / visits aggregation |
| `src/cfish/` | Goal §4 search (stock eval) |
| `models/teacher/` | Lc0 teacher |
| `data/` | Raw + processed (see above) |
| `LEGACY/` | Old product (8-head / F3 / Python αβ / HCE / ICTP 2026-07) |

---

## Next

1. Stream-convert more of `lichess_db_standard_rated_2026-07.pgn.zst` with `scripts/lichess_dump_to_fen_value_visits.py n m` (`[n, m)`, m exclusive; do not inflate the full PGN on disk). Games 1–10 000 are in JSON slices (older inclusive runs may share a boundary game).
2. Export `lichess_kaggle_40k` → `fen_value_visits/fen_value_visits_lichess_kaggle_40k.json`.
3. Label `puzzles_sample_16k` if that slice should enter the JSON folder.
4. Re-run `scripts/join_fen_value_visits.py` after new slices.
5. Grow unique labeled EPDs toward \(10^6\).
6. Only then Goal §2 (844 encoder + DualHidden train).

---

## Notes for AI

Facts a later agent should not re-discover or violate. Spec is [Goal.md](Goal.md); this file is **status**; [README.md](README.md) is the short pointer.

1. **Five-step order is binding.** Dataset → dual-POV 2-hidden NNUE → task-vector MoE → Cfish `evaluate` swap → ACPL / STS / BayesElo. Do not skip ahead into search, hardware, or thesis polish.
2. **`LEGACY/` is gitignored and dead as a product path.** Old 8-bucket / single-head Python engine / HCE / ICTP 2026-07 live there. Read it to recover the 844 encoder and `DualHiddenNNUE`, then **re-implement under live `src/tinymlinternship/`**. Do not import from `LEGACY/` or revive the Python αβ engine.
3. **Goal §3 is not the 8 piece-count buckets.** Live `features/bucket.py` is the old B–C scheme (piece count + queen). The live MoE plan is: embeddings \(h\), task vectors \(\delta=\nabla_{w^{(head)}}\mathcal L\), cluster, linear dispatcher on \(h\), freeze L1, fine-tune expert heads. Dispatcher is **inference-only**.
4. **Live student (2026-08-23).** `src/tinymlinternship/features/` is the 844 encoder; `src/tinymlinternship/nnue/` is `DualHiddenNNUE` (default L1 W=64 per POV, L2=128). Train: `scripts/train_nnue.py` on per-slice `features.npz`. **Test slice** `fen_value_visits_lichess_db_standard_rated_2026-07_100000-101000`; all other folders are train. `--smoke` caps train at 20k random rows. Checkpoints: `models/checkpoints/nnue/`.
5. **Student architecture (Goal §2 / README).** Sparse **844** input; shared L1 `844 → W` (≈128, int8-intended) run **twice** (own-side + opponent-side / mirrored board); CReLU; concat to `2W` **after** L1; L2 ≈256; one output neuron; **tanh** \(v\in[-1,+1]\). L1 is the incremental accumulator; only L2+head (+ later dispatcher) recompute at a node.
6. **844 layout (from LEGACY encoder, still the spec).** 768 piece-square − 32 pawn ranks 1/8 − 32 perspective-king file-compress + 4 castling + 8 EP file + 64 under-attack + 64 king-attackers = 844. Perspective king is 32-file-mirrored; **enemy king stays 64**. Castling rights must be taken from the **unmirrored** board (python-chess lies on flipped boards). Recover `index_map.py` / `encoder.py` / `mirror.py` / `tactical.py` from `LEGACY/tinymlinternship/features/`.
7. **POV contract.** Dual accumulators are White-view and Black-view (Black = `board.mirror()` then encode as White). Concat order at the head is **[side-to-move, opponent]**. **Target `value` is always White-POV expected reward** \((W-L)/1000\) after flipping STM WDL — not STM-POV. Do not train against STM-signed labels.
8. **Do not reimplement search.** Keep **Cfish** (`src/cfish/`, `run-cfish.bat`) αβ / qsearch / TT. Replace only `evaluate` / `nnue_evaluate`. Map \(v\in[-1,+1]\) onto Cfish `Value` (centipawn-like, `CP_SCALE=1000` in `eval_lc0.py`). Stock net `nn-62ef826d1a6d.nnue` is a search baseline, not the student.
9. **Ship gate is MoE vs vanilla**, not vs Stockfish. ACPL (Stockfish judge in `tools/stockfish/`) and STS are supporting; BayesElo vanilla vs MoE is the decision.
10. **Hardware target is tiny** (Wio-class RAM/flash). Keep L1 sparse/incremental and quantization in mind (CReLU clip 127, int8 L1) even while training float32 in PyTorch.
11. **Canonical labels are `{fen, value, visits}`.** Unique key = **EPD** (FEN fields 1–4: board + STM + castling + EP). Halfmove/fullmove ignored (`board.epd()` / SHA-256). `visits` = extract multiplicity. Older parquet schema used `expected_reward`; Goal JSON uses `value`.
12. **Join rules.** Slices in `data/processed/board_eval/fen_value_visits/<stem>/`; joined table in the **parent** (`fen_value_visits.parquet` + `.json` twin + `.manifest.json`). Count without rewriting: `scripts/count_fen_value_visits.py`. `scripts/join_fen_value_visits.py`: rglob `fen_value_visits_*`; parquet wins over JSON for the same stem; sum visits; **visit-weighted** value rounded to **3 decimals**; sort visits desc. Re-join after any slice relabel.
13. **Joined size (2026-08-22/23):** **2 044 090** unique EPDs, visits sum **2 439 060** — Goal \(\gtrsim 10^6\) is met. Numbers in older PROJECT.md paragraphs may lag; trust the manifest. Unique-EPD ≠ i.i.d.: consecutive plies from the same game are correlated. Opening repeats dominate `visits_max`.
14. **Slice pitfalls.** Join glob is `fen_value_visits_*`. Overlapping dump ranges (`0-1000` JSON vs `1-1000` parquet) and twins (`lichess` vs `lichess_mini`) can **double-count** if both stems remain. Dump converter range is **half-open `[n, m)`**. Never inflate the 27 GiB `lichess_db_standard_rated_2026-07.pgn.zst`; stream with `scripts/lichess_dump_to_fen_value_visits.py`.
15. **Teacher.** Default net **`791556.pb.gz`** (`LC0_NETWORK_DEFAULT` / preset `fast`). T1-256 and BT4 sit unused under `models/teacher/networks/`. Binary `models/teacher/lc0/lc0.exe` v0.32.1. Error text in `eval_lc0.py` may still say “BT4 weights” — it means the configured weights file, usually 791556.
16. **Lc0 `go` is not Stockfish `go`.** Default `go nodes 1` = **value head only** (fast, blind to short mates). `go depth N` is Lc0 **MCTS average tree depth**. Puzzle slice was re-labeled at **depth 2** (`scripts/relabel_fen_value_visits_lc0.py`); most other slices are still `nodes 1`. WDL permille is **side-to-move**; UCI `score mate N` is STM (positive = STM mates). Terminal checkmates can be labeled without the engine.
17. **Do not mix ChessBench.** LEGACY ChessBench bags are Stockfish `win_prob`, not Lc0 \(\hat v\). Out of the Goal JSON.
18. **Train extra:** `pip install -e ".[train]"` (torch + pyarrow). Run **`py -3.12`** from repo root; `pythonpath = src` in pytest. Scripts prepend `src/` themselves. PowerShell: no `&&`; chain with `;`.
19. **LEGACY train defaults are the wrong Goal §2 model.** `LEGACY/scripts/train_nnue.py` defaulted to **F3 single-head** (`844 → W` concat `2W → 1`). Goal wants **`DualHiddenNNUE`**: L1 `844→W`, L2 `2W→H`, head `H→1`, tanh. Demo dims were W=128, H=128 or 256 — pick Goal’s “≈128 / ≈256” unless a later note records otherwise. Loss: MSE on `value` (visit-weight if you use `visits`).
20. **Eval output vs training target.** Training target is White-POV \(v\). Search eval must still be usable from the side to move (Cfish convention). Convert at the `evaluate` hook; do not silently train STM-signed \(v\).
21. **Data/models are gitignored** (`data/raw/`, `data/processed/`, `/models/`, `*.pt`, `*.nnue`). Code + tests + this file are what you commit. Daily notes (`YYYY-MM-DD.md`, `DAILY-NOTES/`) are typically gitignored too.
22. **Tests that exist on the live path:** `tests/test_fen_value_visits.py`, `tests/test_lichess_dump_batch.py`, `tests/test_relabel_fen_value_visits_lc0.py`. LEGACY has encoder/NNUE tests (`test_features.py`, `test_nnue_model.py`, …) — port what you need; do not assume they run against live `src/`.
23. **Doc roles.** [Goal.md](Goal.md) = what to build. **PROJECT.md** = what is on disk. [README.md](README.md) = order + live tree. [ai-feed.md](ai-feed.md) = cleanup log. Session plan/log = root `YYYY-MM-DD.md`. When they disagree on counts, prefer `fen_value_visits.manifest.json`.
24. **Do not invent a second student stack.** No new Python search, no HCE product path, no 8-head training as the Goal student. Incremental L1 add/sub on make/unmake is required for Cfish; a dense 844 matmul is only acceptable in the PyTorch trainer.
25. We created a dataset of 2M FEN codes with Lc0 evaluation (and visit count).

---
