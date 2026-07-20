# AI feed — SARDINE pipeline map

_Snapshot: 2026-07-20. Progress vs blueprint (`TODOs.md`); code map for agents._

---

## Progress status (build pipeline A–J)

```text
A  Feature encoder (PC)     ████████████████████  DONE (844-dim; device parity open)
B  Search skeleton (PC)     ████████████░░░░░░░░  PARTIAL  v0.3 (αβ + qsearch + MVV-LVA)
C  Train bucketed NNUE      ████████░░░░░░░░░░░░  MINI SET  C–E smoke; full volume + nnue-pytorch open
D  Queen-split ablation     ░░░░░░░░░░░░░░░░░░░░  NOT STARTED
E  Incremental accumulators ░░░░░░░░░░░░░░░░░░░░  NOT STARTED (device)
F  Port C / Wio             ░░░░░░░░░░░░░░░░░░░░  NOT STARTED
G  Full search stack        ██░░░░░░░░░░░░░░░░░░  qsearch only; TT/LMR/NMP/ID open
H  Elo gate ≥1700           ░░░░░░░░░░░░░░░░░░░░  NOT STARTED (d1 ACPL heuristic only)
I  Iterate if miss gate     ░░░░░░░░░░░░░░░░░░░░  —
J  v2 polish                ░░░░░░░░░░░░░░░░░░░░  —
```

| Area | Now | Next bottleneck |
| ---- | --- | --------------- |
| **Encoder** | ✅ 844 dual-POV, 8 buckets | Device parity (C) |
| **Search** | ✅ v0.3 αβ + capture qsearch | TT, nodes/s, killers, ID |
| **Data** | ✅ Mini labeled + merge (5306/214) | Full Lichess dump + re-label |
| **NNUE train** | ✅ Pilot ChessBench + smoke production path | nnue-pytorch / scale |
| **Bot (PC)** | ✅ HCE / NNUE / Lc0 eval + self-play GIFs | True human PvP UI; UCI |
| **Device** | ❌ | After stronger PC net + search |

**Tests:** 106 passed (2026-07-20).  
**Checkpoints:** `models/checkpoints/nnue/pilot_W128_844/`, `smoke_prod_W128_844/`.  
**Source of truth for checkboxes:** [TODOs.md](TODOs.md) · blueprint [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md) · assets [ASSETS.md](ASSETS.md).

---

## Main components (architecture)

```text
                    ┌─────────────────────────────────────────┐
                    │              DATA PIPELINE              │
  raw PGN/chunks ──►│ download → extract FEN → label (Lc0)    │
                    │ → merge train/val + manifest            │
                    └──────────────────┬──────────────────────┘
                                       │ expected_reward ∈ [-1,1]
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │                 TRAIN                   │
  fen (+ features)─►│ ChessbenchDataset → BucketedNNUE        │
                    │ train_nnue.py → checkpoint best.pt      │
                    └──────────────────┬──────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           ▼                           ▼                           ▼
    ┌─────────────┐           ┌─────────────────┐          ┌──────────────┐
    │  FEATURES   │           │     ENGINE      │          │     BOT      │
    │ encode_dual │◄──────────│ search + eval   │─────────►│ self-play /  │
    │ bucket_id   │           │ hce|nnue|lc0    │          │ ACPL / GIF   │
    └─────────────┘           └─────────────────┘          └──────────────┘
           │                           │
           └──────────── Wio / C ──────┘  (not started)
```

| Package / area | Role |
| -------------- | ---- |
| `src/tinymlinternship/features/` | Sparse 844 encoder, buckets, tactical bits |
| `src/tinymlinternship/data/` | Lc0 parse, ChessBench preprocess, ASSETS schema/merge helpers |
| `src/tinymlinternship/nnue/` | Model + dataset loader |
| `src/tinymlinternship/engine/` | Search, eval backends, perft |
| `src/tinymlinternship/bot_eval/` | Stockfish ACPL → heuristic Elo |
| `src/tinymlinternship/visualization/` | Pygame board, GIF export, engine game loop |
| `src/tinymlinternship/config/settings.py` | Paths (data, teacher, checkpoints) |
| `scripts/` | CLI entrypoints for the above |
| `legacy/pre-sardine/` | Pre-SARDINE value-net / Arduino — **not** active path |
| `models/teacher/` | Lc0 binary + nets (labeling + optional eval) |
| `models/checkpoints/nnue/` | Trained student nets |
| `data/raw/`, `data/processed/` | Datasets |

---

## 1 · Dataset creation

End-to-end product: unlabeled FENs → teacher `expected_reward` → merged `train.parquet` / `val.parquet` + `manifest.json`.

### Scripts (`scripts/`)

| File | Role |
| ---- | ---- |
| `download_lc0.py` | Download Lc0 training chunks (~1–2 GB) → `data/raw/lc0/` |
| `download_data.py` | Kaggle / Lichess smoke (`games.csv`) — **not** production NNUE volume |
| `download_chessbench.py` | ChessBench `.bag` (encoder/train smoke only) |
| `download_teacher.py` | Install Lc0 binary + networks under `models/teacher/` |
| `download_hf_teacher.py` | HF value teachers (optional / study; not production labels) |
| `lichess_pgn_to_fen.py` | PGN → FEN + `bucket_id` / piece_count / has_queen (pre-label) |
| `prepare_lc0_dataset.py` | Lc0 chunks → filtered `positions.parquet` |
| `prepare_chessbench_dataset.py` | ChessBench bag → parquet w/ precomputed features + rewards |
| `label_positions.py` | UCI Lc0 WDL → uniform `expected_reward` (White POV) |
| `merge_training_sets.py` | Merge labeled blocks → train/val by `game_id` + `manifest.json` |
| `stats_lc0_processed.py` | Bucket/ply stats on Lc0 chunks (QA pre-label) |
| `smoke_test_lc0_chunk.py` | Parse one chunk smoke |
| `smoke_test_teacher.py` | Teacher WDL on startpos |
| `plot_piece_count_distribution.py` | Bucket design study (piece-count hist) |
| `study_chessbench.py` | ChessBench exploration helper |

### Libraries (`src/tinymlinternship/`)

| File | Role |
| ---- | ---- |
| `data/schema.py` | ASSETS columns, validate rewards, split-by-game, manifest |
| `data/lc0_parser.py` | Parse Lc0 training chunk format |
| `data/lc0_preprocess.py` | Filters (ply, game length, buckets) |
| `data/lc0_shards.py` | Shard/index helpers for raw Lc0 |
| `data/chessbench_preprocess.py` | Bag row → FEN + features + reward |
| `features/encoder.py` | `encode_dual` / `encode_perspective` (used when features not stored) |
| `features/bucket.py` | `bucket_id`, piece_count, has_queen |
| `features/index_map.py`, `mirror.py`, `tactical.py` | 844-dim layout |
| `engine/eval_lc0.py` | Teacher WDL → expected_reward helpers (shared formula) |
| `config/settings.py` | Default data / teacher paths |

### Typical artifacts

```text
data/raw/lc0/ … , data/raw/lichess_smoke50.pgn
data/processed/lichess/positions.parquet   # pre-label
data/processed/lc0/positions.parquet
data/processed/labeled/lichess_labeled.parquet
data/processed/labeled/lc0_labeled.parquet
data/processed/labeled/train.parquet
data/processed/labeled/val.parquet
data/processed/labeled/manifest.json
```

### Tests

`tests/test_data.py`, `test_dataset_schema.py`, `test_lc0_parser.py`, `test_lc0_preprocess.py`, `test_download_lc0.py`, `test_chessbench_preprocess.py`, `test_features.py`, `test_tactical.py`

---

## 2 · Training (produce NNUE)

Student: bucketed NNUE **844 → W → 8 experts → tanh**, target = `expected_reward`.

### Scripts

| File                        | Role                                                                  |
| --------------------------- | --------------------------------------------------------------------- |
| `train_nnue.py`             | **Main** PyTorch smoke/pilot train → `models/checkpoints/nnue/<run>/` |
| `plot_nnue_architecture.py` | Architecture diagram (docs / presentation)                            |

_Not in-repo yet (blueprint): nnue-pytorch fork, gradual L1 prune, PTQ int8 export._

### Libraries

| File | Role |
| ---- | ---- |
| `nnue/model.py` | `BucketedNNUE` (shared L1, CReLU, 8 heads) |
| `nnue/dataset.py` | Parquet loader: precomputed features **or** FEN → `encode_dual` |
| `nnue/__init__.py` | Public exports |
| `features/*` | Feature dim / encoder used at load time |
| `config/settings.py` | `NNUE_CHECKPOINTS_DIR`, default checkpoint path |

### Checkpoints (examples)

| Path | Notes |
| ---- | ----- |
| `models/checkpoints/nnue/pilot_W128_844/` | ChessBench pilot, val_mse **0.056** (wired in engine default) |
| `models/checkpoints/nnue/smoke_prod_W128_844/` | Mini production labels, 2 ep, val_mse **0.247** |

### Tests

`tests/test_nnue_model.py` (forward, dataset ChessBench + FEN path)

---

## 3 · Bot & inference (engine play / “PvP”)

Today = **PC engine**: static eval + alpha-beta search. Primary “play” is **engine self-play** and **single-position search**. There is **no dedicated human-vs-bot CLI/UI** yet; closest interactive path is pygame playback of a generated game (`record_engine_game.py`).

### Scripts — play / record

| File                      | Role                                                                                    |
| ------------------------- | --------------------------------------------------------------------------------------- |
| `run_engine.py`           | **Inference CLI**: FEN (+ moves) → best move / score (`--eval hce nnue lc0`, `--depth`) |
| `record_engine_game.py`   | Engine **self-play** → pygame + GIF + PGN (`--eval`, `--depth`, qsearch flags)          |
| `record_teacher_game.py`  | Self-play with Lc0 teacher eval (baseline strength viz)                                 |
| `record_hf_game.py`       | Self-play with HF value net (study)                                                     |
| `sunfish_selfplay_pgn.py` | Sunfish baseline games for ACPL calibration                                             |

### Scripts — strength / gate

| File | Role |
| ---- | ---- |
| `eval_bot_acpl.py` | Self-play (or PGN) → Stockfish ACPL → heuristic Elo |
| `eval_game_elo.py` | Per-player Elo estimate from one game record |
| `bench_teacher_move.py` | Teacher move latency bench |
| `bench_teacher_nets.py` | Compare teacher nets |

### Libraries — search & eval

| File | Role |
| ---- | ---- |
| `engine/search.py` | Alpha-beta, quiescence, MVV-LVA, `search_best_move` |
| `engine/eval_factory.py` | `make_eval_fn` / `EVAL_CHOICES` (`hce`, `nnue`, `lc0`) |
| `engine/eval_hce.py` | Hand-crafted evaluation (default) |
| `engine/eval_nnue.py` | Load checkpoint → centipawns for search |
| `engine/eval_lc0.py` | Live Lc0 teacher as eval backend |
| `engine/eval_chess_lite.py` | HF chess_lite-style value (optional) |
| `engine/perft.py` | Move-gen correctness |
| `engine/__init__.py` | Engine public API / version |
| `nnue/model.py` | Forward pass used by `eval_nnue` |
| `features/encoder.py` | Position → sparse 844 for NNUE |
| `bot_eval/acpl.py` | ACPL / Elo heuristic math |
| `bot_eval/stockfish_path.py` | Resolve Stockfish binary |
| `visualization/playback.py` | `play_engine_game` game loop |
| `visualization/pygame_board.py` | Board render |
| `visualization/gif_export.py` | GIF export |
| `visualization/game_paths.py` | Output path helpers |
| `config/settings.py` | `NNUE_CHECKPOINT_DEFAULT`, etc. |

### Quick entrypoints

```bash
# One position
py -3.12 scripts/run_engine.py --eval nnue --depth 2 --fen "..."

# Self-play GIF (closest to “watch the bot play”)
py -3.12 scripts/record_engine_game.py --eval nnue --depth 1 --headless --output images/nnue_d1_game.gif

# Strength gate (16 games typical)
py -3.12 scripts/eval_bot_acpl.py --eval nnue --depth 1 --games 16
```

### Tests

`tests/test_engine.py`, `test_eval_nnue.py`, `test_eval_lc0.py`, `test_perft.py`, `test_bot_eval_acpl.py`, `test_visualization.py`

---

## Explicitly out of these three buckets (but related)

| Area | Files |
| ---- | ----- |
| Docs / progress | `TODOs.md`, `Goal.md`, `PROJECT.md`, `ASSETS.md`, `NOTES/*`, daily `YYYY-MM-DD.md` |
| Presentation | `scripts/build_sardine_presentation.py`, `verify_presentation.py`, `presentations/` |
| Legacy TinyML / Arduino | `legacy/pre-sardine/` |
| Device port | _none active_ (blueprint E–F) |

---

## Mini production path (commands that already ran)

```bash
# Label
py -3.12 scripts/label_positions.py --input data/processed/lichess/positions.parquet ...
py -3.12 scripts/label_positions.py --input data/processed/lc0/positions.parquet ...

# Merge
py -3.12 scripts/merge_training_sets.py \
  -i data/processed/labeled/lichess_labeled.parquet data/processed/labeled/lc0_labeled.parquet \
  -o data/processed/labeled --val-fraction 0.05 --seed 42

# Smoke train
py -3.12 scripts/train_nnue.py \
  --train data/processed/labeled/train.parquet \
  --val data/processed/labeled/val.parquet \
  --epochs 2 --batch-size 256 --run-name smoke_prod_W128_844

# Infer
py -3.12 scripts/run_engine.py --eval nnue --nnue-checkpoint models/checkpoints/nnue/pilot_W128_844/best.pt --depth 1
```
