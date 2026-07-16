# SARDINE — Pipeline Assets

Quick map of **datasets**, **models**, and **scripts** used by the SARDINE pipeline.  
Paths are relative to the project root. Run commands with `py -3.12` from the root unless noted.

Longer command recipes: [NOTES/Commands.md](NOTES/Commands.md) · data notes: [NOTES/Datasets.md](NOTES/Datasets.md) · teachers: [NOTES/Models.md](NOTES/Models.md) · blueprint: [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md).

---

## Ideal final training set (production target)

This is the **canonical training product** the pipeline must produce before a production NNUE run (nnue-pytorch / full `train_nnue`). Everything below “what we have now” is either smoke, pilot, or intermediate.

### One-sentence definition

> **Positions sampled from real human games (Lichess primary) plus a strong-play supplement (Lc0), each labeled once by a fixed Lc0 teacher into a scalar expected reward in \([-1,+1]\), stored as parquet with FEN + bucket metadata and a natural bucket mix — no stratified resampling.**

### Uniformity rule (non-negotiable)

Blocks may live in **separate files** (`lichess_*.parquet`, `lc0_*.parquet`, then optional merge). They must still be **one training policy**:

| Must be the same for every production row | Must **not** differ by source |
| ----------------------------------------- | ----------------------------- |
| Training target column: **`expected_reward`** only | Lichess using game `result` while Lc0 uses `best_q` |
| Formula: White-POV expected outcome ∈ **[-1, +1]** from fixed Lc0 WDL | ChessBench / SF-cp / chunk Q mixed into the same train set |
| Same **`teacher_network`** (and preferably same binary) for all blocks in a run | “fast” labels on Lichess + BT4 labels on Lc0 without a full re-label |
| Same pre-label schema (`fen`, `bucket_id`, …) before labeling | Different column names / meanings per block |

**Separate is OK; heterogeneous evals are not.**  
If a block already has another score (`best_q`, ChessBench value, centipawns), it is **metadata or discard** until `label_positions.py` writes a uniform `expected_reward`.  
Train/val loaders must read **only** `expected_reward` as `target` — never `best_q` / `root_q` / game result.

### Data sources (positions only)

| Role | Source | What we take | What we do **not** take |
| ---- | ------ | ------------ | ----------------------- |
| **Primary (~majority of rows)** | [Lichess monthly PGN](https://database.lichess.org/) | Legal mid-game FENs from human games | Game result as label; raw PGN at train time |
| **Supplement (volume + strong play)** | Lc0 training chunks (`data/raw/lc0/`) | FENs from self-play positions (classical boards) | Lc0 feature planes; chunk `best_q` / `root_q` as labels |
| **Excluded from NNUE train** | Kaggle `games.csv`, ChessBench bags, pre-labeled SF-cp dumps | — | Used only for smoke, stats, or encoder wiring |

**Filters (positions):**

- Prefer games with **≥ 16 full moves** (aligned with piece-count study / blueprint).
- Optional opening skip: e.g. `min_ply ≥ 32` globally, relaxed for bucket 7 (`p=32`) so early-game density is not zero.
- **Natural bucket distribution** — keep the empirical frequencies from the source games; do **not** force 1/8 per bucket for training (queen-split `bucket_id` is for routing only).
- Sample every position or every *N* plies (`--sample-every`) for volume control; document the choice in the dataset manifest.

### Who labels (teacher)

| Field | Ideal value |
| ----- | ----------- |
| **Engine** | Lc0 binary (`models/teacher/lc0/lc0.exe`) |
| **Network** | **One fixed “latest best” (or chosen production) net** for the whole dataset — same weights for all rows (currently installed candidates: `791556` fast, T1-256, BT4). Record net id + sha256 in the manifest. |
| **Protocol** | UCI: `position fen …` → value head WDL (permille) → scalar label |
| **Label formula** | Intermediate: \(Q_{\mathrm{STM}} = (W - L) / 1000\). **Stored training target:** **White POV** expected reward in \([-1,+1]\) (flip sign when Black to move) — matches `Lc0Teacher` / search stack. |
| **Not labels** | Stockfish centipawns, game outcome (1-0/0-1/½), ChessBench values, Lc0 chunk `best_q` from a different generating net |

Fallback if Lc0 unavailable: Stockfish `UCI_ShowWDL` + `eval` (same White-POV scalar). Do not mix teachers inside one train/val split without recording it.

### File layout (ideal on disk)

```text
data/
├── raw/
│   ├── lichess/                    # monthly .pgn(.zst) from database.lichess.org
│   ├── lichess_smoke50.pgn         # small smoke only (from Kaggle SAN) — not production
│   └── lc0/                        # chunks + tars + manifest
├── processed/
│   ├── lichess/
│   │   ├── positions.parquet       # FEN + bucket_* only (pre-label)
│   │   └── positions.stats.json
│   └── lc0/
│       ├── positions.parquet       # FEN + bucket_* only (pre-label)
│       └── splits/ …               # optional pre-label splits
└── labeled/                        # ← TRAIN HERE
    ├── train.parquet               # merged Lichess + Lc0, labeled
    ├── val.parquet
    ├── manifest.json               # sources, teacher net, filters, counts, bucket hist
    └── (optional) lichess_*.parquet / lc0_*.parquet before merge
```

**Ideal production paths (targets, may not exist yet):**

| Artifact | Path |
| -------- | ---- |
| Lichess FENs (unlabeled) | `data/processed/lichess/positions.parquet` |
| Lc0 FENs (unlabeled) | `data/processed/lc0/positions.parquet` |
| **Training split** | `data/labeled/train.parquet` (or `data/processed/labeled/train.parquet`) |
| **Validation split** | `data/labeled/val.parquet` |
| **Dataset card** | `data/labeled/manifest.json` |

### Row schema (parquet)

Each training row is one position. **Required columns for production train:**

| Column | Type | Meaning |
| ------ | ---- | ------- |
| `fen` | string | Standard FEN (legal position) |
| `expected_reward` | float32 | Teacher label, **White POV**, range **[-1, +1]** |
| `bucket_id` | int8 | SARDINE router bucket `0…7` (`features/bucket.py`) |
| `piece_count` | int8 | Pieces on board (kings included), 2…32 |
| `has_queen` | bool | Either side still has a queen |
| `stm_white` | bool | Side to move is White (derivable from FEN; store for loaders) |
| `ply` | int32 | Half-move index from game start (if known) |
| `source` | string | `"lichess"` \| `"lc0"` (provenance) |
| `game_id` | string | Stable id within source (Lichess id / chunk game key) |

**Optional but recommended:**

| Column | Meaning |
| ------ | ------- |
| `wdl_win`, `wdl_draw`, `wdl_loss` | Raw teacher WDL permille (side to move) for audit |
| `teacher_network` | Net filename or id used for this row |
| `white` / `black` / `result` | Game headers (debug only; **not** training targets) |
| `white_features` / `black_features` | Sparse 844 index lists if precomputed (speed); else encode from FEN at train time |

**Loader contract (student NNUE):** map row → dual sparse 844 features (from FEN or precomputed indices) + `bucket_id` + target `expected_reward`. Features always come from the **SARDINE encoder**, never from Lc0 planes.

### Format & quality rules

| Rule | Spec |
| ---- | ---- |
| **On-disk format** | Apache **Parquet** (columnar; pandas/pyarrow); CSV only for tiny smoke |
| **Uniform eval** | Every production block’s train target is **`expected_reward`** (same formula + same teacher net). Blocks may stay separate files until merge. |
| **Train/val split** | Fixed seed; e.g. ~95% / 5% by **game** (not by position) to limit leakage |
| **Label range** | Reject or clamp only after investigation; pipeline must assert all `expected_reward ∈ [-1, +1]` |
| **Teacher consistency** | Single net id for entire `train`+`val` of a run (all sources); write it in `manifest.json` |
| **No stratified rebalance** | Bucket histogram may be skewed (e.g. few queenless near-full boards); that is intentional |
| **Size (order of magnitude)** | Blueprint-scale: enough for 100-epoch train — typically **10⁵–10⁶+** labeled positions once Lichess dumps are in; smoke (tens–thousands) is **not** production |
| **Manifest must record** | Source URLs/paths, game filters, sample rate, teacher binary+net+sha256, label formula, row counts, bucket histogram, create date |

### Build chain (scripts)

```text
database.lichess.org  ──PGN──►  data/raw/lichess/
                                      │
                                      ▼
                         lichess_pgn_to_fen.py
                                      │
                                      ▼
                         data/processed/lichess/positions.parquet
                                      │
data/raw/lc0/ ──► prepare_lc0_dataset.py ──► data/processed/lc0/positions.parquet
                                      │
                                      ▼
                         label_positions.py  (Lc0 UCI teacher; keeps metadata)
                                      │
                                      ▼
                         merge_training_sets.py
                                      │
                                      ▼
              data/processed/labeled/{train,val}.parquet  +  manifest.json
                                      │
                                      ▼
                         nnue-pytorch / train_nnue  ──►  models/checkpoints/nnue/
```

### What is **not** the ideal set (status today)

| Artifact | Why it is not production |
| -------- | ------------------------ |
| `data/processed/chessbench/` + ChessBench pilot labels | Smoke for encoder/train wiring; SF/ChessBench values, not Lc0-on-Lichess |
| `data/processed/labeled/*smoke*` | Tiny checks (startpos, 10 CB FENs, 50 Lichess-smoke FENs) |
| `data/raw/games.csv` | Outcomes + weak games; no per-position teacher eval |
| Lc0 `best_q` inside chunks | Wrong teacher net / not our labeling policy |
| `pilot_W128_844` | Trained on ChessBench pilot — useful baseline, not the final student |

---

## Datasets (what exists now)

### Raw

| Asset | Path | Role | How to obtain |
| ----- | ---- | ---- | ------------- |
| **Lc0 training chunks** | `data/raw/lc0/` (`tars/`, `chunks/`, `manifest.json`) | Position volume for FEN sampling (supplement). ~1.1 GiB gzip chunks | `scripts/download_lc0.py` |
| **ChessBench bags** | `data/raw/chessbench/test/` (`action_value_data.bag`, `state_value_data.bag`) | Smoke / pilot features only — **not** production train | `scripts/download_chessbench.py` |
| **Kaggle games CSV** | `data/raw/games.csv` | Piece-count / bucket stats (~20k games) — **not** NNUE train | `scripts/download_data.py` |
| **Lichess-style smoke PGN** | `data/raw/lichess_smoke50.pgn` | 50 games (from Kaggle SAN) for pipeline smoke | Built for `lichess_pgn_to_fen` smoke |
| **Lichess monthly PGN (ideal)** | `data/raw/lichess/` | Primary production FENs | [database.lichess.org](https://database.lichess.org/) — not downloaded yet |

### Processed

| Asset | Path | Role |
| ----- | ---- | ---- |
| **ChessBench parquet** | `data/processed/chessbench/` (`positions.parquet`, `splits/train.parquet`, `val.parquet`, `manifest.json`) | Pilot NNUE smoke train/val |
| **Lc0 parquet (pilot)** | `data/processed/lc0/` (`positions.parquet` = **3 149** natural filter-pass; `splits/` stratified pilot; `manifest.json`) | Full ASSETS pre-label schema; buckets 1/3/5 empty in this scan |
| **Lichess FENs (mini)** | `data/processed/lichess/positions.parquet` | 2 371 FENs, full pre-label schema, all 8 buckets (from smoke50 PGN) |
| **Lichess FEN smoke (legacy name)** | `data/processed/lichess/smoke_fens.parquet` | Same generation path; prefer `positions.parquet` |
| **Startpos labels (smoke)** | `data/processed/labeled/smoke_labeled.parquet` | Startpos smoke via `label_positions.py` |
| **ChessBench labels (smoke)** | `data/processed/labeled/chessbench_smoke10.parquet` | 10 FENs labeled with Lc0 WDL |
| **Lichess labels (smoke)** | `data/processed/labeled/lichess_smoke_labeled.parquet` | 50 FENs, Lc0 `791556`, ~5.9 pos/s |

### Derived analytics

| Asset | Path | Script |
| ----- | ---- | ------ |
| Piece-count / bucket distribution | `excel/piece_count_distribution.xlsx`, `excel/piece_count_distribution_10k.xlsx` | `scripts/plot_piece_count_distribution.py` |
| Plots | `plots/` (ACPL JSON/PGN, architecture PNG) | eval + plot scripts below |

---

## Models

### Student NNUE (SARDINE)

| Checkpoint | Path | Notes |
| ---------- | ---- | ----- |
| **`pilot_W128_844`** (active) | `models/checkpoints/nnue/pilot_W128_844/best.pt` | W=128, 844-dim encoder; default for `--eval nnue` |
| Config / history | same dir (`config.json`, `history.json`, `last.pt`) | Training metadata |
| Older pilot | `models/checkpoints/nnue/pilot_W128_chessbench/` | Earlier ChessBench run |

Train (smoke only):

```bash
pip install -e ".[train]"
py -3.12 scripts/prepare_chessbench_dataset.py
py -3.12 scripts/train_nnue.py --epochs 10 --run-name pilot_W128_844
```

### Teachers & external engines

| Asset | Path | Role |
| ----- | ---- | ---- |
| **Lc0 binary** | `models/teacher/lc0/lc0.exe` | UCI labeling + teacher play |
| **Lc0 fast net (default play/label smoke)** | `models/teacher/lc0/791556.pb.gz` | Fast CPU net |
| **Lc0 T1-256** | `models/teacher/networks/t1-256x10-distilled-swa-2432500.pb.gz` | Stronger/slower alternative |
| **Lc0 BT4** | `models/teacher/networks/BT4-1024x15x32h-swa-6147500.pb.gz` | Quality ref (very slow) |
| **Teacher manifest** | `models/teacher/manifest.json` | Installed binary + default network record |
| **Stockfish** | `models/teacher/stockfish/stockfish.exe` | ACPL / Elo gate judge |
| **Sunfish** | `models/teacher/sunfish/` | Weak baseline for ACPL calibration |
| **chess_lite (HF)** | `models/teacher/hf/chess_lite/chess_lite.pth` | Fast PyTorch baseline (weak play) |
| **Artoria Zero small (HF)** | `models/teacher/hf/artoria-zero/small/checkpoint.pt` | HF transformer baseline |

Install teachers:

```bash
py -3.12 scripts/download_teacher.py      # lc0 + networks
py -3.12 scripts/download_hf_teacher.py   # chess_lite + artoria-small
py -3.12 scripts/smoke_test_teacher.py
```

External sources: [Lc0 training / nets](https://training.lczero.org/) · [lczero.org storage](https://storage.lczero.org/) · [chess_lite](https://huggingface.co/satana123/chess_lite) · [Artoria Zero](https://huggingface.co/Shinapri/artoria-zero).

---

## Useful scripts

### Run the bot / search

| Script | What it does |
| ------ | ------------ |
| **`scripts/run_engine.py`** | Best move for a FEN / move list (`--eval hce\|nnue\|lc0`, `--depth`) |
| **`scripts/record_engine_game.py`** | Self-play → GIF + PGN (`images/sardine_game.gif` by default) |
| **`scripts/record_teacher_game.py`** | Self-play with Lc0 as eval |
| **`scripts/record_hf_game.py`** | Self-play with HF teacher (chess_lite / artoria) |

```bash
# Single position
py -3.12 scripts/run_engine.py --depth 1
py -3.12 scripts/run_engine.py --eval nnue --depth 2
py -3.12 scripts/run_engine.py --eval nnue --nnue-checkpoint models/checkpoints/nnue/pilot_W128_844/best.pt --depth 2

# Self-play GIF
pip install -e ".[viz]"
py -3.12 scripts/record_engine_game.py --eval nnue --depth 2 --headless --max-plies 80
py -3.12 scripts/record_engine_game.py --eval hce --depth 2 --no-quiescence --headless
```

Frozen bot recipes: [NOTES/agents/](NOTES/agents/) (`nnue-w128-844-d1.md`, `nnue-w128-844-d2.md`, `hce-d2-qsearch.md`).

### Data & labeling

| Script | What it does |
| ------ | ------------ |
| `scripts/download_lc0.py` | Download/extract Lc0 training shards |
| `scripts/download_chessbench.py` | ChessBench raw bags |
| `scripts/download_data.py` | Kaggle `games.csv` |
| **`scripts/lichess_pgn_to_fen.py`** | Stream PGN → FEN + full pre-label schema (`bucket_id`, `stm_white`, `source=lichess`, …) |
| `scripts/prepare_lc0_dataset.py` | Lc0 chunks → filtered parquet + splits (ASSETS columns via `positions_to_dataframe`) |
| `scripts/prepare_chessbench_dataset.py` | Bags → ChessBench train/val parquet |
| `scripts/stats_lc0_processed.py` | Bucket / filter stats on Lc0 data |
| `scripts/smoke_test_lc0_chunk.py` | Parse one chunk end-to-end |
| **`scripts/label_positions.py`** | FEN → Lc0 WDL → `expected_reward` (White POV); **preserves** metadata + `teacher_network` |
| **`scripts/merge_training_sets.py`** | Merge labeled parquets → `train.parquet` / `val.parquet` / `manifest.json` (split by `game_id`) |
| Schema helpers | `src/tinymlinternship/data/schema.py` — column contract, split, manifest |

### Train & evaluate

| Script | What it does |
| ------ | ------------ |
| `scripts/train_nnue.py` | PyTorch pilot train (ChessBench smoke) |
| **`scripts/eval_bot_acpl.py`** | Self-play + Stockfish ACPL → heuristic Elo |
| `scripts/eval_game_elo.py` | ACPL/Elo from an existing PGN |
| `scripts/sunfish_selfplay_pgn.py` | Sunfish self-play PGN for baseline gate |
| `scripts/bench_teacher_move.py` | One-move Lc0 smoke latency |
| `scripts/bench_teacher_nets.py` | Compare teacher networks |

```bash
py -3.12 scripts/eval_bot_acpl.py --eval nnue --depth 1 --max-plies 80 --no-quiescence --sf-movetime-ms 100
```

### Code package (library, not CLI)

| Area | Path |
| ---- | ---- |
| Feature encoder (844) | `src/tinymlinternship/features/` |
| Search + eval hooks | `src/tinymlinternship/engine/` (`search.py`, `eval_hce.py`, `eval_nnue.py`, `eval_lc0.py`) |
| NNUE model + dataset | `src/tinymlinternship/nnue/` |
| Lc0 / ChessBench data I/O | `src/tinymlinternship/data/` |
| ACPL helpers | `src/tinymlinternship/bot_eval/` |
| GIF / pygame board | `src/tinymlinternship/visualization/` |

### Demo artifacts

| Asset | Path |
| ----- | ---- |
| Engine self-play GIF / PGN | `images/sardine_game.gif`, `images/sardine_game.pgn` |
| HCE / NNUE depth-2 demos | `images/hce_d2_game.gif`, `images/nnue_d2_game.gif` |
| Dated multi-game PGN/GIF | `images/games/` |
| Gate ACPL results | `plots/*_gate_acpl.json`, `plots/*_gate.pgn` |

---

## Pipeline order (asset flow)

**Production (ideal):**

```text
Lichess PGN + Lc0 chunks
    → lichess_pgn_to_fen.py / prepare_lc0_dataset.py
    → data/processed/{lichess,lc0}/positions.parquet
    → label_positions.py     (teacher: models/teacher/lc0 + fixed net)
    → data/labeled/{train,val}.parquet + manifest.json
    → nnue-pytorch / train_nnue.py
    → models/checkpoints/nnue/
    → run_engine.py / record_engine_game.py  (--eval nnue)
    → eval_bot_acpl.py       (judge: models/teacher/stockfish)
```

**Smoke / pilot only (do not confuse with production):**

```text
ChessBench bags | games.csv | lichess_smoke50.pgn
    → prepare_chessbench_dataset.py | lichess_pgn_to_fen.py
    → pilot train or *smoke* labeled parquets
```
