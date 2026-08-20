# SARDINE — Assets

Main **models**, **data**, and **eval artifacts** for the engine. Paths are relative to the repo root.  
Progress: [PROJECT.md](PROJECT.md) · architecture: [blueprint](NOTES/SARDINE%20Engine%20Blueprint.md) · schema code: `src/tinymlinternship/data/schema.py`.

Labels for training are Lc0 WDL → **`expected_reward`** (White POV, \([-1,+1]\)). Stockfish is the ACPL judge only (not a train label source).

---

## Student NNUE (trained)

| Title | Path | Description |
| ----- | ---- | ----------- |
| **Pilot NNUE W128 844** (default student / ladder ref) | `models/checkpoints/nnue/pilot_W128_844/` | Multi-head pilot (`best.pt`). W=128, 844-dim dual-POV; ChessBench pilot splits (~60k train, 5 ep). Default for `--eval nnue`. d1 ACPL gate ~139 / Elo ~1465. |
| **F3 single-head mini** | `models/checkpoints/nnue/single_W128_mini_ep30/` | First **single-head** train (F3): mini labeled `data/processed/labeled/{train,val}.parquet` (5306 / 214), 30 ep, best val_mse **0.196** @ ep6. d1 ACPL gate **~583** / Elo floor **400** — not playable yet. |
| **F3 path smoke** | `models/checkpoints/nnue/f3_path_smoke/` | 1-ep wiring check for `SingleHeadNNUE` on same mini set. |
| **Pilot NNUE W128 ChessBench** (older) | `models/checkpoints/nnue/pilot_W128_chessbench/` | Earlier ChessBench pilot; superseded by `pilot_W128_844`. |
| **Smoke prod W128 844** | `models/checkpoints/nnue/smoke_prod_W128_844/` | Short multi-head smoke (2 ep) on mini labeled set. |

---

## Teachers & external nets

| Title | Path | Description |
| ----- | ---- | ----------- |
| **Lc0 binary** | `models/teacher/lc0/lc0.exe` | UCI teacher for labeling and teacher-play baselines. |
| **Lc0 fast net** | `models/teacher/lc0/791556.pb.gz` | Default fast CPU net used for smoke / mini labeling (`teacher_network` on mini set). |
| **Lc0 T1-256** | `models/teacher/networks/t1-256x10-distilled-swa-2432500.pb.gz` | Stronger/slower alternative teacher net. |
| **Lc0 BT4** | `models/teacher/networks/BT4-1024x15x32h-swa-6147500.pb.gz` | Large quality reference net (slow). |
| **Teacher manifest** | `models/teacher/manifest.json` | Record of installed Lc0 binary + default network. |
| **Cfish stock NNUE** | `src/cfish/nn-62ef826d1a6d.nnue` | Stock Hybrid NNUE shipped next to `cfish.exe` (baseline tree, not SARDINE student). |
| **Sunfish** | `models/teacher/sunfish/` | Weak open-source baseline for ACPL calibration. |
| **Stockfish judge** | PATH / `STOCKFISH_PATH` (not in-repo; optional local under `tools/stockfish/`) | ACPL / Elo gate only. |

---

## Training data — raw

| Title                    | Path                                                  | Description                                                        |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------ |
| **Lc0 training chunks**  | `data/raw/lc0/` (`tars/`, `chunks/`, `manifest.json`) | ~1+ GiB Lc0 self-play shards for FEN sampling (supplement source). |
| **Lichess smoke PGN**    | `data/raw/lichess_smoke50.pgn`                        | 50 games for pipeline smoke (not production volume).               |
| **ChessBench bags**      | `data/raw/chessbench/test/`                           | Raw bags for encoder/train pilot only — not production labels.     |
| **Kaggle games CSV**     | `data/raw/games.csv`                                  | Piece-count / stats smoke only — not NNUE train.                   |
| **Lichess monthly dump** | `data/raw/lichess/`                                   | Production primary source — **not downloaded yet**.                |

---

## Training data — processed & labeled

| Title | Path | Description |
| ----- | ---- | ----------- |
| **Mini train/val (merged)** | `data/processed/labeled/train.parquet`, `val.parquet` | Uniform mini set: Lichess smoke + Lc0 positions labeled with Lc0 `791556` → `expected_reward`. ~5 520 rows (train 5306 / val 214). |
| **Labeled set manifest** | `data/processed/labeled/manifest.json` | Sources, teacher net, row counts, bucket histogram for the mini merge. |
| **Lichess labeled block** | `data/processed/labeled/lichess_labeled.parquet` | Lichess-source block before merge. |
| **Lc0 labeled block** | `data/processed/labeled/lc0_labeled.parquet` | Lc0-source block before merge. |
| **Label smokes** | `data/processed/labeled/smoke_labeled.parquet`, `chessbench_smoke10.parquet`, `lichess_smoke_labeled.parquet` | Tiny pipeline checks (startpos / 10 CB / 50 Lichess FENs). |
| **Lichess FENs (unlabeled)** | `data/processed/lichess/positions.parquet` | ~2.4k FENs from smoke50 PGN (pre-label schema). |
| **Lc0 FENs (unlabeled)** | `data/processed/lc0/positions.parquet` | Filtered Lc0 positions + optional `splits/`. |
| **ChessBench pilot splits** | `data/processed/chessbench/splits/{train,val}.parquet` | ~60k / 3k rows used to train `pilot_W128_*` (SF/ChessBench values — pilot only). |
| **Piece-count study** | `data/excel/piece_count_distribution.xlsx`, `piece_count_distribution_10k.xlsx` | Bucket design stats from Lichess-style games. |

---

## Evaluation artifacts (ACPL gates)

| Title | Path | Description |
| ----- | ---- | ----------- |
| **Cfish Hybrid d5** | `images/plots/PGN_and_JSON/cfish_hybrid_d5_gate_acpl.json` | Stock Cfish Hybrid, depth 5 self-play; ACPL **~42** / Elo heur. **~2435**. |
| **NNUE pilot d1** | `images/plots/PGN_and_JSON/nnue_d1_gate_acpl.json` | Student `pilot_W128_844` depth-1 gate (multi-game ladder). |
| **F3 single-head mini d1** | `images/plots/PGN_and_JSON/single_W128_mini_d1_gate_acpl.json` | First single-head student gate; ACPL **~583** / Elo floor **400** (below random). |
| **HCE d1** | `images/plots/PGN_and_JSON/hce_d1_gate_acpl.json` | Python handcrafted eval depth-1 baseline. |
| **Random d1** | `images/plots/PGN_and_JSON/random_d1_gate_acpl.json` | Untrained null floor; ACPL **~276** / Elo floor **400**. |
| **Sunfish / Lc0 gates** | `images/plots/PGN_and_JSON/sunfish_*_gate_acpl.json`, `images/plots/PGN_and_JSON/lc0_*_gate_acpl.json` | Other ladder points at d1/d2. |
| **Omar human games** | `images/plots/PGN_and_JSON/omar_game_acpl.json`, `images/plots/PGN_and_JSON/omar_game_2_3_acpl.json` | Per-side Stockfish ACPL on sample human PGNs under `images/games/`. |

(PGNs for many gates live next to the JSON files; `*.pgn` is gitignored.)

---

## Engine binary & demos

| Title | Path | Description |
| ----- | ---- | ----------- |
| **Cfish** | `src/cfish/cfish.exe` | UCI baseline tree (cwd must be `src/cfish/`). Launcher: `run-cfish.bat`. |
| **Self-play demos** | `images/games/`, `images/*_game.gif` | GIF/PGN reels (HCE, NNUE, Cfish, teachers). |

---
