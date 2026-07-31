
# Chess dataset map (positions + evaluations)

Slim index of paths for FEN positions, teacher labels (`expected_reward`), and the scripts that produce them.  
Canonical layout and label rules: [ASSETS.md](ASSETS.md) · survey: [NOTES/Datasets.md](NOTES/Datasets.md).

⭐ = production-critical (primary train path or schema contract)

---

## Folders

| Path                                                                                 | Role                                                                                       |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| [data/](data/)                                                                       | Dataset root (`raw/` + `processed/`)                                                       |
| [data/raw/](data/raw/)                                                               | Downloads / immutable inputs                                                               |
| ⭐ [data/raw/lc0/](data/raw/lc0/)                                                     | Lc0 training tars, extracted `.gz` chunks, `manifest.json`                                 |
| [data/raw/chessbench/](data/raw/chessbench/)                                         | ChessBench Research `.bag` (smoke / pilot only)                                            |
| [data/raw/games.csv](data/raw/games.csv)                                             | Kaggle Lichess sample games (bucket stats only, not NNUE train)                            |
| [data/raw/lichess_smoke50.pgn](data/raw/lichess_smoke50.pgn)                         | Small PGN smoke for PGN→FEN path                                                           |
| [data/processed/](data/processed/)                                                   | FENs + labeled train products                                                              |
| ⭐ [data/processed/lc0/](data/processed/lc0/)                                         | Unlabeled Lc0 positions (`positions.parquet`, splits, stats)                               |
| ⭐ [data/processed/lichess/](data/processed/lichess/)                                 | Unlabeled Lichess / PGN FENs (`positions.parquet`, smoke parquets)                         |
| [data/processed/chessbench/](data/processed/chessbench/)                             | ChessBench pilot parquet (smoke wiring only)                                               |
| ⭐ [data/processed/labeled/](data/processed/labeled/)                                 | **Train here** — labeled blocks + merged `train.parquet` / `val.parquet` / `manifest.json` |
| [data/excel/](data/excel/)                                                           | Piece-count / bucket distribution spreadsheets                                             |
| ⭐ [src/tinymlinternship/data/](src/tinymlinternship/data/)                           | Library: schema, Lc0 parser/preprocess, ChessBench bag reader                              |
| ⭐ [src/tinymlinternship/nnue/dataset.py](src/tinymlinternship/nnue/dataset.py)       | PyTorch loader: parquet → dual-844 features + `expected_reward`                            |
| [src/tinymlinternship/config/settings.py](src/tinymlinternship/config/settings.py)   | Path constants (`RAW_DATA_DIR`, `LC0_*`, `CHESSBENCH_*`, …)                                |
| ⭐ [src/tinymlinternship/features/bucket.py](src/tinymlinternship/features/bucket.py) | `bucket_id` / piece-count / queen-split metadata on positions                              |

### Library modules (`src/tinymlinternship/data/`)

| Module | Role |
| ------ | ---- |
| ⭐ [schema.py](src/tinymlinternship/data/schema.py) | Pre-label / labeled column contract, manifests, game-id split |
| [lc0_shards.py](src/tinymlinternship/data/lc0_shards.py) | Curated Lc0 shard list (download targets) |
| ⭐ [lc0_parser.py](src/tinymlinternship/data/lc0_parser.py) | Protobuf V6 chunk → board / FEN |
| ⭐ [lc0_preprocess.py](src/tinymlinternship/data/lc0_preprocess.py) | Filter, sample, dataframe export for Lc0 positions |
| [chessbench_preprocess.py](src/tinymlinternship/data/chessbench_preprocess.py) | `.bag` state-value records → training rows |

---

## Scripts that download / load data

| Script | Role |
| ------ | ---- |
| ⭐ [scripts/download_lc0.py](scripts/download_lc0.py) | Download + extract Lc0 training tars → `data/raw/lc0/` |
| [scripts/download_chessbench.py](scripts/download_chessbench.py) | Download ChessBench test `.bag` → `data/raw/chessbench/` |
| [scripts/download_data.py](scripts/download_data.py) | Download Kaggle `datasnaek/chess` → `data/raw/games.csv` |
| ⭐ [scripts/download_teacher.py](scripts/download_teacher.py) | Lc0 binary + networks for labeling (not position data) |
| [scripts/smoke_test_lc0_chunk.py](scripts/smoke_test_lc0_chunk.py) | Load one Lc0 `.gz` chunk and validate FENs |
| [scripts/study_chessbench.py](scripts/study_chessbench.py) | Inspect ChessBench `.bag` records / value semantics |
| [scripts/stats_lc0_processed.py](scripts/stats_lc0_processed.py) | Bucket / ply survival stats over raw Lc0 chunks |
| [scripts/plot_piece_count_distribution.py](scripts/plot_piece_count_distribution.py) | Load `games.csv` positions → piece-count distribution plots/xlsx |
| ⭐ [scripts/train_nnue.py](scripts/train_nnue.py) | Train loop that **loads** labeled parquet via `ChessbenchDataset` |

Related tests (load / schema checks):  
[tests/test_data.py](tests/test_data.py) · [tests/test_dataset_schema.py](tests/test_dataset_schema.py) · [tests/test_lc0_parser.py](tests/test_lc0_parser.py) · [tests/test_download_lc0.py](tests/test_download_lc0.py) · [tests/test_chessbench_preprocess.py](tests/test_chessbench_preprocess.py) · [tests/test_lc0_preprocess.py](tests/test_lc0_preprocess.py)

---

## Scripts that edit / transform data

| Script | Role |
| ------ | ---- |
| ⭐ [scripts/prepare_lc0_dataset.py](scripts/prepare_lc0_dataset.py) | Lc0 chunks → filter / sample → `data/processed/lc0/positions.parquet` (+ splits) |
| ⭐ [scripts/lichess_pgn_to_fen.py](scripts/lichess_pgn_to_fen.py) | Stream PGN → FEN + `bucket_id` metadata → `data/processed/lichess/` |
| [scripts/prepare_chessbench_dataset.py](scripts/prepare_chessbench_dataset.py) | ChessBench `state_value` bag → pilot labeled parquet under `processed/chessbench/` |
| ⭐ [scripts/label_positions.py](scripts/label_positions.py) | Add Lc0 WDL → White-POV `expected_reward` (+ WDL columns) on FEN parquet |
| ⭐ [scripts/merge_training_sets.py](scripts/merge_training_sets.py) | Merge labeled blocks → `train.parquet` / `val.parquet` / `manifest.json` (split by `game_id`) |

### Typical production flow

```text
download_lc0.py / (Lichess PGN on disk)
        │
        ▼
prepare_lc0_dataset.py     lichess_pgn_to_fen.py
        │                           │
        └───────────┬───────────────┘
                    ▼
            label_positions.py      ← uniform expected_reward
                    ▼
          merge_training_sets.py
                    ▼
     data/processed/labeled/{train,val}.parquet
                    ▼
              train_nnue.py
```

Smoke / non-production: `download_chessbench.py` → `prepare_chessbench_dataset.py` (pilot wiring only).
