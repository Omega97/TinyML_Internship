# AI feed — SARDINE code map

_Snapshot: 2026-07-22 (post-reassessment). Progress: [TODOs.md](TODOs.md) · status: [PROJECT.md](PROJECT.md) · spec: [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md)._

---

## Progress (A–J)

```text
A  Feature encoder (PC)     DONE (844-dim; 8 buckets interim; device parity open)
B  Search skeleton (PC)     PARTIAL  v0.3 (αβ + qsearch + MVV-LVA)
C  Train bucketed NNUE      MINI SET  smoke prod path; full volume + nnue-pytorch open
D  Bucket ablation          NOT STARTED  (locks 8 vs 4 / peers)
E–F Device / C port         NOT STARTED
G  Full search stack        qsearch + MVV-LVA only
H  Elo gate ≥1700           ACPL heuristic only
```

**Next bottleneck:** longer mini train → full Lichess volume → nnue-pytorch / prune → TT + search polish.

**Tests:** 106 passed (2026-07-20). Checkpoints: `models/checkpoints/nnue/pilot_W128_844/`, `smoke_prod_W128_844/`.

**Buckets:** code = **8** queen-split until §D (G3). Blueprint lists 4 piece-count as ablation baseline.

---

## Packages

| Path | Role |
| ---- | ---- |
| `src/.../features/` | 844 encoder, `bucket_id`, tactical |
| `src/.../data/` | Lc0 / ChessBench preprocess, schema |
| `src/.../nnue/` | BucketedNNUE + dataset |
| `src/.../engine/` | search, HCE, NNUE, Lc0, perft |
| `src/.../bot_eval/` | ACPL (Stockfish via PATH) |
| `src/.../visualization/` | pygame / GIF |
| `scripts/` | CLI: label, merge, train, eval, record |
| `models/teacher/` | Lc0 + nets (HF weights not shipped) |
| `data/raw|processed/` | Datasets (gitignored bulk) |

**Removed (2026-07-22):** `legacy/pre-sardine/`, in-repo Stockfish tree, empty `core/` / `evaluation/` / `datasets/` / `models/` packages.

---

## Critical scripts

| Script | Role |
| ------ | ---- |
| `lichess_pgn_to_fen.py` / `prepare_lc0_dataset.py` | FEN extract |
| `label_positions.py` | Lc0 → `expected_reward` |
| `merge_training_sets.py` | train/val + manifest |
| `train_nnue.py` | PyTorch smoke / mini train |
| `run_engine.py` / `record_engine_game.py` | Search + GIF |
| `eval_bot_acpl.py` | ACPL heuristic Elo |
| `prepare_chessbench_dataset.py` | Smoke only |

Label rule: train on **`expected_reward` only** ([ASSETS.md](ASSETS.md)).

---

## Daily notes

Local only: `DAILY-NOTES/YYYY-MM/*.md` (**gitignored**). Do not expect them on remote.
