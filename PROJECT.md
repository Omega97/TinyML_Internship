# KITSUNE (new version) + fresh data hand-off

> Scope: status of the **new** KITSUNE algorithm defined in `GOAL.md`, and what to
> do with the fresh training data in `data/processed/board_eval_2/`.

---

## 1. KITSUNE — new-version implementation status

Reference files:

- Base model + soft-CE loss: `src/tinymlinternship/nnue/model.py` (`DualHiddenNNUE`),
  `scripts/train_nnue-gpu-compact.py` (`ce_loss` @ `:222`, model @ `:665`).
- KITSUNE + MoE pipeline: `scripts/moe-training-UI/runner.py`
  (clustering stage @ `:1614`, `_micro_bucket_means` @ `:498`,
  `_kitsune_macro_cluster` @ `:524`, `_fine_tune_experts` @ `:807`).
- Router wrappers: `src/tinymlinternship/nnue/moe.py` (`KitsuneDispatcher` @ `:173`,
  `LinearDispatcher` @ `:70`).
- Sample gradients: `src/tinymlinternship/nnue/sample_gradients.py`
  (`sample_head_gradients_and_norms` returns the normalized direction and the raw ‖Δ‖).

| #   | GOAL.md requirement                                                                                                               | Status                                       | Notes / location                                                                                                                                                                                                            |
| --- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Train base NNUE (L1 `W`, L2 `H`, dual-POV, 3-way WDL) on **all** data with CE(student, teacher WDL)                               | 🟢 In place, ⚠️ not yet run on new data      | `DualHiddenNNUE` + `ce_loss` (soft CE on STM WDL). Existing checkpoints predate `board_eval_2`; base must be retrained on the new ~143M rows.                                                                               |
| 2   | Partition L1 into `B'` mini-clusters with **`B' ≈ 1000`**                                                                         | 🟠 Partial — mechanism exists, `B'` is wrong | `b_prime = k * micro_bucket_factor` (`runner.py:1618`). Default factor = 4 → `B'` tiny; max tested 128× (→ `B'=512`). `B'=1000` has **not** been run, and there is no absolute-`B'` parameter.                              |
| 3a  | Compute ASG (mean sample gradient) per mini-cluster                                                                               | 🟢 In place                                  | `_micro_bucket_means` (`runner.py:498`) averages the L2-normalized projected head gradients (direction only; magnitude is handled separately by step 5).                                                                    |
| 3b  | Quick-cluster the ASGs → `B` centroids                                                                                            | 🟢 In place                                  | `_kitsune_macro_cluster` stage A: spherical k-means on unit-norm prototypes (`runner.py:560`).                                                                                                                              |
| 3c  | Balanced stitch: iterate centroids round-robin, claim closest unused mini-cluster until `N·α`, then nearest-centroid for the rest | 🟢 In place                                  | `_kitsune_macro_cluster` stage B round-robin draft (`runner.py:570-586`), `α = kitsune_alpha`. Matches GOAL steps 1)-2).                                                                                                    |
| 4   | Fine-tune experts on their `B` partitions                                                                                         | 🟢 In place                                  | `_fine_tune_experts` (`runner.py:807`): per-partition fine-tune, L1 frozen, rprop/Adam/SGD, early stop.                                                                                                                     |
| 5   | **Gradient magnitude per partition** (before fine-tuning)                                                                         | 🟢 Done                                      | `compute_gradients` now writes `gradient_norms.npy`; `_gradient_magnitude_by_partition` (runner.py) aggregates mean/std/min/max per partition and writes `gradient_magnitude.json` + `gradient_magnitude_by_partition.png`. |
| 6   | **Contrast base-L1 clustering vs KITSUNE**                                                                                        | 🟠 Partial                                   | Both `clustering="l1"` and `"kitsune"` are runnable, but no artifact directly compares them (ARI/overlap, per-partition gradient stats, CE table).                                                                          |
| 7   | (Stretch) **Prune the dispatcher** (biggest part at `B'=1000`)                                                                    | 🔴 Not started                               | `KitsuneDispatcher` = `LinearDispatcher(in_dim → B')` + MAP table. At `B'=1000`, `in_dim=2W` → ~`2W·1000` params, which dwarfs the base head. No pruning exists.                                                            |

### 1a. What already works end-to-end

The two-stage KITSUNE pipeline is implemented and has produced results
(`RESULTS/KITSUNE/kitsune_moe.csv`, `RESULTS/moe/*_kitsune_*`): L1 micro-k-means →
mean gradient per micro-bucket → balanced-draft macro-clustering → linear
micro-dispatcher + `MAP[k']→b` → per-expert fine-tune. The balanced-draft
recombination matches the GOAL procedure exactly.

### 1b. The real gaps (what to build next)

1. **`B' ≈ 1000`.** Add an absolute micro-bucket count (or set
   `micro_bucket_factor` so `K · factor ≈ 1000`) and re-run. Watch two things:
   - `_kmeans_labels` fits on `max_fit_rows=5_000_000` (`runner.py:449`); with
     `B'=1000` a larger fit sample may be warranted.
   - Dispatcher output width becomes `1000`, so the linear dispatcher dominates
     parameter count (feeds stretch goal #7).
2. **L1 vs KITSUNE comparison.** Emit a single report (ARI/overlap + per-partition
   gradient stats + CE) contrasting the base-L1 partition against the KITSUNE
   recombination.

---

## 2. Fresh training data — `data/processed/board_eval_2/`

### Inventory

- **14 `.zip` files**, one per row-range group (`f_2M_5M.zip` … `f_49M-50M.zip`).
- **Full corpus (all zips): 254 slice folders, 142,842,055 rows (~143M positions).**
- **Unpacked so far: 3 groups (`f_2M_5M`, `f_5M_10M`, `f_10M_15M`) → 80 slices,
  38,917,999 positions.** The remaining 11 zips (`f_15M_20M.zip` … `f_49M-50M.zip`,
  ≈103.9M positions) are not yet extracted.
- Source: `fen_value_visits_lichess_db_standard_rated_2026-07_<start>-<end>_d95`
  (game ranges 2,000,000 → 50,000,000, `d95` filter).
- Each slice is **already encoded** and contains, inside the zip:
  - `features.npz` — sparse dual-POV 844 + STM WDL (train-ready)
  - `features.meta.json`
  - source `… .json` **and** `… .parquet`
- Count current positions with `python scripts/count_board_positions.py`.

### What to do

1. **Extract the remaining zips.** `f_2M_5M`, `f_5M_10M`, `f_10M_15M` are already
   unpacked directly under `data/processed/board_eval_2/`. Extract the rest the
   same way (the pipeline discovers slices recursively through the group folders,
   `discover_slice_folders`, `dataset.py:559`):

   ```bash
   for z in data/processed/board_eval_2/f_*M*.zip; do
     unzip -q "$z" -d data/processed/board_eval_2/
   done
   ```

   Then point the trainer at the `board_eval_2` root
   (`--slices-dir data/processed/board_eval_2`).

2. **Disk budget.** Full extraction ≈ **105 GB** (features.npz 75 GB + json/parquet
   ~30 GB). Training strictly needs `features.npz` + `features.meta.json` plus one
   source file (json/parquet) per slice for discovery — a lean extract is ~78 GB.

3. **No re-encoding needed.** The compact trainer
   (`train_nnue-gpu-compact.py` → `load_compact_split`) reads `features.npz`
   directly via mmap and reads `n_rows` from `features.meta.json`; it only
   re-encodes with `--rebuild-cache`. Note: the stored `features.meta.json`
   `"source"` path is an absolute path from the machine that produced the data,
   so `slice_db_is_valid` would *mismatch* and trigger a re-encode **only** in the
   `ensure_slice_feature_db`/rebuild path — avoid `--rebuild-cache` unless you
   actually want to re-encode.

4. **Retrain the base on all of it.** Run, e.g.:

   ```bash
   python3 scripts/train_nnue-gpu-compact.py \
     --slices-dir data/processed/board_eval_2 \
     --hidden-dim 64 --hidden2-dim 128 --epochs ... \
     --test-fraction 0.01
   ```

   This is the prerequisite for the KITSUNE pipeline, which currently loads a
   base checkpoint (`run_kitsune_headless.py` has a hardcoded
   `/home/omar/...` path that does not exist here — retrain and point it at the
   new checkpoint).

5. **Optional dedup check.** The `d95` slices share the 2026-07 dump; the existing
   pipeline already excludes overlapping holdout slices via
   `overlapping_dump_slices` (`dataset.py:77`), so the 1% test split is safe.

---
