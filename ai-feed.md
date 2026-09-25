# Thesis results battery

The method in `thesis-draft.md` chapter 3 is five steps on one frozen NNUE.

1. Train a base DualHidden NNUE on the full Lc0 WDL set (soft cross-entropy).
2. Compute L2-normalised per-sample gradients of that loss w.r.t. the head (L2 + output), and store a 48-d projection.
3. Cluster those gradients. Fixed B uses mini-batch k-means. Density uses DBSCAN.
4. Train a linear dispatcher from the frozen L1 activations (2W) to the bucket id.
5. Fine-tune one head per bucket, initialised from the base head, with L1 frozen.

Inference is one L1 pass, one dispatcher argmax, and one expert head. Elo and nodes-per-second stay out of this battery.

Run everything from `/home/omar/jupyterlab/TinyML_Internship` with `/home/omar/jupyterlab/venv/bin/python`.

## What is already in the repo

| Step | Where | State |
| --- | --- | --- |
| Base training | `scripts/train_nnue-gpu.py`, checkpoints under `models/checkpoints/nnue/` | Several widths are trained. Copies are in `RESULTS/base/`. |
| Sample gradients | `src/tinymlinternship/nnue/sample_gradients.py`, `moe_pipeline.compute_gradients` | Analytic head gradients, L2-normalised, projected to 48-d float16. A 2,000,000-row cache for the reference net is `data/processed/board_eval/moe/moe_b4_2m/`. |
| Mini-batch k-means | `nnue/cluster.py` `fit_minibatch_kmeans` | Used by the training pipeline. |
| DBSCAN | `nnue/cluster.py` `fit_dbscan` / `fit_dbscan_for_b` | Training path. ε is a quantile of 8-NN distances on the same 0.1 grid as the explorer. The fit uses at most 100,000 rows; every training row is then assigned to the nearest core centroid. Noise is counted, then assigned the same way. |
| Dispatcher | `nnue/moe.py` `LinearDispatcher`, `moe_pipeline.train_dispatcher` | Adam, lr 1e-2 → 1e-3, 8 epochs, 10% validation, best checkpoint restored. |
| Expert heads | `moe_pipeline.fine_tune_experts` | L1 frozen, head copied from the base, soft CE, default 2 epochs, 10% bucket holdout. `--expert-labels dispatcher` or `kmeans`. |
| Scores and plots | `evaluate_moe`, `moe_plots.py` | Global test CE and MAE for the base, the dispatched MoE, and an oracle that routes with the gradient centroid. Per-bucket bar of base-head CE vs expert CE on that bucket's holdout. Cluster size, centroid cosine, PCA, t-SNE, dispatcher accuracy, confusion. |
| Explorer | `cluster_explorer_ui.py` | Visual only. Mini-batch k-means, k-medoids, DBSCAN, OPTICS. It does not train experts. |

Two finished MoE runs already exist, both on the reference net `W128_H256` (`models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt`, which is 100 epochs in `config.json`). Copies, including `moe.pt` and the plots, are under `RESULTS/moe/preliminary/`.

| Run | Rows | B | Test CE base | Test CE MoE | Test CE oracle | Dispatcher val acc | Mean bucket-holdout base → expert |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kmeans_b3_1m` | 1,000,000 | 3 | 0.63019 | 0.63041 | 0.62831 | 0.600 | 0.63152 → 0.63172 |
| `kmeans_b4_2m` | 2,000,000 | 4 | 0.63019 | 0.63029 | 0.62843 | 0.556 | 0.63075 → 0.63087 |

On the B=4 run the four bucket holdouts fall into two pairs: experts 0 and 2 sit near CE 0.35, experts 1 and 3 near CE 0.94. The partition separates positions the base already fits from positions it does not. Two epochs of fine-tuning leave the expert within 0.001 of the base head on every bucket (expert 3 is the only one slightly lower: 0.94083 vs 0.94100). The oracle, which routes with the true gradient cluster, is about 0.002 CE better than the base on the 50,000-row test split. The learned dispatcher gives that gain back.

Full-precision rows are `RESULTS/tables/ce.csv`.

## Features for the battery

Done, and used by the commands below:

- `--algorithm minibatch_kmeans|dbscan` and `--dbscan-epsilon` on `scripts/run_moe_pipeline.py` and `scripts/cluster_gradients.py`. When ε is omitted, DBSCAN searches `{0.1 … 0.9}` for the count closest to `--n-clusters`. The chosen ε, the trial table, and whether B matched are written to `diagnostics.json`.
- `--gradient-cache` symlinks an existing `gradients.npy` / `train_pack.pt` / `meta.json` / `slice_ids.npy` / `local_rows.npy` into the run directory so B and the algorithm can change without a second gradient pass.
- `scripts/stage_thesis_results.py` copies base checkpoints and the two finished MoE runs into `RESULTS/`.
- `scripts/run_thesis_battery.py` prints or runs the grid and rebuilds `RESULTS/tables/ce.csv`, `RESULTS/plots/test_ce.png`, `RESULTS/plots/bucket_holdout_ce.png`, and `RESULTS/plots/bucket_ce_bars.png`.

Still to build. None of these block wave 1. They are the follow-ups once the grid exists.

- Silhouette on a subsample of the projected gradients, stored next to inertia and centroid cosine in `diagnostics.json`. Section 3.5.5 asks for it. Inertia per run is already a column in `ce.csv`; a single elbow chart across B is a short plot on top of that column.
- Per-bucket CE on the global test split. The bar the battery writes today (`expert_ce_by_bucket.png`) is the 10% holdout inside each expert's training bucket, which is that expert's own data. `moe_vs_base_ce.png` is one number for the whole 50,000-row test set. A third figure would split that test set by the dispatcher bucket and score the base head and the expert on each slice.
- Density-peak clustering (Rodriguez & Laio), which section 3.5.3 discusses. This battery uses mini-batch k-means and DBSCAN.
- OPTICS in the training pipeline. It remains an explorer option.
- A full-data gradient pass. The reference cache is 2,000,000 rows. The W128/H256 scaling curve in `RESULTS/base/scaling/` was trained out to about 90,000,000 positions. Raise `--max-rows` only after the 2M grid shows the MoE moving CE.
- Quantization, Cfish nodes-per-second, and Elo. Separate from this CE battery.

## Schedule

Wave 0 is done. Base nets and the two preliminary MoE runs are in `RESULTS/`, with the comparison bars in `RESULTS/plots/`.

Wave 1, next, on the reference net only. Six cells, shared 2M gradient cache, dispatcher labels, two expert epochs: mini-batch k-means at B = 2, 4, 8 and DBSCAN targeted at the same three B values. The existing 2M dispatcher epochs took about 10 seconds each. The new cost per cell is clustering, eight dispatcher epochs, and two expert epochs. DBSCAN repeats a 100k-row ε search in each of its three cells. Plan on an afternoon, then read `RESULTS/plots/bucket_ce_bars.png` and `RESULTS/tables/ce.csv`.

Wave 2, after wave 1, still on the reference cache. For k-means B = 2, 4, 8: one repeat that fine-tunes experts on the cluster labels (`--expert-labels kmeans`), and one repeat with eight expert epochs. Run the B values where wave 1 is flat or where the oracle beats the base. The question is whether the flat holdout is the two-epoch budget or the dispatcher relabeling.

Wave 3, after the reference grid has a direction. One k-means B=4 probe on each other width, each with its own 2M gradient pass: `W32_H64`, `W64_H128`, `W128_H128`, `W256_H512`. If a width moves test CE, clone the wave 1 commands onto that checkpoint and drop `--gradient-cache`.

Elo after the CE table is stable.

## Commands

Refresh the staged copies and the summary plots:

```bash
cd /home/omar/jupyterlab/TinyML_Internship
/home/omar/jupyterlab/venv/bin/python scripts/stage_thesis_results.py
/home/omar/jupyterlab/venv/bin/python scripts/run_thesis_battery.py --summary
```

Print the grid again after any edit to `results_battery.py`:

```bash
/home/omar/jupyterlab/venv/bin/python scripts/run_thesis_battery.py --print
```

Run a wave as one process. Wave 1 is the default of `--run`. The script links the gradient cache, then trains.

```bash
/home/omar/jupyterlab/venv/bin/python scripts/run_thesis_battery.py --run --wave 1
/home/omar/jupyterlab/venv/bin/python scripts/run_thesis_battery.py --run --wave 2
/home/omar/jupyterlab/venv/bin/python scripts/run_thesis_battery.py --run --wave 3
```

The same cells one at a time. Each command writes `moe.pt`, `dispatcher.pt`, `eval.json`, `expert_metrics.json`, `diagnostics.json`, and `plots/` under its `--work-dir`.

### Wave 1 — reference net, B = 2, 4, 8

```bash
/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/kmeans_b2 \
  --plots-dir RESULTS/moe/W128_H256/kmeans_b2/plots \
  --run-name kmeans_b2 --max-rows 2000000 --max-test 50000 \
  --n-clusters 2 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/kmeans_b4 \
  --plots-dir RESULTS/moe/W128_H256/kmeans_b4/plots \
  --run-name kmeans_b4 --max-rows 2000000 --max-test 50000 \
  --n-clusters 4 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/kmeans_b8 \
  --plots-dir RESULTS/moe/W128_H256/kmeans_b8/plots \
  --run-name kmeans_b8 --max-rows 2000000 --max-test 50000 \
  --n-clusters 8 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/dbscan_b2 \
  --plots-dir RESULTS/moe/W128_H256/dbscan_b2/plots \
  --run-name dbscan_b2 --max-rows 2000000 --max-test 50000 \
  --n-clusters 2 --algorithm dbscan \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/dbscan_b4 \
  --plots-dir RESULTS/moe/W128_H256/dbscan_b4/plots \
  --run-name dbscan_b4 --max-rows 2000000 --max-test 50000 \
  --n-clusters 4 --algorithm dbscan \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/dbscan_b8 \
  --plots-dir RESULTS/moe/W128_H256/dbscan_b8/plots \
  --run-name dbscan_b8 --max-rows 2000000 --max-test 50000 \
  --n-clusters 8 --algorithm dbscan \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m
```

To pin DBSCAN to one ε quantile instead of searching for B, add `--dbscan-epsilon 0.3` (values 0.1 through 0.9). The resulting B is whatever that density produces, and the dispatcher is sized to it.

### Wave 2 — label source, then a longer fine-tune

```bash
/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/kmeans_b2_clusterlabels \
  --plots-dir RESULTS/moe/W128_H256/kmeans_b2_clusterlabels/plots \
  --run-name kmeans_b2_clusterlabels --max-rows 2000000 --max-test 50000 \
  --n-clusters 2 --algorithm minibatch_kmeans \
  --expert-labels kmeans --expert-epochs 2 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/kmeans_b4_clusterlabels \
  --plots-dir RESULTS/moe/W128_H256/kmeans_b4_clusterlabels/plots \
  --run-name kmeans_b4_clusterlabels --max-rows 2000000 --max-test 50000 \
  --n-clusters 4 --algorithm minibatch_kmeans \
  --expert-labels kmeans --expert-epochs 2 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/kmeans_b8_clusterlabels \
  --plots-dir RESULTS/moe/W128_H256/kmeans_b8_clusterlabels/plots \
  --run-name kmeans_b8_clusterlabels --max-rows 2000000 --max-test 50000 \
  --n-clusters 8 --algorithm minibatch_kmeans \
  --expert-labels kmeans --expert-epochs 2 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/kmeans_b2_e8 \
  --plots-dir RESULTS/moe/W128_H256/kmeans_b2_e8/plots \
  --run-name kmeans_b2_e8 --max-rows 2000000 --max-test 50000 \
  --n-clusters 2 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 8 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/kmeans_b4_e8 \
  --plots-dir RESULTS/moe/W128_H256/kmeans_b4_e8/plots \
  --run-name kmeans_b4_e8 --max-rows 2000000 --max-test 50000 \
  --n-clusters 4 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 8 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H256/kmeans_b8_e8 \
  --plots-dir RESULTS/moe/W128_H256/kmeans_b8_e8/plots \
  --run-name kmeans_b8_e8 --max-rows 2000000 --max-test 50000 \
  --n-clusters 8 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 8 --dispatcher-epochs 8 \
  --gradient-cache data/processed/board_eval/moe/moe_b4_2m
```

### Wave 3 — one B=4 probe per other width

These compute a fresh gradient pack. There is no `--gradient-cache`.

```bash
/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H256_e100_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W32_H64/kmeans_b4 \
  --plots-dir RESULTS/moe/W32_H64/kmeans_b4/plots \
  --run-name kmeans_b4 --max-rows 2000000 --max-test 50000 \
  --n-clusters 4 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H128_512x8198_e1000/best.pt \
  --work-dir RESULTS/moe/W64_H128/kmeans_b4 \
  --plots-dir RESULTS/moe/W64_H128/kmeans_b4/plots \
  --run-name kmeans_b4 --max-rows 2000000 --max-test 50000 \
  --n-clusters 4 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h128_H128_e100_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W128_H128/kmeans_b4 \
  --plots-dir RESULTS/moe/W128_H128/kmeans_b4/plots \
  --run-name kmeans_b4 --max-rows 2000000 --max-test 50000 \
  --n-clusters 4 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8

/home/omar/jupyterlab/venv/bin/python scripts/run_moe_pipeline.py \
  --checkpoint models/checkpoints/nnue/dual_h256_H512_e200_bpe512_bs10000/best.pt \
  --work-dir RESULTS/moe/W256_H512/kmeans_b4 \
  --plots-dir RESULTS/moe/W256_H512/kmeans_b4/plots \
  --run-name kmeans_b4 --max-rows 2000000 --max-test 50000 \
  --n-clusters 4 --algorithm minibatch_kmeans \
  --expert-labels dispatcher --expert-epochs 2 --dispatcher-epochs 8
```

After any cell:

```bash
/home/omar/jupyterlab/venv/bin/python scripts/run_thesis_battery.py --summary
```

## Results produced

Checkpoint directory names and the real widths disagree. The names below are the widths in `config.json`. Each copy is `best.pt`, `config.json`, `history.json`, `ce.png`, and `source.json`.

| RESULTS path | Source folder | W | H | Params | Epochs | Best test CE |
| --- | --- | --- | --- | --- | --- | --- |
| `RESULTS/base/W32_H64/` | `dual_h128_H256_e100_bpe512_bs10000` | 32 | 64 | 31,395 | 100 | 0.6555 |
| `RESULTS/base/W64_H128/` | `dual_h128_H128_512x8198_e1000` | 64 | 128 | 70,979 | 100 | 0.6725 |
| `RESULTS/base/W128_H128/` | `dual_h128_H128_e100_bpe512_bs10000` | 128 | 128 | 141,443 | 100 | 0.6387 |
| `RESULTS/base/W128_H256/` | `dual_h128_H256_e200_bpe512_bs10000` | 128 | 256 | 174,723 | 100 | 0.6315 |
| `RESULTS/base/W256_H512/` | `dual_h256_H512_e200_bpe512_bs10000` | 256 | 512 | 480,515 | 200 | 0.6289 |

`W64_H128` used a different batch schedule (`512 × 8198`, test fraction 0.05), so its higher CE is not a pure width comparison. The other four share batches-per-epoch 512 and batch size 10,000. `W256_H512` ran 200 epochs and a 5% test split.

Scaling curves for the reference width, already trained, are in `RESULTS/base/scaling/` (`variable_dataset_size_ce_256.txt` reaches test CE 0.6307 at about 90,000,000 positions, which matches the reference checkpoint).

Preliminary MoE, mini-batch k-means, dispatcher labels, two expert epochs:

- `RESULTS/moe/preliminary/kmeans_b3_1m/` — `moe.pt`, dispatcher, metrics, and the eight plots (PCA, t-SNE, sizes, centroid cosine, dispatcher accuracy, confusion, per-bucket CE, base vs MoE vs oracle).
- `RESULTS/moe/preliminary/kmeans_b4_2m/` — the same files on the 2,000,000-row cache.
- `RESULTS/plots/test_ce.png` — test CE, base vs MoE vs oracle, both runs.
- `RESULTS/plots/bucket_holdout_ce.png` — mean holdout CE, base head vs expert.
- `RESULTS/plots/bucket_ce_bars.png` — one pair of bars per bucket (the per-dataset comparison).
- `RESULTS/tables/ce.csv` and `RESULTS/manifest.json`.

Gradient tensors stay in `data/processed/board_eval/moe/`. They are not duplicated into `RESULTS/`.

## Results still to produce

Each future cell gets the same bundle as the preliminary runs: `moe.pt`, `dispatcher.pt`, `eval.json`, `expert_metrics.json`, `diagnostics.json`, `summary.json`, and `plots/expert_ce_by_bucket.png` plus `plots/moe_vs_base_ce.png`. `--summary` folds them into `ce.csv` and the three comparison figures.

| Directory | What it answers |
| --- | --- |
| `RESULTS/moe/W128_H256/kmeans_b{2,4,8}/` | Does fixed-B k-means at the thesis grid beat the base, on the test set and on each bucket? |
| `RESULTS/moe/W128_H256/dbscan_b{2,4,8}/` | Same question for DBSCAN. `diagnostics.json` records the ε that was selected and the B it actually found. |
| `RESULTS/moe/W128_H256/kmeans_b{2,4,8}_clusterlabels/` | Experts trained on the k-means ids rather than the dispatcher ids. |
| `RESULTS/moe/W128_H256/kmeans_b{2,4,8}_e8/` | Same partition, eight expert epochs. |
| `RESULTS/moe/W32_H64/kmeans_b4/` and the matching dirs for `W64_H128`, `W128_H128`, `W256_H512` | Whether a single B=4 MoE moves CE at other widths. |
| Later, same folders with a larger `--max-rows` | Whether the 2M subsample was the limit. |
| Elo, left out of this folder until the CE table is in | Playing strength of the base net against the chosen MoE. |
