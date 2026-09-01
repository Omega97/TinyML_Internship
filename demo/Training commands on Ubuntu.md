
# Commands (Ubuntu / Nvidia Spark)

Mirror of [Training commands.md](Training%20commands.md) for Linux. Use `python3.12` (not `py -3.12`). Run from the **repo root**.

```bash
cd ~/jupyterlab/TinyML_Internship
# or: cd ~/TinyML_Internship
source ~/jupyterlab/venv/bin/activate   # if the venv lives next to the repo
python3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

Long jobs: `tmux new -s train` (detach `Ctrl-b d`, reattach `tmux a -t train`).

**Do not use `--test <slice>`.** Current trainers take `--test-fraction` (default `0.10`). Per-epoch 3 200-row CE is noisy; prefer `--test-subset-size 10000` (or 50k) and read `matched 50k-row CE` at the end. Unique `--run-name` so you do not overwrite `best.pt`.

First `train_*.py` after copying slices from Windows **re-encodes** every folder (`encode 844` bars): `features.meta.json` stores a Windows absolute path. After that, re-runs skip encode. Or encode once:

```bash
python3.12 -u scripts/encode_slice_features.py
python3.12 -u scripts/train_nnue.py --encode-only
```

`--rebuild` / `--rebuild-cache` forces a full re-encode. Do not pass them on a normal train.

Lc0 for dump/relabel: `settings.py` still points at `models/teacher/lc0/lc0.exe`. On Spark put a Linux ARM `lc0` there (copy or symlink as `lc0.exe`) plus `791556.pb.gz`. `scripts/download_teacher.py` fetches the **Windows** zip — skip it on aarch64; `--network-only` is fine for the net.

---

## Relabel values with WDL (variable depth)

```bash
python3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_mini --depth 0 --in-place

python3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_kaggle_10k --depth 0 --in-place

python3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_puzzles --depth 1 --in-place
```

---

## Convert to npz

```bash
python3.12 -u scripts/encode_slice_features.py

python3.12 -u scripts/encode_slice_features.py --rebuild

python3.12 -u scripts/encode_slice_features.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_60000-65000_d90 --rebuild

python3.12 -u scripts/encode_slice_features.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_65000-70000_d90 --rebuild
```

---

## Improve depth 🐳

Needs a working Linux `lc0` (see header). Default `--depth 1` is value-head-ish; `--depth 2` is slower and better for puzzles / hard slices.

```bash
python3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_puzzles --depth 2 --in-place

python3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_15000-20000_d90 --depth 2 --in-place

python3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_20000-25000_d90 --depth 2 --in-place

python3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_25000-30000_d90 --depth 2 --in-place
```

After an in-place JSON change, re-encode that slice:

```bash
python3.12 -u scripts/encode_slice_features.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_puzzles --rebuild
```

---

## Download dump (if missing)

Keep the `.pgn.zst` compressed (~27 GiB). Do not inflate.

```bash
python3.12 -u scripts/download_lichess_dump.py --month 2026-07
```

---

## Build Dataset - Generate Data 🧱 (sparse input, WDL + npz)

`n` included, `m` excluded. Default `--dropout 0.90` → stem `…_d90`. Writes JSON + parquet + `features.npz` (skip npz with `--skip-encode`). Labeling needs Lc0.

Smoke:

```bash
python3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10
python3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10 --dropout 0
python3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10 --max-draw 0.30
```

Grow the dump (same ranges as the Windows cheatsheet):

```bash
python3.12 -u scripts/lichess_dump_to_fen_value_visits.py 1010000 1020000 --dropout 0.95
python3.12 -u scripts/lichess_dump_to_fen_value_visits.py 1020000 1030000 --dropout 0.95
python3.12 -u scripts/lichess_dump_to_fen_value_visits.py 1030000 1040000 --dropout 0.95
python3.12 -u scripts/lichess_dump_to_fen_value_visits.py 1040000 1050000 --dropout 0.95
echo done
```

Count unique EPDs (no join write):

```bash
python3.12 -u scripts/count_fen_value_visits.py
python3.12 -u scripts/count_fen_value_visits.py --quiet
```

Join all slices:

```bash
python3.12 -u scripts/join_fen_value_visits.py
```

---

## Database of extreme positions (P(draw) <= 5%)

```bash
python3.12 -u scripts/lichess_dump_to_fen_value_visits.py 100000 105000 --max-draw 0.05 --dropout 0.9
```

Openings only:

```bash
python3.12 -u scripts/lichess_dump_to_fen_value_visits.py 100000 105000 --dropout 0 --max-moves 10
```

---

## Training of the mini model (linear + softmax, no hidden layers)

```bash
# smoke
python3.12 -u scripts/train_linear_wdl.py --epochs 10 --smoke --run-name linear_wdl_smoke --plot plots/linear_wdl_smoke_ce.png

python3.12 -u scripts/train_linear_wdl.py --epochs 100 --smoke --lr 1e-2 --run-name linear_wdl_smoke --plot plots/linear_wdl_smoke_ce.png

# short
python3.12 -u scripts/train_linear_wdl.py --epochs 10 --lr 0.01 --run-name linear_wdl --plot plots/linear_wdl_ce.png

# fraction split + linear decay
python3.12 -u scripts/train_linear_wdl.py --epochs 50 --lr 0.01 --lr-end 0.001 --batch-size 2048 --batches-per-epoch 32 --test-fraction 0.10 --test-subset-size 3200 --run-name linear_wdl_frac10_fast --plot plots/linear_wdl_frac10_fast_ce.png
```

---

## Inspect best and worst guesses

```bash
python3.12 -u scripts/inspect_nnue_positions.py linear_wdl_smoke --ckpt best --n 20 --sample 1000 --seed 0 --slice fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90

python3.12 -u scripts/inspect_nnue_positions.py dual_h128_H256_spark_32x4096 --ckpt best --n 20 --sample 1000 --seed 0
```

---

## Faster Training Small Model

```bash
python3.12 -u scripts/train_linear_wdl.py --epochs 100 --lr 0.01 --run-name linear_wdl_spark --plot plots/linear_wdl_ce.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast
```

---

## Faster Training Medium Model

```bash
python3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 32 --run-name medium_h32_fast --plot plots/medium_h32_fast_ce.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast

python3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 64 --run-name medium_h64_fast --plot plots/medium_h64_fast_ce.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast

python3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 128 --run-name medium_h128_fast --plot plots/medium_h128_fast_ce.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast
```

---

## Faster Training Dual Model 🧠

`--fast` = `--batch-size 256 --batches-per-epoch 40` (~10k rows/epoch). Fine for a pipeline check; on Spark prefer the larger-batch block below.

```bash
python3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 16 --hidden2-dim 32 --run-name dual_h16_H32_fast --plot plots/dual_nnue_ce_16_32.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast

python3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 32 --hidden2-dim 64 --run-name dual_h32_H64_fast --plot plots/dual_nnue_ce_32_64.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast

python3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 64 --hidden2-dim 128 --run-name dual_h64_H128_fast --plot plots/dual_nnue_ce_64_128.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast

python3.12 -u scripts/train_nnue.py --epochs 200 --lr 0.01 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_fast --plot plots/dual_nnue_ce_128_256.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast

python3.12 -u scripts/train_nnue.py --epochs 200 --lr 0.01 --hidden-dim 256 --hidden2-dim 512 --run-name dual_h256_H512_fast --plot plots/dual_nnue_ce_256_512.png --test-fraction 0.10 --test-subset-size 5000 --train-val-subset-size 5000 --fast
```

---

## Run ALL (small ladder)

```bash
python3.12 -u scripts/train_linear_wdl.py --epochs 100 --lr 0.01 --run-name linear_wdl_spark --plot plots/linear_wdl_ce.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast

python3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 32 --run-name medium_h32_fast --plot plots/medium_h32_fast_ce.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast

python3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 16 --hidden2-dim 32 --run-name dual_h16_H32_fast --plot plots/dual_nnue_ce_16_32.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast
```

---

## Baseline Compare to FFNN

```bash
python3.12 -u scripts/train_ffnn.py --epochs 50 --lr 0.001 --hidden1 256 --hidden2 256 --run-name ffnn_baseline --test-fraction 0.10 --fast
```

---

## With scheduler

```bash
python3.12 -u scripts/train_nnue.py --epochs 300 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_sched300 --plot plots/dual_nnue_ce_128_256_scheduler_300_epoch.png --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --fast

python3.12 -u scripts/train_nnue.py --epochs 1000 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_sched1000 --plot plots/dual_nnue_ce_128_256_scheduler_1000_epoch.png --test-fraction 0.10 --test-subset-size 3200 --train-val-subset-size 3200 --batches-per-epoch 40 --batch-size 256
```

---

## NNUE Spark batch size ⭐️

Default `--fast` is 256 × 40. On Spark use fewer, larger steps. Rows/epoch = `batches-per-epoch × batch-size`.

```bash
# 32 × 1024 = 32 768 rows/epoch
python3.12 -u scripts/train_nnue.py --epochs 300 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_spark_32x1024 --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --batches-per-epoch 32 --batch-size 1024 --plot plots/dual_nnue_ce_128_256_scheduler_32x1024.png

# 32 × 2048 = 65 536
python3.12 -u scripts/train_nnue.py --epochs 300 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_spark_32x2048 --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --batches-per-epoch 32 --batch-size 2048 --plot plots/dual_nnue_ce_128_256_scheduler_32x2048.png

# 32 × 4096 = 131 072  (current Spark default)
python3.12 -u scripts/train_nnue.py --epochs 300 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_spark_32x4096 --test-fraction 0.10 --test-subset-size 10000 --train-val-subset-size 10000 --batches-per-epoch 32 --batch-size 4096 --plot plots/dual_nnue_ce_128_256_scheduler_32_bps_bs_4096.png
```

Optional: `--device cuda` (auto if a GPU is visible), `--compile` (`torch.compile`, off by default).

Checkpoints: `models/checkpoints/nnue/<run-name>/{best.pt,last.pt,history.json,config.json,ce.png}`. Plot also under `plots/`.

---

## Bundle npz (copy / zip slice DBs)

```bash
python3.12 -u scripts/bundle_slice_npz.py
python3.12 -u scripts/bundle_slice_npz.py --no-copy
python3.12 -u scripts/bundle_slice_npz.py --clean
```
