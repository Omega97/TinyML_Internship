
# Commands

```powershell
cd C:\Users\monfalcone\PycharmProjects\TinyMLInternship
```


## Relabel values with WDL (variable depth)

```powershell
py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_mini --depth 0 --in-place

py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_kaggle_10k --depth 0 --in-place

py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_puzzles --depth 1 --in-place
```


## Convert to npz
```powershell
py -3.12 -u scripts/encode_slice_features.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_60000-65000_d90 --rebuild

py -3.12 -u scripts/encode_slice_features.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_65000-70000_d90 --rebuild
```


## Improve depth
```powershell
py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_puzzles --depth 2 --in-place

py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_15000-20000_d90 --depth 2 --in-place

py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_20000-25000_d90 --depth 2 --in-place

py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_25000-30000_d90 --depth 2 --in-place
```


## 🧱 Build dataset Generate data (sparse input, WDL proba in output) & npz
```powershell
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 400000 405000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 405000 410000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 410000 415000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 415000 420000 --dropout 0.9
```


## Database of extreme positions (P(draw) <= 5%)
```powershell
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10 --max-draw 0.05 --dropout 0.9
```
 
 
## Training of the mini model (linear + softmax no hidden layers) tests
```powershell
py -3.12 -u scripts/train_linear_wdl.py --epochs 10 --smoke --run-name linear_wdl_smoke --plot plots/linear_wdl_smoke_ce.png

py -3.12 -u scripts/train_linear_wdl.py --epochs 100 --smoke --lr 1e-2 --run-name linear_wdl_smoke --plot plots/linear_wdl_smoke_ce.png

py -3.12 -u scripts/train_linear_wdl.py --epochs 10 --lr 0.01 --run-name linear_wdl --plot plots/linear_wdl_ce.png
```


## Actual training 
```
py -3.12 -u scripts/train_linear_wdl.py --epochs 100 --lr 0.01 --run-name linear_wdl --plot plots/linear_wdl_ce.png
```


## Inspect best and worst guesses
```powershell
py -3.12 -u scripts/inspect_nnue_positions.py linear_wdl_smoke --ckpt best --n 20 --sample 1000 --seed 0 --slice fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90
```


## Faster Training Small Model 
```powershell
py -3.12 -u scripts/train_linear_wdl.py --epochs 100 --lr 0.01 --run-name linear_wdl --plot plots/linear_wdl_ce.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast
```


## Faster Training Medium Model 

```powershell
py -3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 32 --run-name medium_h32_fast --plot plots/medium_wdl_ce.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 64 --run-name medium_h64_fast --plot plots/medium_wdl_ce.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 128 --run-name medium_h128_fast --plot plots/medium_wdl_ce.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast
```

## Faster Training Dual Model

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 16 --hidden2-dim 32 --run-name dual_h16_H32_fast --plot plots/dual_nnue_ce_64_64.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 32 --hidden2-dim 64 --run-name dual_h32_H64_fast --plot plots/dual_nnue_ce_64_128.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 64 --hidden2-dim 128 --run-name dual_h64_H128_fast --plot plots/dual_nnue_ce_64_128.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_nnue.py --epochs 200 --lr 0.01 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_fast --plot plots/dual_nnue_ce_128_256.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 5000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_nnue.py --epochs 200 --lr 0.01 --hidden-dim 256 --hidden2-dim 512 --run-name dual_h256_H512_fast --plot plots/dual_nnue_ce_256_512.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 5000 --train-val-subset-size 5000 --fast
```

---

## Run ALL

```powershell
py -3.12 -u scripts/train_linear_wdl.py --epochs 100 --lr 0.01 --run-name linear_wdl --plot plots/linear_wdl_ce.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 32 --run-name medium_h32_fast --plot plots/medium_wdl_ce.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 16 --hidden2-dim 32 --run-name dual_h16_H32_fast --plot plots/dual_nnue_ce_64_64.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast
```

---

## With scheduler

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 300 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_fast --plot plots/dual_nnue_ce_128_256_scheduler_300_epoch.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 5000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_nnue.py --epochs 1000 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_fast --plot plots/dual_nnue_ce_128_256_scheduler_1000_epoch.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 3200 --train-val-subset-size 3200 --batches-per-epoch 40 --batch-size 256 
```

## NNUE Variable batch size 
default batch-size = 256
default batches-per-epoch = 40

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 300 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_fast --plot plots/dual_nnue_ce_128_256_scheduler_5.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 3200 --train-val-subset-size 3200 --batches-per-epoch 32 --batch-size 1024 

py -3.12 -u scripts/train_nnue.py --epochs 200 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --run-name dual_h128_H256_fast --plot plots/dual_nnue_ce_128_256_scheduler_6.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 3200 --train-val-subset-size 3200 --batches-per-epoch 32 --batch-size 2048
```

Best test_ce=0.696645 