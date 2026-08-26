

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


## Build dataset (only select one position in 10 at random, to improve diversity)
```powershell
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 2000 3000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 15000 20000 --dropout 0.9
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


## Generate data + npz
```powershell
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 305000 310000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 310000 315000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 315000 320000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 320000 325000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 325000 330000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 330000 335000 --dropout 0.9

py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 335000 340000 --dropout 0.9
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
```powershell
py -3.12 -u scripts/train_linear_wdl.py --epochs 100 --lr 0.01 --run-name linear_wdl --plot plots/linear_wdl_ce.png
```

## Inspect best and worst guesses
```powershell
py -3.12 -u scripts/inspect_nnue_positions.py linear_wdl_smoke --ckpt best --n 20 --sample 1000 --seed 0 --slice fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90
```


## Faster Training Small Model 
```powershell
py -3.12 -u scripts/train_linear_wdl.py --epochs 100 --lr 0.01 --run-name linear_wdl --plot plots/linear_wdl_ce.png --test-subset-size 10000 --train-val-subset-size 5000 --fast
```

## Faster Training Medium Model 
```powershell
py -3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 32 --run-name medium_h32_fast --plot plots/medium_wdl_ce.png --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 64 --run-name medium_h64_fast --plot plots/medium_wdl_ce.png --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_medium_wdl.py --epochs 100 --lr 0.01 --hidden-dim 128 --run-name medium_h128_fast --plot plots/medium_wdl_ce.png --test-subset-size 10000 --train-val-subset-size 5000 --fast
```

## Faster Training Full Model
```powershell
py -3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 64 --hidden2-dim 64 --run-name dual_W64_H64_fast --plot plots/dual_nnue_ce_64_64.png --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_nnue.py --epochs 100 --lr 0.01 --hidden-dim 64 --hidden2-dim 128 --run-name dual_W64_H128_fast --plot plots/dual_nnue_ce_64_128.png --test-subset-size 10000 --train-val-subset-size 5000 --fast

py -3.12 -u scripts/train_medium_wdl.py --epochs 50 --lr 0.01 --hidden-dim 64 --run-name medium_h64_fast --plot plots/medium_wdl_ce.png --test fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90 --test-subset-size 10000 --train-val-subset-size 5000 --fast
```
