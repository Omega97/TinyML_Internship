# Demo — train dual-POV NNUE (L1 64×2, L2 128)

Goal §2 student: sparse **844** features, shared L1 **64** per POV (king-mirrored own / opponent, concat **128**), L2 **128**, tanh White-POV \(v\in[-1,+1]\).

Run from the **repo root**. Training reads per-slice `features.npz` (no chess encode, no visits).

**Split**

| Role | Folder under `data/processed/board_eval/fen_value_visits/` |
| ---- | ---------------------------------------------------------- |
| **Test** | `fen_value_visits_lichess_db_standard_rated_2026-07_100000-101000` (~61k) |
| **Train** | every other slice folder |

---

## Prerequisites

```powershell
pip install -e ".[train]"
```

| Need | Path / notes |
| ---- | ------------ |
| CLI | `scripts/train_nnue.py` |
| Slice DBs | `data/processed/board_eval/fen_value_visits/<slice>/features.npz` |
| Model | `DualHiddenNNUE` in `src/tinymlinternship/nnue/model.py` |
| Python extras | `torch`, `pyarrow`, `matplotlib`, `tqdm` |

Loss is **unweighted MSE**. Default device is CUDA if present, else CPU. `torch.compile` is off unless you pass `--compile`.

---

## Commands to run

From the repo root.

**1. Smoke test** — 5 epochs, 20k random train rows, full 100000–101000 test:

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 5 --smoke --run-name dual_W64_H128_smoke --plot plots/nnue_smoke_mse.png
```

**2. Whole training run** — all train slices, full 100000–101000 test:

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 10 --run-name dual_W64_H128 --plot plots/nnue_mse.png
```

**3. Faster training** — smaller batches and fewer steps per epoch. Each batch is `batch_size` positions drawn at random **across all train slices** (a slice is picked uniformly, then a row inside it):

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 5 --fast --run-name dual_W64_H128_fast --plot plots/nnue_fast_mse.png
```

Same thing with explicit knobs (`--fast` is `--batch-size 256 --batches-per-epoch 40`):

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 5 --batch-size 256 --batches-per-epoch 40 --run-name dual_W64_H128_fast --plot plots/nnue_fast_mse.png
```

If a slice is missing `features.npz`, encode first:

```powershell
py -3.12 -u scripts/encode_slice_features.py
```

**IMPORTANT**: validate the results on the test set *(change the name of the model)*:
```powershell
py -3.12 -u scripts/inspect_nnue_positions.py dual_W64_H128
```

---

## Encode slices (once)

If `features.npz` is missing in a folder:

```powershell
py -3.12 -u scripts/encode_slice_features.py
```

| File | Role |
| ---- | ---- |
| `<slice>/<slice>.json` | source `{fen, value, visits}` |
| `<slice>/features.npz` | dual-POV sparse 844 + White-POV value (**no visits**) |
| `<slice>/features.meta.json` | row count / source fingerprint |

---

## Smoke (5 epochs, 20k random train rows)

`--smoke` samples 20k train rows. Evaluates on the full 100000–101000 holdout.

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 5 --smoke --run-name dual_W64_H128_smoke --plot plots/nnue_smoke_mse.png
```

Checkpoint: `models/checkpoints/nnue/dual_W64_H128_smoke/` · plot: `plots/nnue_smoke_mse.png`

---

## Whole training run

All train slices, full 100000–101000 test, 10 epochs.

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 10 --run-name dual_W64_H128 --plot plots/nnue_mse.png
```

Checkpoint: `models/checkpoints/nnue/dual_W64_H128/` · plot: `plots/nnue_mse.png`

Optional (explicit CPU / batch):

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 10 --run-name dual_W64_H128 --batch-size 2048 --lr 1e-3 --device cpu --plot plots/nnue_mse.png
```

---

## Faster training

Wall time per epoch is roughly **(steps per epoch) × (cost of one batch)**.

| Knob | Effect |
| ---- | ------ |
| `--batch-size` | Smaller → each step is cheaper (less RAM, faster backward on CPU). A full pass then needs *more* steps, so this alone does not shorten a full epoch. |
| `--batches-per-epoch` | Caps optimizer steps. This is the main way to shorten an epoch. |
| `--fast` | Sets `--batch-size 256` and `--batches-per-epoch 40` (~10k mixed samples/epoch instead of ~2M). |
| `--smoke` | 20k mixed samples/epoch (then `ceil` to whole batches). |

Every train batch samples `batch_size` positions by: pick a train **slice uniformly**, then a **row uniformly** in that slice. Puzzles / mini / Lc0 are not drowned by the large dump slices.

```powershell
py -3.12 -u scripts/train_nnue.py --epochs 5 --fast --run-name dual_W64_H128_fast --plot plots/nnue_fast_mse.png
```

Checkpoint: `models/checkpoints/nnue/dual_W64_H128_fast/` · plot: `plots/nnue_fast_mse.png`

| Flag | Meaning |
| ---- | ------- |
| `--smoke` | Cap train at 20k mixed samples/epoch unless `--max-train` is set |
| `--fast` | `batch-size 256` and `40` mixed batches/epoch |
| `--batches-per-epoch N` | Optimizer steps per epoch (`0` = pool size / batch size) |
| `--max-train N` | Random mixed samples/epoch (`0` = all) |
| `--test-slice NAME` | Override holdout folder (default: `…_100000-101000`) |
| `--slices-dir PATH` | Parent of per-slice folders |
| `--run-name NAME` | Checkpoint folder name (default includes a UTC stamp) |
| `--plot PATH` | MSE figure (default `plots/<run-name>_mse.png`) |
| `--hidden-dim` / `--hidden2-dim` | L1 width per POV (64) / L2 width (128) |
| `--encode-only` | Ensure slice DBs exist and exit |
| `--rebuild-cache` | Re-encode `features.npz` even if valid |
| `--compile` | `torch.compile` |
| `--workers N` | DataLoader workers (default **0**; data is already in RAM) |

---

## Where artifacts land

Default `--output-dir` is `models/checkpoints/nnue/`. A run named `dual_W64_H128_smoke` writes:

| Artifact | Path |
| -------- | ---- |
| **Best weights** (lowest test MSE) | `models/checkpoints/nnue/dual_W64_H128_smoke/best.pt` |
| Last epoch | `models/checkpoints/nnue/dual_W64_H128_smoke/last.pt` |
| Config | `models/checkpoints/nnue/dual_W64_H128_smoke/config.json` |
| Per-epoch metrics | `models/checkpoints/nnue/dual_W64_H128_smoke/history.json` |
| MSE plot (copy in the run dir) | `models/checkpoints/nnue/dual_W64_H128_smoke/mse.png` |
| **MSE plot (repo `plots/`)** | `plots/nnue_smoke_mse.png` (or `plots/<run-name>_mse.png`) |

Without `--plot`, the figure is still `plots/<run-name>_mse.png`. Checkpoints under `models/` are gitignored; `plots/` is not.

---

## Shape

```text
white / black sparse 844
        │
   L1 Linear 844 → 64 + CReLU   (shared, run twice)
        │
   concat [STM ‖ opponent] → 128
        │
   L2 Linear 128 → 128 + CReLU
        │
   head Linear 128 → 1 + tanh → White-POV value ∈ [-1, +1]
```

~70 721 parameters at the default widths.
