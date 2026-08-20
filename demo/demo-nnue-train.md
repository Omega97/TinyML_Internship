# Demo — train the base dual-hidden NNUE (128, 128)

Full training recipe for the **2-hidden** student on the unified board_eval dataset.

**Architecture:** dual-POV L1 `844 → 128` → concat `256` → L2 `256 → 128` → head `128 → 1` → tanh  
**Params:** ~141k · **Data:** `data/processed/board_eval/` (labeled teacher pairs)

Run from the **repo root**.

---

## Prerequisites (checklist)

| Item                | Status / path                                                                           |
| ------------------- | --------------------------------------------------------------------------------------- |
| Unified dataset     | `data/processed/board_eval/dataset.json` (~37k labeled)                                 |
| Train / val parquet | `data/processed/board_eval/splits/train.parquet` · `val.parquet` (~35.3k / 1.9k)        |
| Model               | `DualHiddenNNUE` in `src/tinymlinternship/nnue/model.py` (`--architecture dual_hidden`) |
| Train CLI           | `scripts/train_nnue.py`                                                                 |
| Eval CLI            | `scripts/run_engine.py --eval nnue --nnue-checkpoint …`                                 |
| ACPL judge          | Stockfish on PATH / `STOCKFISH_PATH` · `scripts/eval_bot_acpl.py`                       |
| PyTorch             | `pip install torch` · `pip install -e .`                                                |

Rebuild dataset (if needed):

```powershell
py -3.12 scripts/build_dataset_json.py
# then refresh splits:
py -3.12 -c "import json,pandas as pd,chess; from pathlib import Path; from tinymlinternship.features.bucket import bucket_id,has_queen,piece_count; d=json.loads(Path('data/processed/board_eval/dataset.json').read_text(encoding='utf-8')); rows=[]; 
# prefer: keep existing splits unless dataset changed
print('use existing splits unless you re-exported')"
```

Splits already on disk from the last export — re-export only if `dataset.json` changed.

---

## Smoke (1–2 epochs)

```powershell
py -3.12 scripts/train_nnue.py `
  --architecture dual_hidden `
  --hidden-dim 128 `
  --hidden2-dim 128 `
  --epochs 2 `
  --batch-size 256 `
  --lr 1e-3 `
  --run-name dual_base_W128_H128_smoke `
  --train data/processed/board_eval/splits/train.parquet `
  --val data/processed/board_eval/splits/val.parquet
```

Checkpoint: `models/checkpoints/nnue/dual_base_W128_H128_smoke/best.pt`

One-shot bestmove:

```powershell
py -3.12 scripts/run_engine.py --eval nnue `
  --nnue-checkpoint models/checkpoints/nnue/dual_base_W128_H128_smoke/best.pt `
  --depth 1
```

---

## Full training (base model 128 / 128)

```powershell
py -3.12 scripts/train_nnue.py `
  --architecture dual_hidden `
  --hidden-dim 128 `
  --hidden2-dim 128 `
  --epochs 40 `
  --batch-size 256 `
  --lr 1e-3 `
  --run-name dual_base_W128_H128_ep40 `
  --train data/processed/board_eval/splits/train.parquet `
  --val data/processed/board_eval/splits/val.parquet
```

| Output | Path |
| ------ | ---- |
| Best weights | `models/checkpoints/nnue/dual_base_W128_H128_ep40/best.pt` |
| Last epoch | `…/last.pt` |
| Config / history | `…/config.json`, `history.json` |

~15–20 s/epoch on CPU (encode-on-the-fly) → ~10–15 min for 40 ep (order of magnitude).

---

## After training — Stockfish ACPL gate

```powershell
$env:STOCKFISH_PATH = (Resolve-Path tools\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe).Path

py -3.12 scripts/eval_bot_acpl.py --eval nnue `
  --nnue-checkpoint models/checkpoints/nnue/dual_base_W128_H128_ep40/best.pt `
  --depth 1 --no-quiescence --games 3 --max-plies 80 --sf-movetime-ms 100 `
  --output-pgn images/plots/PGN_and_JSON/dual_base_W128_H128_d1_gate.pgn `
  --json images/plots/PGN_and_JSON/dual_base_W128_H128_d1_gate_acpl.json `
  --no-gif --verbose --no-progress
```

---

## Shape reference

```text
white/black sparse 844
        │
   L1 Linear 844→128 + CReLU  (shared, dual POV)
        │
   concat STM ‖ Opp → 256
        │
   L2 Linear 256→128 + CReLU
        │
   head Linear 128→1 + tanh → expected_reward ∈ [-1,1]
```

CLI one-shot (depth or time): [demo-nnue.md](demo-nnue.md).
