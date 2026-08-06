# Demo — run the NNUE bot (startpos / FEN, depth or time)

One-shot **best move** with SARDINE’s Python search + a trained NNUE value net.  
From the **repo root**. Needs PyTorch and the checkpoint on disk.

Related: self-play GIF → [demo-nnue-gif.md](demo-nnue-gif.md).

---

## Prerequisites

```powershell
pip install -e .
pip install torch
```

| Need | Notes |
| ---- | ----- |
| CLI | `scripts/run_engine.py` |
| Eval | `--eval nnue` |
| Checkpoint | default: `models/checkpoints/nnue/pilot_W128_844/best.pt` |
| Dual base **128 / 128** (train) | [demo-nnue-train.md](demo-nnue-train.md) |
| Dual W128 H256 (earlier run) | `models/checkpoints/nnue/dual_W128_H256_board_eval_ep40/best.pt` |

---

## Checkpoints

| Name | Path | Notes |
| ---- | ---- | ----- |
| **Pilot (default)** | `models/checkpoints/nnue/pilot_W128_844/best.pt` | Multi-head pilot; ladder d1 ~ACPL 139 |
| **Dual base 128/128** | `models/checkpoints/nnue/dual_base_W128_H128_ep40/best.pt` | Full train: [demo-nnue-train.md](demo-nnue-train.md) |
| **Dual W128 H256** | `models/checkpoints/nnue/dual_W128_H256_board_eval_ep40/best.pt` | Earlier 40-ep run · same data |

Omit `--nnue-checkpoint` to use the default pilot. For dual-hidden nets, pass the path explicitly.

---

## 1. Startpos — fixed depth `n`

```powershell
# Default pilot, depth 1
py -3.12 scripts/run_engine.py --eval nnue --depth 1

# Depth n (example n=3)
py -3.12 scripts/run_engine.py --eval nnue --depth 3

# Dual-hidden student, depth 2, qsearch cap 6
py -3.12 scripts/run_engine.py --eval nnue `
  --nnue-checkpoint models/checkpoints/nnue/dual_W128_H256_board_eval_ep40/best.pt `
  --depth 2 --max-qsearch-depth 6
```

**Example output:**

```text
bestmove d2d4
info side White depth 2 score cp 25 nodes 1234
```

---

## 2. Startpos — time wall (`--movetime`)

Iterative deepening until the budget is used (depth not fixed).

```powershell
# 1 second
py -3.12 scripts/run_engine.py --eval nnue --movetime 1

# 10 seconds, dual-hidden net
py -3.12 scripts/run_engine.py --eval nnue `
  --nnue-checkpoint models/checkpoints/nnue/dual_W128_H256_board_eval_ep40/best.pt `
  --movetime 10 --max-qsearch-depth 6

# Cap how deep ID may go (optional)
py -3.12 scripts/run_engine.py --eval nnue --movetime 5 --max-search-depth 32
```

If both `--movetime` and `--depth` are set, **`--movetime` wins** (`--depth` is ignored).

---

## 3. Custom FEN

```powershell
py -3.12 scripts/run_engine.py --eval nnue --depth 2 `
  --fen "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

# Dual net + 2 s wall clock
py -3.12 scripts/run_engine.py --eval nnue `
  --nnue-checkpoint models/checkpoints/nnue/dual_W128_H256_board_eval_ep40/best.pt `
  --movetime 2 `
  --fen "4qk2/8/8/8/8/8/8/4R1K1 w - - 0 1"
```

---

## 4. After some UCI moves from startpos

```powershell
py -3.12 scripts/run_engine.py --eval nnue --depth 2 --moves "e2e4 e7e5 g1f3"
```

---

## Useful flags

| Flag | Meaning |
| ---- | ------- |
| `--eval nnue` | Use NNUE value (required for this demo) |
| `--nnue-checkpoint PATH` | Weights file |
| `--depth N` | Fixed alpha-beta depth |
| `--movetime SEC` | Time budget (seconds), iterative deepening |
| `--max-search-depth N` | Max ID depth with `--movetime` (default 64) |
| `--fen "..."` | Position (default = startpos) |
| `--moves "e2e4 …"` | UCI moves applied before search |
| `--quiescence` / `--no-quiescence` | Capture qsearch at leaves (default: on) |
| `--max-qsearch-depth K` | Cap qsearch (e.g. 6) |
| `--version` | Print SARDINE engine version |

---

## Compare with Cfish (optional)

Cfish is a separate UCI binary (stock Hybrid NNUE), not the Python student:

```powershell
# smoke Cfish depth 5 (cwd must be src/cfish — script handles it)
py -3.12 scripts/cfish.py
```

Interactive: `run-cfish.bat`.

---

## Self-play GIF (not one-shot)

```powershell
# see demo-nnue-gif.md
py -3.12 scripts/record_engine_game.py --eval nnue --depth 1 --no-quiescence --max-plies 80 --headless `
  --nnue-checkpoint models/checkpoints/nnue/dual_W128_H256_board_eval_ep40/best.pt `
  --output images/games/nnue_dual_demo.gif
```
