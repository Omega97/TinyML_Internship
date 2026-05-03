
# Project Blueprint

## **Scope & Milestones**

1. Get a minimal NNUE-like policy network running inference on XIAO (no search). Measure latency/power on random positions.
2. Add shallow search 
3. distillation/pruning + Elo testing (power optimization)
3. Full system (input/output).

---

## **Models**

- [Minimal Model](...)
  - **Architecture**: Very small fully connected network (e.g. 768 → 32 → 32 → 1) or even smaller.
  - **Input**: Simple material + piece-square tables or basic 768-bit HalfKP (sparse).
  - **Output**: Position evaluation in centi-pawns (float)
  - **Size**: < 10K parameters.
  - **Why**: Fits easily on XIAO, fast inference, easy to quantize to 4-bit.
  - **Use case**: First prototype, baseline for distillation.

### [Small NNUE](...)
  - **Architecture**: `768 → 64 → 32 → 1` or `512 → 64 → 32 → 1` (HalfKP style)
  - **Input**: HalfKP (King + Piece relative positions) — the classic NNUE input.
  - **Output**: Position evaluation in centi-pawns (float)
  - **Size**: ...
  - **Why**: This is the sweet spot for TinyML. Strong enough to beat random play significantly, small enough to run with shallow search on MCU.

### 3. Good / Medium NNUE (Target for good performance)
  - **Architecture**: `768 → 256 → 32 → 1` or modern small nets like `1024 → 128 → 32 → 1`
  - **Input**: ...
  - **Output**: Position evaluation in centi-pawns (float)
  - **Size**: ...
  - **Why**: This is what many strong open-source tiny chess NNs use. You can get decent Elo (1600–2000+) with light search.


---

## **Data & Training**

- Large chess datasets for distillation.
  -  [Lichess database](https://database.lichess.org/)
    
- Small chess datasets for testing.
  - [small Kaggle database](https://www.kaggle.com/datasets/datasnaek/chess)

---
