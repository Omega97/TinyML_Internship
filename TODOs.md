# Project Checkpoint Map

*SARDINE — bucketed NNUE chess engine for the Wio Terminal (SAMD51). Last updated: 2026-07-02.*

Blueprint: [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md) · Gap tracker: [ai-feed.md](ai-feed.md)

**Target:** ~**1700 Elo** in ~1 s/move · 192 KB RAM · ~500 KB flash · no cloud.

---

## Foundation (done)

Work from the pre-SARDINE phase — still valid as hardware and tooling baseline:

- [x] Wio bring-up — flash, TFT, Serial, SAMD51 limits ([NOTES/Performance.md](NOTES/Performance.md))
- [x] Repo scaffold — `src/`, `scripts/`, `tests/`, `NOTES/`
- [x] Kaggle `games.csv` smoke pipeline — `scripts/download_data.py`, `tests/test_data.py`
- [x] Legacy value-MLP on device — int8 export, sparse L1, benchmarks; archived in `legacy/pre-sardine/`

---

## SARDINE v1 — build pipeline

Mirrors the blueprint mermaid flow: encoder → search → train → port → Elo gate.

### 1 · Feature encoder

- [x] 716 sparse features — index map, king mirror, castling/EP meta (`src/tinymlinternship/features/`)
- [x] Dual-perspective encoder + 8-bucket router (`encode_dual`, `bucket_id`)
- [x] Step-1 gate — golden FEN snapshots, **29 tests** (`tests/test_features.py`)
- [ ] Encoder parity on device (C port with search, step 6)

### 2 · Search skeleton (PC)

- [ ] `engine/` — alpha-beta + quiescence, move gen, perft
- [ ] Eval hook stub (constant score) + TT format prototype
- [ ] Benchmark nodes/s on PC before Wio port

### 3 · Training data

- [ ] `scripts/download_lc0.py` — **~1–2 GB** Lc0 subset under `data/raw/lc0/`
- [ ] Stockfish centipawn labels on sampled positions
- [ ] Bucket-stratified resampling (games ≥ 16 moves; queen-split)

### 4 · Bucketed NNUE

Architecture (updated diagram): shared **716 → 16** FFNN (int8, called twice) → own + opponent accumulators → **concat 32** → bucket router → expert head **32 → 1** (×8 buckets). Hidden **CReLU**; output **tanh via LUT**.

- [ ] Train in **nnue-pytorch** — shared accumulator + 8 expert heads
- [ ] Calibrated int8 export + tanh LUT generation
- [ ] Queen-split ablation (per-bucket eval MSE vs piece-count baseline)

### 5 · Incremental NNUE + full search

- [ ] Lazy add/sub accumulators; full refresh on king centre-file crossing
- [ ] Search stack — futility, LMR, null-move, lazy eval, iterative deepening
- [ ] TT 128–160 KB layout; MVV-LVA + killer moves; **SPSA** tuning

### 6 · Port Wio + Elo gate

- [ ] Port search + NNUE to **C**; benchmark `-O3` vs `-Os`
- [ ] Minimal **UCI over Serial** (TFT off during search)
- [ ] **Elo gate ≥ 1700** (e.g. cutechess-cli)

### v2 (after gate)

- [ ] Policy guidance head off shared accumulator
- [ ] MCTS exploration, opening book
- [ ] SCReLU / QAT / compact transformer — only if gate missed

---

## Research & community

### Documented

- [x] **FIDE & Google Efficient Chess AI Challenge** (Kaggle) — [NOTES/FIDE & Google Efficient Chess AI Challenge.md](NOTES/FIDE%20%26%20Google%20Efficient%20Chess%20AI%20Challenge.md)
  - **Elo per byte** under hard RAM/flash caps (~5 MiB) — direct inspiration for SARDINE's constraint mindset (we have *less*: 192 KB RAM)
  - **1st — linrock / minifish:** micro-NNUE + Cfish port; strength in optimized weights, not bloated search code
  - **2nd — Approvers:** pruned input features (704-dim geometric zeros) — ancestor of our 716 layout
  - **4th — Nagiss:** aggressive geometric pruning + stripped alpha-beta (LMR, NMP)
  - **9th — HCE + SPSA:** hand-crafted eval + automated tuning can compete when NNUE is too heavy; `-O3` compiler win
  - **Dog (ESP32):** ~1700+ Elo with NNUE + TT on ~320 KB RAM — feasibility reference for Wio target

### To explore

- [ ] Deep-read top-5 Kaggle writeups — map patterns to 192 KB / 500 KB flash budget
- [ ] **Urusov** ESP32 engine (~20 kNps, ~2023 Elo, no NNUE) — search throughput baseline
- [ ] **Lc0** small nets / nnue-pytorch community recipes for bucketed training
- [ ] **chessprogramming.org** wiki + Discord; **r/chessprogramming**
- [ ] Papers on compressing chess engines onto microcontrollers
- [ ] Thesis thread — *explainable tiny chess & world models* (longer horizon)
- [ ] ICGA, competing paradigms for machine intelligence

---