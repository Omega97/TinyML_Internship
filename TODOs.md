# SARDINE Blueprint — Progress

_Checkpoint map vs NOTES/SARDINE Engine Blueprint.md only. Last updated: 2026-07-03._

---

## Build pipeline

Stesso ordine del diagramma _Build Pipeline_ nel blueprint.

### A · Feature encoder (PC + device parity)

- [x] Pruned **716** features (pawn prune, king mirror, castling, EP) — `index_map.py`, `encoder.py`, `mirror.py`
- [x] Dual-perspective sparse input — `encode_dual()`
- [x] Enemy-king full 64-square resolution (only the perspective king is mirror-compressed to 32 slots)
- [x] **Castling coordinate-frame fix** — rights from `base` + K/Q swap if mirrored (EP from `view`); golden startpos + `test_castling_*`
- [x] 8-bucket router (piece count + queen-split), boundaries closed at **p ≤ 12** (no gap)
- [x] Gate test (`tests/test_features.py`, golden FEN) — **29** tests
- [ ] Encoder parity on **device** (C, with search) — step F

---

### B · Search skeleton on PC

- [ ] perft
- [x] HCE eval bring-up — `evaluate_hce` (`tests/test_engine.py`)
- [x] 1-ply search bring-up — `search_best_move` (not alpha-beta / depth)
- [ ] Eval hook at depth-search nodes (NNUE swap-in later)
- [ ] TT entry format prototype + PC benchmark
- [ ] Alpha-beta (blueprint: C++ on PC first)
- [ ] Nodes/s benchmark on PC
- [ ] Node-budget model vs Urusov ESP32 baseline (~20 kNps, heuristics-only) — estimate reachable depth once eval latency is measured

---

### C · Train bucketed NNUE

Architecture: shared **716 → 16** (int8, dual call, **same weights applied twice** — own-POV + opponent-POV, concatenated by side-to-move, not by fixed color) → concat **32** → router → expert **32 → 1** × 8 · **CReLU** hidden **and output** (no tanh — tanh LUT was explicitly rejected earlier in favor of CReLU to avoid extra flash + int8-quantization complexity; correct this if it's still in the design anywhere downstream).

- [x] Lc0 subset **~1–2 GB** (`data/raw/lc0/`, `scripts/download_lc0.py`)
- [ ] Games **≥ 16** moves; bucket-stratified resampling
- [ ] Stockfish centipawn labels
- [ ] **nnue-pytorch** train (shared accumulator + 8 heads)
- [ ] Calibrated **int8** export (histogram post-training weights, scale onto [-127,127])
- [ ] Measure fp32→int8 eval-error gap — **decide acceptance threshold** (e.g. <5–10 centipawn avg delta) before treating PTQ as sufficient
- [ ] Magnitude pruning ~80% post-training
- [x] Kaggle `games.csv` smoke only (not NNUE training) — `scripts/download_data.py`
- [x] Piece-count distribution for bucket design — `plot_piece_count_distribution.py`, `excel/piece_count_distribution_10k.xlsx`

---

### D · Queen-split ablation

- [ ] Per-bucket eval MSE vs piece-count-only baseline (Stockfish-labeled val set, stratified like training)
- [ ] Define "decisive vs ambiguous" threshold (e.g. >5% relative MSE change per bucket = decisive; 2–5% or mixed-direction buckets = ambiguous → escalate)
- [ ] Playing-strength test — **only if** per-bucket results are ambiguous or contradictory

---

### E · Incremental accumulators (device)

- [ ] Lazy add/sub on shared layer (bucket-agnostic)
- [ ] Full refresh on king centre-file crossing
- [ ] Lazy accumulator updates (TT cutoffs)
- [ ] Castling-bit add/sub (rare — only on king/rook moves or rook capture)
- [ ] En-passant-bit add/sub (frequent — flips near every ply, but cheap: single bit)

---

### F · Port search + NNUE to C (Wio)

- [ ] C engine core (after playable PC search)
- [ ] Benchmark **`-O3` vs `-Os`** on Wio
- [ ] int8 weights in flash; int16 accumulators in RAM (no tanh LUT — see note under C)

---

### G · Full search stack + tuning

Phased rollout dal blueprint (tutto v1, non rinviato salvo dove indicato):

- [ ] Alpha-beta + **quiescence**
- [ ] **Futility** + **LMR** + **null-move**
- [ ] **Lazy evaluation** (paired with lazy accumulators)
- [ ] **Iterative deepening** (TT stable)
- [ ] TT **128–160 KB** — format decision: ~10 B tight pack vs 16 B byte-aligned entry, decided by wall-clock nodes/sec + depth reached on Wio, **not** hit-rate alone
- [ ] Move ordering: **MVV-LVA** + **killer moves** (depth > 4)
- [x] Countermove history — **out of v1** per blueprint (killers only at depth > 4)
- [ ] **SPSA** search/heuristic tuning

---

### H · Elo gate

- [ ] **≥ 1700 Elo** (blueprint gate) — e.g. cutechess-cli
- [ ] Minimal **UCI over Serial**
- [ ] TFT **off** during search; Serial for debug

---

### I · Iterate (if gate missed)

- [ ] SCReLU fallback (hidden layer) — clip int16 accumulator to activation range **before** squaring (load-bearing, avoids overflow); square in int16; multiply-accumulate with int8 weights in **int32**
- [ ] Quantization-aware training (only if int8 gap too large) — stay on nnue-pytorch first; only evaluate Grapheus or in-pipeline QAT if PTQ gap threatens the Elo gate
- [ ] TT format / bucket scheme revision

---

### J · v2 (after gate)

- [ ] Minimal UCI polish
- [ ] Policy guidance head (off shared accumulator, 16 → move-ranking; watch per-node latency vs ~1 s budget)
- [ ] Opening book
- [ ] SCReLU / QAT / compact transformer fallback (~210K design) — only if needed
- [ ] Tactical MoE axis (`inCheck`, capture threat) — only if switching cost analysis shows it's worth it earlier than assumed

**Explicitly deferred in blueprint (v1):** MCTS · tactical MoE heads · autoencoder warm-start · separate pattern tables · opening book · Grapheus/QAT · MicroChess stack surfing · MicroChess bare-metal patterns.

---

## Open questions / research (not blocking, but untracked otherwise)

- [x] Dog (ESP32) RAM budget study — feasibility reference in blueprint §Memory; TT-dominant plan unchanged
- [ ] Compact-transformer fallback evaluation criteria — define what "underperforms" means for the v2 policy head before deciding to invoke this fallback

---
