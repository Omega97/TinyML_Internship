# SARDINE — Chess Engine for the Wio Terminal

## Mission

Build a playable chess bot that runs **entirely on-device** (no cloud, no GPU) on the Seeed Wio Terminal — a microcontroller with only **120 MHz CPU, 192 KB RAM, and 512 KB flash**. The goal is to maximize playing strength (Elo) per byte of memory through extreme optimization.

---

## What Makes SARDINE Special

### 1. **Hardware-Aware Architecture**
Unlike typical chess engines that assume gigabytes of RAM, SARDINE is designed from the ground up for severe memory constraints:
- **Bucketed NNUE** with 8 specialized expert heads (routed by piece count + queen presence)
- **Shared accumulator** (716 → 16, dual-perspective) computed once per position
- **CReLU activation** + **tanh lookup table** (no runtime `tanh()` computation)
- **Incremental add/sub updates** (lazy evaluation, bucket-agnostic)

### 2. **Memory Budget Discipline**
Every byte is accounted for:
- **Transposition Table**: 128–160 KB (TT-dominant strategy)
- **Accumulators + search stack**: ~16 KB
- **Scratch space**: ~16 KB
- **Weights**: ~10% of flash (int8 quantized)

### 3. **Phased Build Pipeline**
1. Feature encoder (PC + device parity)
2. Search skeleton in C++ on PC (perft, eval hooks, TT benchmark)
3. Train bucketed NNUE (Lc0 data, bucket-stratified sampling)
4. Port to Wio Terminal with incremental accumulators
5. Full search stack (quiescence, futility, LMR, null-move, killer moves, iterative deepening)
6. Elo gate test (≥1700)

---

## Expected Results

### Performance Targets
| Metric | Target | Reference |
|--------|--------|-----------|
| **Elo** | ≥ 1700 | Dog (ESP32, ~1700+ Elo on-device) |
| **Move time** | ~1 second | Urusov's ESP32 engine (~20 kNps, ~2023 Elo) |
| **Search depth** | 6–10 ply | Depends on eval latency + move-gen overhead |
| **Memory footprint** | < 192 KB RAM | Wio Terminal limit |
| **Flash usage** | < 512 KB | Wio Terminal limit |

### Deliverables
1. **Working chess engine** on Wio Terminal that plays legal moves within 1 second
2. **NNUE value network** (716 → 16 → 1, bucketed, int8 quantized) running at >1M evals/sec
3. **Alpha-beta search** with quiescence, transposition table, and move ordering (MVV-LVA + killer moves)
4. **Minimal UCI interface** over Serial for engine-vs-engine testing (cutechess-cli compatible)
5. **Reproducible pipeline** from Python training to C deployment on device

### Validation
- **PC ↔ Device parity**: Feature encoder and NNUE forward pass produce identical outputs on PC and Wio
- **Elo testing**: Play against known-strength baselines (Stockfish level 1–3, or other tiny engines)
- **Power profiling** (optional): Measure energy per move with OTII Arc

---

## Why This Matters

SARDINE demonstrates that **strong chess AI can run on severely constrained hardware** — a microcontroller with less memory than a single JPEG image. This has implications for:
- **Edge computing**: Deploying intelligent decision-making without cloud dependency
- **Green AI**: Low-power, energy-efficient inference
- **Education**: Hands-on TinyML and embedded systems learning
- **Research**: Exploring the limits of model compression and hardware-aware design

The project is a proof-of-concept that **Elo per byte** can be optimized systematically, paving the way for more complex AI on tiny devices.

---

## References

- **Dog** (Folkert van Heusden): ESP32 chess engine with NNUE, ~1700+ Elo, 320 KB RAM  
  [GitHub](https://github.com/folkertvanheusden/Dog) · [Site](https://vanheusden.com/chess/Dog/)
  
- **Urusov's ESP32 engine**: ~20 kNps, heuristics-only, ~2023 Elo (baseline for search throughput)

- **Stockfish NNUE**: State-of-the-art NNUE architecture (inspiration for bucketed design)  
  [Chessprogramming](https://www.chessprogramming.org/Stockfish_NNUE)

- **nnue-pytorch**: Training framework for NNUE networks  
  [GitHub](https://github.com/official-stockfish/nnue-pytorch)

---

## Timeline

- **Phase 1** (Current): Feature encoder (716 sparse, dual-perspective, bucket router) — in progress
- **Phase 2**: Search skeleton on PC (alpha-beta + quiescence, perft, TT benchmark)
- **Phase 3**: NNUE training (Lc0 data subset, bucket-stratified, int8 export)
- **Phase 4**: Port to Wio Terminal (incremental accumulators, C port, benchmark)
- **Phase 5**: Full search stack + tuning (SPSA, iterative deepening)
- **Phase 6**: Elo gate test (≥1700) → v2 scope (policy head, UCI polish)

---

**SARDINE** — *Small but mighty.* 🐟