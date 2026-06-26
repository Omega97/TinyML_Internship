# Project Checkpoint Map

*TinyML Chess — compressing a value network onto the Wio Terminal (SAMD51). Last updated: 2026-06-26.*

---

## Preparing the repo

- [x] Basic repo structure (`src/`, `scripts/`, `tests/`, `examples/`)
- [x] Data download pipeline — `scripts/download_data.py`
- [x] Dataset tests — `tests/test_data.py`
- [x] Example game runner — `examples/example_game.py`
- [x] FEN utilities and featurizer — `examples/example_fen.py`, `src/tinymlinternship/datasets/featurizer.py`
- [x] Policy inference tests — `tests/test_policy_inference.py`
- [x] Model definitions — `TinyValueMLP` / `UltraTinyValueMLP` family in `src/tinymlinternship/models/value.py`
- [x] PC inference runner — `scripts/run_model.py`
- [x] Project docs — `PROJECT.md`, `README.md`, `NOTES/`

---

## Familiarizing with hardware

- [x] Wio Terminal connected and flashed (`Blink.ino`)
- [x] Arduino IDE + **Seeed SAMD Boards** package installed
- [x] Tutorial on loading a model on the *Wio Terminal*
- [x] Game of Life demo optimized and running on device
- [x] TFT + Serial I/O working (2.4" LCD, 115200 baud)
- [x] Documented hardware limits — 120 MHz CPU, 192 KB RAM, ~500 KB flash ([NOTES/Performance.md](NOTES/Performance.md))

---

## Export & deployment pipeline

- [x] End-to-end export chain documented — [export_pipeline.md](export_pipeline.md)
- [x] PyTorch → TorchScript → C header pipeline — `scripts/prepare_for_arduino.py`, `scripts/bin_to_c_header.py`
- [x] Wio-specific int8 export scripts — `prepare_wio_{nano,tiny,small,medium,big,huge}.py`
- [x] **Script to generate model structure and parameters** — `scripts/generate_wio_weights.py`, `scripts/wio_int8_common.py`
- [x] FEN → C array helper — `scripts/fen_to_c_array.py`
- [x] **Modify the .mio to import the model correctly**
- [x] Model size sweep on device — nano → huge (`768→16→8→1` through `768→512→64→1`)
- [x] Huge model fits at ~96% flash (488 KB weights + ~60 KB sketch overhead)

---

## First inference on device

- [x] Hand-written forward pass for `UltraTinyValueMLP` on Wio
- [x] PC ↔ device parity confirmed (FEN input → same inferred value within float precision)
- [x] Int8 quantization path — ~4× lower weight memory vs float32
- [x] Fixed critical **sign bug** — cast `pgm_read_byte` to `(int8_t)` before float conversion (negative weights were read as 251)
- [x] **Accuracy restoration** — removed hidden-layer binarization `(h > 0.5f ? 1 : 0)`; activations stay `float` between layers
- [x] **Dynamic architecture support** — layer dims driven by macros (`FC1_IN_DIM`, `FC1_OUT_DIM`, …) instead of hardcoded `768`

---

## First benchmarks

- [x] Run random network of **maximum size** (huge: `768→512→64→1`) on device
- [x] Performance benchmark — full nano→huge latency sweep ([NOTES/Performance.md](NOTES/Performance.md))
  - nano **1.4 ms** · tiny **2.8 ms** · small **5.8 ms** · medium **11.4 ms** · big **22.6 ms** · huge **45.0 ms**
  - Latency scales ~2× per tier; bottleneck is `pgm_read_byte` flash bus stalls, not FPU
- [x] **Benchmark honesty fix** — flat ~2.01M evals/s was a `-Os` dead-code artifact (`forward()` elided from `loop()`). Fixed with `volatile forwardSink`, interval EMA, 1s warm-up discard
- [x] Fix calling-time metric — throughput now reflects real `evaluate()` calls
- [x] Removed misleading `K` suffix from evals/s display
- [x] Performance report — Excel table in `PRIVATE/performance.xlsx`
- [x] **Real-time health monitoring** — `freeRam()` integrated into LCD/Serial output
- [x] Integer **LUT vs on-the-fly float** conversion experiment (no speedup on FPU SAMD51, as expected)

---

## Performance optimization

- [x] Sketch refactor — modular `config.h`, `Int8ValueNet`, `WioBoard`, `Benchmark` (weights included once; fixes 3× PROGMEM duplication)
- [x] **Sparse Layer 1** — skip `pgm_read_byte` when `x[j] == 0` (~32 active squares / 768 features → ~95% fewer L1 multiplications)
- [x] Optimized forward pass — huge **53 ms → 45 ms** (~15% faster); nano **1.8 ms → 1.4 ms** (~22% faster)
- [ ] Sparse L2/L3 guards (skip reads on zero activations)
- [ ] Memory-hierarchy tweak — stage hottest weight block (e.g. full L1) from PROGMEM into RAM at boot
- [ ] Profile display overhead — benchmark with Serial/TFT updates disabled
- [ ] Further improve NN inference throughput

---

## Training & data

- [ ] **Data pipeline** — refine preprocessing and decide on final dataset (Kaggle `games.csv` downloaded; no pretrained chess model in repo yet)
- [ ] Train a mini chess bot on existing CSV, then re-export int8 header for Wio
- [ ] Distillation / pruning from a larger teacher model
- [ ] Elo testing against a known-strength baseline

---

## Search & full engine

- [ ] Add shallow Negamax search on top of value net
- [ ] Model search overhead as `overhead + cost/node × nodes`
- [ ] Input system (board state capture) and output system (move selection / display)
- [ ] Power profiling with OTII

---

## Research & community

- [ ] Thesis — *Explainable tiny chess and World Models* (working title opposite of Magnus = "big"). Can models build an internal board?
- [ ] Non-linear probes — can a large model reconstruct game state from latent vectors? 🤖
- [ ] World models to compress state info and facilitate training? 🌍
- [ ] Review papers on compressing chess engines onto small chips
- [ ] Chess ESP32 — check what it is
- [ ] Reddit projects — explore interesting tiny-chess work
- [ ] Competing paradigms for machine intelligence — what is this about?
- [ ] ICGA — what is it?
- [ ] Join chessprogramming.org Discord
- [ ] Explore r/chessprogramming

---

## Key takeaways so far

| Insight | Detail |
| ------- | ------ |
| Flash is the bottleneck | `pgm_read_byte` stalls dominate; wider layers → more fetches → slower evals |
| Sparse input helps L1 | ~32 pieces on board skips most of the first layer |
| Measurement matters | Compiler can silently remove inference; always verify with `objdump` / volatile sink |
| Quantization saves space, not time | Int8 weights are ~4× smaller but FPU dequant path is still the hot loop |
| Max deployable model | `768→512→64→1` at ~96% flash; **45 ms/eval** after optimizations |