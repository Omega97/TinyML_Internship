---
description: Weekly updates on the work on the repo.
---
# Report del Progetto
**Progetto di Tirocinio all'ICTP**

## 22 Aprile

Meeting all'ufficio del prof. Zennaro

### Obbiettivo del Progetto

> Caricare un'AI di scacchi su un componente di edge-computing. L'obbiettivo è avere un prodotto funzionante, che bilancia prestazioni e consumo. 
> Misureremo l'Elo della policy contro una policy di Elo noto e comparabile (molto efficiente, le partite durano meno di un secondo). 
> Se il chip dovesse essere in grado di gestire una *tree search*, dovremmo modellare il consumo del chip come $overhead + consumo/nodo * nodi$ . 
> Inoltre, la *tree search* userebbe una *value function*. Possiamo misurare la forza di questa in modo simile, trasformandola però prima in una *policy* (valutando ciascuna mossa possibile).
> Per completare il prodotto, bisogna valutare il sistema di input dello stato del gioco, e output dell'azione del bot. 

#### Spunti teorici
- Valutare le performance di una rete neurale quantizzata e distillata (a vari livelli di compressione) rispetto al bot originale, e ricavare una "legge fisica"
- Do AI models have an inner representation of the chessboard?
- How good is a latent vector representation of the board?


#### Di cosa abbiamo discusso col professore
1. Xiao setup
2. [EdgeImpulse](https://www.edgeimpulse.com) e [SenseCraft](https://sensecraft.seeed.cc)
3. [LiteRT](https://ai.google.dev/edge/litert) e PyTorch workflow
4. OTII
5. [MLSysBook AI kits](https://mlsysbook.ai/kits/)

---

## 23-26 Aprile
- TinyML Framework
- NNUE
- Model Compression
- Knowledge Distillation

(See [NOTES/notes.md](_notes.md) for details.)

---

## 27 Aprile - 3 Maggio
- Xiao Setup
- Edge Impulse
- SenseCraft
- LiteRT
- PyTorch Workflow
- Power Measurement with OTII
- MLSysBook AI Kits

(See [NOTES/notes.md](_notes.md) for details.)

#### Repo work
- basic repo structure
- [download_data.py](scripts/download_data.py)
- [test_data.py](tests/test_data.py)
- [example_game.py](examples/example_game.py)

---

## 4-10 Maggio
- FEN
- Value functions
- Policy functions

(See [NOTES/notes.md](_notes.md) for chess/hardware/software notes; [PROJECT.md](PROJECT.md) for model details.)

#### Repo work
- [example_fen.py](examples/example_fen.py)
- [test_policy_inference.py](tests/test_policy_inference.py)
- [featurizer.py](src/tinymlinternship/datasets/featurizer.py)

---

## 25-31 Maggio

- **Models**: list of model candidates
- **MCU Deployment**: what is it about?
- **Quantization**: How to perform (can it be done in Python??)
- MicroPython
- CircuitPython: what is it?
- **Power Profiling (OTII)**: how to
- TinyTorch not on hardware? How it works? does it work?

---

## 8-14 Giugno

- 12/6 - **Prima esperienza il lab** -  [export pipeline](export_pipeline.md)
 - Docs: [export_pipeline.md](export_pipeline.md), [PRIVATE/private-notes.md](PRIVATE/private-notes.md) (PRIVATE/ index + #core)
 - Software: ONNX, TorchScript
 - Hardware: Wio Terminal
 - `models\exported\my_tiny_model.ts.pt`
 - Arduino IDE

#### Repo work
 - Main pipeline files: [scripts/prepare_for_arduino.py](legacy/pre-sardine/scripts/prepare_for_arduino.py) (TFLite export + C header, cleaned with lazy imports), [scripts/prepare_wio_tiny.py](legacy/pre-sardine/scripts/prepare_wio_tiny.py) (Wio-specific), [scripts/bin_to_c_header.py](legacy/pre-sardine/scripts/bin_to_c_header.py) (binary to C array)
 - Model files: [models/policy.py](legacy/pre-sardine/src/tinymlinternship/models/policy.py) (TinyPolicy), [models/value.py](legacy/pre-sardine/src/tinymlinternship/models/value.py) (TinyValueMLP / UltraTinyValueMLP)
 - Fixes: lazy heavy imports, type TODO cleanup, import style fix in policy.py
 - **Run policy and value functions** by running the example commands in [scripts/run_model.py](legacy/pre-sardine/scripts/run_model.py).

**Pipeline chain (start-to-finish):**  
- [models/value.py](legacy/pre-sardine/src/tinymlinternship/models/value.py) (UltraTinyValueMLP) + [scripts/prepare_wio_tiny.py](legacy/pre-sardine/scripts/prepare_wio_tiny.py) 
- → `models/checkpoints/tiny_value_wio.pt` 
- → `torch.jit.trace` + save → `models/exported/my_tiny_model.ts.pt` (58.4 KB) 
- → [scripts/bin_to_c_header.py](legacy/pre-sardine/scripts/bin_to_c_header.py) (my_tiny_model.ts.pt --var-name g_chess_model --out ...) 
- → `models/arduino/models/my_tiny_model.h` (g_chess_model + _len=59754 for #include / TFLM)

#### Hardware 
- Connected the Wio to the PC
- `Blink.ino`

---

## 15-21 Giugno

- **Wio Terminal - Game of Life demo works!** Optimized the default demo.
- Wio Terminal value net verification: hand-written forward pass for UltraTinyValueMLP with real FEN input (generated via `scripts/fen_to_c_array.py`) now runs on device.
  - Sketch: `Arduino/Wio_TinyValueTest/Wio_TinyValueTest.ino` (includes `wio_weights.h` + `fen_input.h`, TFT_eSPI LCD using same pattern as Life example).
  - Output on both Serial (115200) and 2.4" LCD (320x240): "Inferred value: -0.056165"
  - Matches PC: `py -3.12 scripts/run_model.py ...` → -0.0562 (within float precision).
  - Parity confirmed. LCD + Serial I/O working.
- Int8 quantization experiment (using prepare_wio_tiny pipeline + wio_int8_weights.h): same 2.16M evals/s as float32 (naive int8 + dequant scales gives no speedup on FPU SAMD51, as expected), but ~4x lower weight memory. Display now includes "Evals/s: 2.16M" and weights filename. See daily note 2026-06-16.md for full log. We are probably not actually running the network.

#### Repo work
- Input helper: [fen_to_c_array.py](legacy/pre-sardine/scripts/fen_to_c_array.py) → `Arduino/Wio_TinyValueTest/fen_input.h`
- Model: [UltraTinyValueMLP](legacy/pre-sardine/src/tinymlinternship/models/value.py) (`768→32→16→1`)
- Export: [generate_wio_weights.py](legacy/pre-sardine/scripts/generate_wio_weights.py) (float32 `wio_weights.h`), [prepare_wio_tiny.py](legacy/pre-sardine/scripts/prepare_wio_tiny.py) (int8 `wio_int8_weights_tiny.h`)
- PC parity: [run_model.py](legacy/pre-sardine/scripts/run_model.py)
- Device sketch: [Arduino/Wio_TinyValueTest/Wio_TinyValueTest.ino](legacy/pre-sardine/Arduino/Wio_TinyValueTest/Wio_TinyValueTest.ino) (hand-written forward pass, TFT + Serial)

---

## 22-28 Giugno

- **Ricerca modelli su Hugging Face** — esplorati checkpoint e architetture open-source per scacchi (AlphaZero-style CNN, ResNet, transformer, NNUE) per capire dimensioni, input encoding e policy/value head rispetto ai limiti Wio.
- **Measuring memory and time to run the NNs correctly!** Extended the Wio value-net performance matrix to **big** (`768→256→64→1`) and **huge** (`768→512→64→1`); full nano→huge sweep now fits on device (huge at ~96% flash).
- **Benchmark honesty fix:** the flat ~2.01M evals/s across all models was a measurement artifact — `-Os` dead-code elimination removed `forward()` from `loop()`. Fixed with `volatile forwardSink`, interval-based EMA rate, and 1s warm-up discard; throughput now scales with model size (~2× latency per tier: nano 1.4 ms → huge 45 ms).
- **Sketch refactor:** split `Wio_TinyValueTest` into `config.h`, `Int8ValueNet`, `WioBoard`, `Benchmark`; weights included once in `Int8ValueNet.cpp` (fixes 3× PROGMEM duplication that overflowed huge). Sparse L1 skips `pgm_read_byte` on empty board squares.
- **24/6 lab session:** optimized forward pass (~15% faster overall; nano 1.8→1.4 ms/call); removed misleading `K` display suffix; updated [NOTES/Performance.md](NOTES/Performance.md) with honest latency/evals/s table and hw–sw synergy notes (flash bus stalls dominate, not FPU). See daily notes [2026-06-22.md](2026-06-22.md), [2026-06-24.md](2026-06-24.md).

#### Repo work
- Models: [BigValueMLP / HugeValueMLP](legacy/pre-sardine/src/tinymlinternship/models/value.py) (nano→huge family)
- Export scripts: [prepare_wio_big.py](legacy/pre-sardine/scripts/prepare_wio_big.py), [prepare_wio_huge.py](legacy/pre-sardine/scripts/prepare_wio_huge.py), [count_model_params.py](legacy/pre-sardine/scripts/count_model_params.py)
- Device sketch: `legacy/pre-sardine/Arduino/Wio_TinyValueTest/` (modular int8 forward + benchmark)

---

## 29 Giugno - 5 Luglio

- **Catalogo modelli** — consolidata la ricerca HF in [NOTES/Models.md](NOTES/Models.md): confronto per famiglia (Dense/Conv, ResNet, Transformer, NNUE, Lc0 edge) con parametri, file size e fattibilità su Wio. Conclusione: i modelli HF (8–100M params) sono fuori budget; NNUE e dual-head custom restano le direzioni più promettenti.
- **Transformer compatto** — definita in [NOTES/chess transformer.md](NOTES/chess%20transformer.md) un'architettura policy+value a **~210K parametri** (input `24×8×8`, 2 blocchi transformer, policy 2048 + value tanh) — ~165× più piccola di ChessBot (34.7M).
- **FIDE & Google Challenge** — analizzate le soluzioni top sotto vincoli estremi (5 MiB RAM, binario ≤ 64 KiB): micro-NNUE, king mirroring, geometric pruning, SPSA tuning. Note: [FIDE & Google Efficient Chess AI Kaggle Challenge](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge).
- **NNUE deep-dive** — nota dedicata su architettura, aggiornamenti incrementali e quantizzazione: [NOTES/NNUE.md](NOTES/NNUE.md).
- **Analisi dataset** — distribuzione piece-count su 1k e 10k partite Lichess (`piece_count_distribution.xlsx`, `piece_count_distribution_10k.xlsx`); usata per progettare i bucket bilanciati del training.
- **SARDINE blueprint** — decisioni in [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md); catalogo opzioni in [SARDINE design options.md](NOTES/SARDINE%20design%20options.md). Vedi sezione [SARDINE Pipeline](README.md#sardine-pipeline) nel README.
- **1 Luglio** — avvio pipeline SARDINE: pre-SARDINE archiviato in `legacy/pre-sardine/`; encoder 716 in `src/tinymlinternship/features/`. Daily note: [2026-07-01.md](2026-07-01.md).
- **2 Luglio** — step 1 chiuso (golden FEN, 29 test); **engine v0.1** (HCE + 1-ply search); self-play registrato in **`sardine_game.gif`** via pygame + gifpgn. Daily note: [2026-07-02.md](2026-07-02.md).
- **3 Luglio** — Lc0 subset scaricato (~1.15 GiB, 54k chunk); parser V6→FEN; pipeline preprocess (stats + pilot parquet 1k posizioni, no SF labels). Encoder/engine SARDINE invariati (HCE + 1-ply). Daily note: [2026-07-03.md](2026-07-03.md).

#### Repo work
- Features: `src/tinymlinternship/features/` — 844 encoder (716 base + tactical 128), bucket router, 33+ tests
- Engine: `src/tinymlinternship/engine/` — HCE + `search_best_move` (v0.1)
- Visualization: `src/tinymlinternship/visualization/` + [scripts/record_engine_game.py](scripts/record_engine_game.py) → `sardine_game.gif`
- Notes: [NOTES/Models.md](NOTES/Models.md), [NOTES/chess transformer.md](NOTES/chess%20transformer.md), [NOTES/NNUE.md](NOTES/NNUE.md), [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md), [FIDE & Google Efficient Chess AI Kaggle Challenge.md](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge)
- Data analysis: [scripts/plot_piece_count_distribution.py](scripts/plot_piece_count_distribution.py) → `piece_count_distribution.xlsx`, `piece_count_distribution_10k.xlsx`
- Archive: `legacy/pre-sardine/` (export pipeline, `Wio_TinyValueTest`, checkpoints)

---

#core