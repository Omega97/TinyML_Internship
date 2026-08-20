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

(See [NOTES/_notes.md](../_notes.md) for details.)

---

## 27 Aprile - 3 Maggio
- Xiao Setup
- Edge Impulse
- SenseCraft
- LiteRT
- PyTorch Workflow
- Power Measurement with OTII
- MLSysBook AI Kits

(See [NOTES/_notes.md](../_notes.md) for details.)

#### Repo work
- basic repo structure
- [download_data.py](../../scripts/download_data.py)
- [test_data.py](../../tests/test_data.py)
- `example_game.py`

---

## 4-10 Maggio
- FEN
- Value functions
- Policy functions

(See [NOTES/_notes.md](../_notes.md) for chess/hardware/software notes; [PROJECT.md](../../PROJECT.md) for model details.)

#### Repo work
- `example_fen.py`
- `test_policy_inference.py`
- `featurizer.py` (removed; features now in [encoder.py](../../src/tinymlinternship/features/encoder.py))

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

- 12/6 - **Prima esperienza il lab** -  export pipeline
 - Docs: `export_pipeline.md`, [PRIVATE/private-notes.md](../../PRIVATE/private-notes.md) (PRIVATE/ index + #core)
 - Software: ONNX, TorchScript
 - Hardware: Wio Terminal
 - `models\exported\my_tiny_model.ts.pt`
 - Arduino IDE

#### Repo work
 - Main pipeline files: `scripts/prepare_for_arduino.py` (TFLite export + C header, cleaned with lazy imports), `scripts/prepare_wio_tiny.py` (Wio-specific), `scripts/bin_to_c_header.py` (binary to C array)
 - Model files: `models/policy.py` (TinyPolicy), `models/value.py` (TinyValueMLP / UltraTinyValueMLP)
 - Fixes: lazy heavy imports, type TODO cleanup, import style fix in policy.py
 - **Run policy and value functions** by running the example commands in `scripts/run_model.py`.

**Pipeline chain (start-to-finish):**  
- `models/value.py` (UltraTinyValueMLP) + `scripts/prepare_wio_tiny.py` 
- → `models/checkpoints/tiny_value_wio.pt` 
- → `torch.jit.trace` + save → `models/exported/my_tiny_model.ts.pt` (58.4 KB) 
- → `scripts/bin_to_c_header.py` (my_tiny_model.ts.pt --var-name g_chess_model --out ...) 
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
- Input helper: `fen_to_c_array.py` → `Arduino/Wio_TinyValueTest/fen_input.h`
- Model: UltraTinyValueMLP (`768→32→16→1`)
- Export: `generate_wio_weights.py` (float32 `wio_weights.h`), `prepare_wio_tiny.py` (int8 `wio_int8_weights_tiny.h`)
- PC parity: `run_model.py`
- Device sketch: `Arduino/Wio_TinyValueTest/Wio_TinyValueTest.ino` (hand-written forward pass, TFT + Serial)

---

## 22-28 Giugno

- **Ricerca modelli su Hugging Face** — esplorati checkpoint e architetture open-source per scacchi (AlphaZero-style CNN, ResNet, transformer, NNUE) per capire dimensioni, input encoding e policy/value head rispetto ai limiti Wio.
- **Measuring memory and time to run the NNs correctly!** Extended the Wio value-net performance matrix to **big** (`768→256→64→1`) and **huge** (`768→512→64→1`); full nano→huge sweep now fits on device (huge at ~96% flash).
- **Benchmark honesty fix:** the flat ~2.01M evals/s across all models was a measurement artifact — `-Os` dead-code elimination removed `forward()` from `loop()`. Fixed with `volatile forwardSink`, interval-based EMA rate, and 1s warm-up discard; throughput now scales with model size (~2× latency per tier: nano 1.4 ms → huge 45 ms).
- **Sketch refactor:** split `Wio_TinyValueTest` into `config.h`, `Int8ValueNet`, `WioBoard`, `Benchmark`; weights included once in `Int8ValueNet.cpp` (fixes 3× PROGMEM duplication that overflowed huge). Sparse L1 skips `pgm_read_byte` on empty board squares.
- **24/6 lab session:** optimized forward pass (~15% faster overall; nano 1.8→1.4 ms/call); removed misleading `K` display suffix; updated [NOTES/Performance.md](../Performance.md) with honest latency/evals/s table and hw–sw synergy notes (flash bus stalls dominate, not FPU). See daily notes [2026-06-22.md](../../DAILY-NOTES/2026-06/2026-06-22.md), [2026-06-24.md](../../DAILY-NOTES/2026-06/2026-06-24.md).

#### Repo work
- Models: `BigValueMLP / HugeValueMLP` (nano→huge family)
- Export scripts: `prepare_wio_big.py`, `prepare_wio_huge.py`, `count_model_params.py`
- Device sketch: `Arduino/Wio_TinyValueTest/` (modular int8 forward + benchmark; later under `legacy/pre-sardine/`, now removed)

---

## 29 Giugno - 5 Luglio

*Daily notes: [06-29](../../DAILY-NOTES/2026-06/2026-06-29.md), [06-30](../../DAILY-NOTES/2026-06/2026-06-30.md), [07-01](../../DAILY-NOTES/2026-07/2026-07-01.md), [07-02](../../DAILY-NOTES/2026-07/2026-07-02.md), [07-03](../../DAILY-NOTES/2026-07/2026-07-03.md); 07-04 vuota.*

- **Catalogo modelli** — consolidata la ricerca HF in [NOTES/Models.md](../Models.md): Dense/Conv, ResNet, Transformer, NNUE, Lc0 edge; i modelli HF (8–100M params) restano fuori budget Wio.
- **Transformer compatto** — [NOTES/chess transformer.md](../chess%20transformer.md): policy+value **~210K** params (`24×8×8`, 2 blocchi) — ~165× più piccolo di ChessBot.
- **FIDE & Google Challenge** — top writeup sotto 5 MiB RAM / ≤64 KiB binario: micro-NNUE, king mirroring, geometric pruning, SPSA. [Kaggle](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge).
- **NNUE deep-dive** — [NOTES/NNUE.md](../NNUE.md); piece-count study su Lichess (`piece_count_distribution*.xlsx`) per i bucket.
- **30/6 — Blueprint SARDINE locked** — scelte componente in [SARDINE Engine Blueprint.md](../SARDINE%20Engine%20Blueprint.md) + [design options](../SARDINE%20design%20options.md): pure C target, micro-NNUE, alpha-beta ladder, int8/CReLU, TT-dominant RAM, Lc0 data.
- **1/7 — Pipeline SARDINE avviata** — pre-SARDINE archiviato in `legacy/pre-sardine/`; encoder **716** (`index_map`, `mirror`, `encoder`, `bucket`) in `src/tinymlinternship/features/`.
- **2/7 — Step 1 gate + engine v0.1** — golden FEN (29 test features); HCE + 1-ply; self-play GIF `sardine_game.gif` (pygame + gifpgn).
- **3/7 — Search v0.3 + dati Lc0** — perft d5 startpos; alpha-beta + capture quiescence; download Lc0 ~1.15 GiB / 54 866 chunk; parser V6→FEN; pilot `positions.parquet` (no label SF). **66 test** passanti a fine giornata.

#### Repo work
- Features: `src/tinymlinternship/features/` — 716 dual-POV + 8 queen-split buckets
- Engine: `engine/eval_hce.py`, `search.py` (v0.3), `perft.py`
- Visualization + [scripts/record_engine_game.py](../../scripts/record_engine_game.py)
- Data: [scripts/download_lc0.py](../../scripts/download_lc0.py), `prepare_lc0_dataset.py`, [NOTES/Lc0 preprocessing pipeline.md](../Lc0%20preprocessing%20pipeline.md)
- Notes: Models, chess transformer, NNUE, Blueprint, FIDE challenge; piece-count plots

---

## 6-12 Luglio

*Daily notes: [07-06](../../DAILY-NOTES/2026-07/2026-07-06.md), [07-07](../../DAILY-NOTES/2026-07/2026-07-07.md), [07-08](../../DAILY-NOTES/2026-07/2026-07-08.md), [07-10](../../DAILY-NOTES/2026-07/2026-07-10.md).*

Settimana del **primo NNUE pilot**, del **gate ACPL** e dell'encoder tattico **844**.

- **6/7 — Teacher + ChessBench** — install Lc0 teacher (`download_teacher.py`, rete `fast`/791556); bench HF (chess_lite ~2 ms/eval ma gioco debole a d1); ChessBench test split → **62 829** pos con `win_prob` SF16; pipeline `prepare_chessbench_dataset.py` → train/val parquet con sparse 716 + `expected_reward`. Labeling UCI batch deprioritizzato (lento). **73 test**.
- **7/7 — Train pilot + ACPL stack** — `BucketedNNUE` + `train_nnue.py`: pilot ChessBench W=128, val_mse **0.058** → poi **844-dim** tactical (`under-attack` + `king-attacker`) retrain `pilot_W128_844` val_mse **0.056**. Hook `--eval nnue` su engine; GIF NNUE d2. Blueprint training pipeline (Lc0 labels, nnue-pytorch path, prune, PTQ). **ACPL gate:** `bot_eval/` + Stockfish 18; NNUE d1 **ACPL 121.1** → Elo euristico **~1644**. **92 test**.
- **8/7 — Ladder d1 + GIF d2** — baseline HCE ACPL **275**; Sunfish d1 **818** (clone + `sunfish_selfplay_pgn.py`). Ordine: **NNUE ≪ HCE < Sunfish**. GIF `hce_d2_game.gif`; progress bar su `record_engine_game.py`; qsearch HCE d2 blowup documentato (usare `--no-quiescence` per HCE d2).
- **10/7 — Label smoke + multi-game gates** — `label_positions.py` (Lc0 WDL → `expected_reward`); smoke startpos + ChessBench10. Suite **94 passed**. Gate **16 partite** d1: NNUE Elo **~1465**, HCE/Sunfish floor **400**. Gate d2 no-qsearch: HCE Elo **~2610**, **NNUE collapse** (ACPL ~1588) — collasso noto del pilot a depth 2.

#### Repo work
- NNUE: `src/tinymlinternship/nnue/` (`model.py`, `dataset.py`), checkpoints `models/checkpoints/nnue/pilot_W128_*`
- Features: `tactical.py` → **844** dim; tests tattici
- Eval: `bot_eval/acpl.py`, [scripts/eval_bot_acpl.py](../../scripts/eval_bot_acpl.py); artefatti `images/plots/*_d1_gate*`, `*_d2_gate*`
- Teachers: Lc0, Sunfish under `models/teacher/`; Stockfish install (poi PATH-only)
- Scripts: `download_teacher.py`, `download_chessbench.py`, `prepare_chessbench_dataset.py`, `label_positions.py`, `sunfish_selfplay_pgn.py`, `plot_nnue_architecture.py`
- Agent cards: `NOTES/agents/`; skill `AI-SKILLS/sardine-repo/`

#### Gate depth-1 (riepilogo, 16 partite dove applicabile)

| Eval | ACPL (ord.) | Elo euristico |
| ---- | ----------- | ------------- |
| NNUE `pilot_W128_844` d1 | ~139 | **~1465** |
| HCE d1 | ~357 | ~400 |
| Sunfish d1 | ~1038 | ~400 |
| HCE d2 no-qsearch | ~24.5 | **~2610** |
| NNUE d2 | collapse | ~400 |

---

## 13-19 Luglio

*Daily notes: [07-16](../../DAILY-NOTES/2026-07/2026-07-16.md), [07-17](../../DAILY-NOTES/2026-07/2026-07-17.md); [07-18](../../DAILY-NOTES/2026-07/2026-07-18.md) solo idea.*

Settimana del **path produzione dati** (Lichess primary + Lc0 supplement, label uniformi Lc0).

- **16/7 — Extract + schema ASSETS** — `lichess_pgn_to_fen.py` su smoke 50 PGN → **2371** FEN, tutti e 8 i bucket (bucket 5 raro). Label smoke 50 pos (~6 pos/s). Schema production: `schema.py`, `merge_training_sets.py`, prelabel columns allineate ASSETS; extract non etichettato Lichess **2371** + Lc0 **3149**. README demo GIF. **100 test**.
- **17/7 — Labeling uniforme mini** — stesso teacher `791556.pb.gz` su entrambi i blocchi: Lichess **2371** (~14 pos/s), Lc0 **3149** (~13 pos/s); fix WDL sintetico su posizioni terminali. Totale ~7 min wall. Merge/train smoke lasciati al giorno successivo.
- **18/7** — idea parcheggiata: bit di presenza pezzi iniziali (+ extra per promozioni) come feature di eval.

#### Repo work
- [scripts/lichess_pgn_to_fen.py](../../scripts/lichess_pgn_to_fen.py), [scripts/label_positions.py](../../scripts/label_positions.py), [scripts/merge_training_sets.py](../../scripts/merge_training_sets.py)
- `src/tinymlinternship/data/schema.py` + `tests/test_dataset_schema.py`
- Labeled: `data/processed/labeled/lichess_labeled.parquet`, `lc0_labeled.parquet` (+ smoke intermediates)
- Prelabel: `data/processed/lichess/positions.parquet`, `data/processed/lc0/positions.parquet`
- Policy: [ASSETS.md](../../ASSETS.md) (uniform `expected_reward` only)

---

## 20-26 Luglio

*Daily notes: [07-20](../../DAILY-NOTES/2026-07/2026-07-20.md), [07-21](../../DAILY-NOTES/2026-07/2026-07-21.md), [07-22](../../DAILY-NOTES/2026-07/2026-07-22.md).*

Settimana del **mini production set end-to-end** e del **project reassessment** (cleanup repo + docs).

- **20/7 — Merge + smoke train production** — `merge_training_sets.py` → `labeled/train.parquet` **5306** / `val.parquet` **214** (split by `game_id`, un solo teacher). Loader patch: encode-on-the-fly da FEN se mancano sparse features. Smoke train `smoke_prod_W128_844` 2 ep: val_mse **0.247** (vs pilot ChessBench 0.056 — atteso: mini set + poche epoch). **106 test**. Pipeline C–E smoke chiusa; full Lichess volume ancora aperta.
- **21/7** — piano train lungo mini set + ACPL sul nuovo ckpt; in sessione: integrazione idee thesis (task vectors / optimal bucketing) in blueprint §Later / pipeline D2. Train lungo e gate strength **non eseguiti** (carry-over).
- **22/7 — Reassessment cleanup** — decisioni A4…J1: hard-delete `legacy/pre-sardine/`; Stockfish **PATH-only** (no tree in-repo); drop HF weights teacher; daily notes solo sotto `DAILY-NOTES/`; Project Report → `NOTES/archive/`; refresh `PROJECT.md`, `Goal.md`, `TODOs.md`, `ASSETS.md`, `ai-feed.md`. ChessBench smoke tenuto; 8 bucket fino a ablation §D.

#### Repo work
- Checkpoints: `models/checkpoints/nnue/smoke_prod_W128_844/`
- Dataset loader: `nnue/dataset.py` (FEN → `encode_dual`)
- Docs: PROJECT / Goal / TODOs / ASSETS / README / Commands / Models; blueprint G3 interim
- Cleanup: no `legacy/pre-sardine/`, no in-repo Stockfish tree, no `models/teacher/hf/`

---

## 27 Luglio - 2 Agosto

*Daily notes: [07-31](../../DAILY-NOTES/2026-07/2026-07-31.md), [08-01](../../DAILY-NOTES/2026-08/2026-08-01.md). Nessuna nota 27–30/7 né 2/8.*

Settimana di **baseline Cfish** (riferimento forte PC); confronto ACPL Cfish vs SARDINE ancora aperto.

- **31/7 — Cfish in repo** — sorgenti + binario + stock net `nn-62ef826d1a6d.nnue` in `src/cfish/`; launcher `run-cfish.bat` (cwd corretto per `EvalFile`). Smoke UCI: **`uciok` / `readyok`**. Self-play Cfish + ACPL Stockfish **non chiusi** in sessione.
- **1/8** — piano: chiudere self-play/ACPL Cfish e confrontare con NNUE SARDINE (`--eval nnue`); **execution log vuoto** (sessione non eseguita o non loggata).
- **2/8** — nessuna daily note; stato prodotto invariato rispetto al 31/7 sul gate Cfish.

#### Repo work
- `src/cfish/` (full Cfish tree + `cfish.exe` + stock NNUE)
- [run-cfish.bat](../../run-cfish.bat), [NOTES/Cfish.md](../Cfish.md)
- Aperti a fine periodo: Cfish self-play + ACPL; formal `go depth 5` + nps archiviato; fix path `scripts/cfish.py`; single-head train F3; full Lichess; Wio port

#### Stato a 2/8 (sintesi verso l'obiettivo del report)

| Area | Stato |
| ---- | ----- |
| PC encoder + search | ✅ 844, αβ + qsearch, MVV-LVA |
| Pilot / mini NNUE | ✅ ChessBench pilot + mini teacher-labeled smoke |
| ACPL judge (Stockfish) | ✅ HCE / NNUE / Sunfish gates; **Cfish ACPL** ❌ |
| Dati production volume | ❌ solo mini (~5.3k train); dump Lichess full open |
| Device Wio / Elo on-device | ❌ non ripreso post–cleanup pre-SARDINE |

---

#core
