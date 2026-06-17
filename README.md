
# Report del Progetto
**Progetto di Tirocinio all'ICTP**

> Dettagli progetto: [PROJECT.md](PROJECT.md)

> Note progetto: [NOTES/notes.md](NOTES/notes.md)

---

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

(See [NOTES/notes.md](NOTES/notes.md) for details.)

---

## 27 Aprile - 3 Maggio
- Xiao Setup
- Edge Impulse
- SenseCraft
- LiteRT
- PyTorch Workflow
- Power Measurement with OTII
- MLSysBook AI Kits

(See [NOTES/notes.md](NOTES/notes.md) for details.)

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

(See [NOTES/notes.md](NOTES/notes.md) for chess/hardware/software notes; [PROJECT.md](PROJECT.md) for model details.)

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

- 12/6 (9:30 - ) - Prima esperienza il lab [export pipeline](export_pipeline.md)
 - Docs: [export_pipeline.md](export_pipeline.md), [PRIVATE/private-notes.md](PRIVATE/private-notes.md) (PRIVATE/ index + #core)
 - Software: ONNX, TorchScript
 - Hardware: Wio Terminal
 - `models\exported\my_tiny_model.ts.pt`
 - Arduino IDE

#### Repo work
 - Main pipeline files: [scripts/prepare_for_arduino.py](scripts/prepare_for_arduino.py) (TFLite export + C header, cleaned with lazy imports), [scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py) (Wio-specific), [scripts/bin_to_c_header.py](scripts/bin_to_c_header.py) (binary to C array)
 - Model files: [models/policy.py](src/tinymlinternship/models/policy.py) (TinyPolicy), [models/value.py](src/tinymlinternship/models/value.py) (TinyValueMLP / UltraTinyValueMLP)
 - Fixes: lazy heavy imports, type TODO cleanup, import style fix in policy.py
 - **Run policy and value functions** by running the example commands in [scripts/run_model.py](scripts/run_model.py).

**Pipeline chain (start-to-finish):**  
- [models/value.py](src/tinymlinternship/models/value.py) (UltraTinyValueMLP) + [scripts/prepare_wio_tiny.py](scripts/prepare_wio_tiny.py) 
- → `models/checkpoints/tiny_value_wio.pt` 
- → `torch.jit.trace` + save → `models/exported/my_tiny_model.ts.pt` (58.4 KB) 
- → [scripts/bin_to_c_header.py](scripts/bin_to_c_header.py) (my_tiny_model.ts.pt --var-name g_chess_model --out ...) 
- → `models/arduino/models/my_tiny_model.h` (g_chess_model + _len=59754 for #include / TFLM)

#### Hardware 
- Connected the Wio to the PC
- `Blink.ino`

---

## 15-21 Giugno

- Wio Terminal value net verification: hand-written forward pass for UltraTinyValueMLP with real FEN input (generated via `scripts/fen_to_c_array.py`) now runs on device.
  - Sketch: `Arduino/Wio_TinyValueTest/Wio_TinyValueTest.ino` (includes `wio_weights.h` + `fen_input.h`, TFT_eSPI LCD using same pattern as Life example).
  - Output on both Serial (115200) and 2.4" LCD (320x240): "Inferred value: -0.056165"
  - Matches PC: `py -3.12 scripts/run_model.py ...` → -0.0562 (within float precision).
  - Parity confirmed. LCD + Serial I/O working.
- Int8 quantization experiment (using prepare_wio_tiny pipeline + wio_int8_weights.h): same 2.16M evals/s as float32 (naive int8 + dequant scales gives no speedup on FPU SAMD51, as expected), but ~4x lower weight memory. Display now includes "Evals/s: 2.16M" and weights filename. See daily note 2026-06-16.md for full log.

---



---

#core
