# Commands

Quick reference for recurring project commands. Run everything from the **project root** unless noted.

**Convention:** `py -3.12` on Windows (use `python3` on Linux/macOS if needed).

---

## Environment

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[viz]"    # pygame + gifpgn — engine game GIF
pip install torch          # legacy export / training only
```

---

## SARDINE engine v0.3 & game GIF (active)

Engine: `src/tinymlinternship/engine/` — alpha-beta + quiescence, pluggable eval (`hce` | `nnue` | `lc0`).  
Frozen bot recipes: **`NOTES/agents/*.md`** (value + search params).  
Visualization: `src/tinymlinternship/visualization/` — pygame + GIF (**gifpgn**).

### Single position — best move

```bash
# HCE (default), depth 1
py -3.12 scripts/run_engine.py --depth 1

# NNUE (844-dim), depth 2
py -3.12 scripts/run_engine.py --eval nnue --depth 2

# Custom checkpoint / FEN
py -3.12 scripts/run_engine.py --eval nnue --nnue-checkpoint models/checkpoints/nnue/pilot_W128_844/best.pt --depth 2
py -3.12 scripts/run_engine.py --fen "4qk2/8/8/8/8/8/8/4R1K1 w - - 0 1" --depth 2
py -3.12 scripts/run_engine.py --moves "e2e4 e7e5" --depth 1
```

### Self-play + GIF (project root)

Writes **`images/sardine_game.gif`** and **`images/sardine_game.pgn`**:

```bash
py -3.12 scripts/record_engine_game.py --headless
py -3.12 scripts/record_engine_game.py --eval nnue --depth 2 --headless --max-plies 80
py -3.12 scripts/record_engine_game.py --eval hce --depth 2 --output images/sardine_game.gif --frame-ms 450
py -3.12 scripts/record_engine_game.py --headless --exporter pygame
```

### NNUE training (smoke only)

Production path = Lichess PGN + Lc0 on-the-fly labels + **nnue-pytorch** (see blueprint §Training pipeline). The commands below validate encoder/engine wiring on ChessBench parquet only:

```bash
pip install -e ".[train]"
py -3.12 scripts/prepare_chessbench_dataset.py
py -3.12 scripts/train_nnue.py --epochs 10 --run-name pilot_W128_844
```

### Bot evaluation — ACPL → Elo (Stockfish)

Blueprint §Bot Evaluation (A1): Stockfish analizza le mosse; **non** è un match win-rate vs avversario.

**Stockfish:** `models/teacher/stockfish/stockfish.exe` (o `--stockfish` / `STOCKFISH_PATH`).

#### Gate depth-1 — self-play + ACPL aggregato

```bash
py -3.12 scripts/eval_bot_acpl.py --eval nnue --depth 1 --max-plies 80 --no-quiescence --sf-movetime-ms 100 --verbose
py -3.12 scripts/eval_bot_acpl.py --eval hce --depth 1 --max-plies 80 --no-quiescence --sf-movetime-ms 100
py -3.12 scripts/eval_bot_acpl.py --pgn plots/nnue_d1_gate.pgn --sf-movetime-ms 100 --json plots/nnue_d1_gate_acpl.json
```

#### Elo per giocatore (bianco e nero separati)

`eval_game_elo.py` — da **PGN** o da lista mosse UCI (`--moves`).

> **Il file di input deve essere in formato PGN** (`.pgn`): testo con header `[Event]`, `[White]`, `[Black]`, `[Result]`, … e mosse in notazione standard. Non usare JSON, CSV o altri formati con `--pgn`.

```bash
# Da PGN (formato obbligatorio per --pgn)
py -3.12 scripts/eval_game_elo.py --pgn plots/nnue_d1_gate.pgn --sf-movetime-ms 100
py -3.12 scripts/eval_game_elo.py --pgn images/sardine_game.pgn --sf-movetime-ms 100 --json plots/game_elo_by_side.json

# Da lista mosse UCI (alternativa senza file)
py -3.12 scripts/eval_game_elo.py --moves "e2e4 e7e5 g1f3 b8c6" --white "BotA" --black "BotB" --sf-movetime-ms 100

# FEN iniziale + mosse UCI
py -3.12 scripts/eval_game_elo.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" --moves "e7e5 g1f3" --sf-depth 16
```

### Teacher 1-ply self-play + GIF (Lc0)

Shallow search with **Lc0** as eval (`go nodes 1` per position, no quiescence). Default `fast` net (~1 s/ply); `bt4` ~10 s/ply — use few plies for a quick demo.

Requires teacher install first: `py -3.12 scripts/download_teacher.py`

```bash
# Default: fast net (791556, ~0.7 s/ply) — prints moves live
py -3.12 scripts/record_teacher_game.py --headless

# Other networks: fast | t1-256 | bt4 | path/to/net.pb.gz
py -3.12 scripts/record_teacher_game.py --headless --network t1-256
py -3.12 scripts/record_teacher_game.py --headless --network bt4 --max-plies 12

# Depth 2
py -3.12 scripts/record_teacher_game.py --headless --depth 2 --max-plies 100

# Shorter game / custom output
py -3.12 scripts/record_teacher_game.py --headless --max-plies 12 --output images/teacher_1ply_game.gif --frame-ms 500

# Smoke: one move from startpos (~11 s on CPU)
py -3.12 scripts/bench_teacher_move.py
```

```bash
# Faster bot
py -3.12 scripts/record_hf_game.py --headless --depth 1 --max-plies 48
```

More SARDINE snippets (encoder tests, manual checks): [SARDINE commands.md](SARDINE%20commands.md).

---

## Data

Download the Kaggle chess dataset into `data/`:

```bash
py -3.12 scripts/download_data.py
```

Download curated Lc0 training shards (~1.2 GiB default subset) into `data/raw/lc0/`:

```bash
py -3.12 scripts/download_lc0.py --dry-run   # plan only
py -3.12 scripts/download_lc0.py             # download + extract .gz chunks
py -3.12 scripts/download_lc0.py --list      # curated shard catalog
py -3.12 scripts/download_lc0.py --max-gb 1  # stop after first shard
```

Lc0 preprocessing (parse → filter → sample; run stats before Lc0 labeling):

```bash
py -3.12 scripts/stats_lc0_processed.py --max-chunks 80 --max-records 30000
py -3.12 scripts/prepare_lc0_dataset.py --max-chunks 120 --total 10000
py -3.12 scripts/smoke_test_lc0_chunk.py
```

Regenerate the Stockfish 16 dataset
```bash
py -3.12 scripts/prepare_chessbench_dataset.py
```

---

## Wio sketch — regenerate headers (legacy)

Sketch: `legacy/pre-sardine/Arduino/Wio_TinyValueTest/`. After regenerating headers, re-upload from Arduino IDE (Board: **Seeed Wio Terminal**, Serial: **115200**).

### FEN → `fen_input.h`

768-element binary board vector (12 planes × 8 × 8). Encoding lives in `legacy/pre-sardine/src/tinymlinternship/datasets/featurizer.py`.

```bash
py -3.12 legacy/pre-sardine/scripts/fen_to_c_array.py "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1" --output legacy/pre-sardine/Arduino/Wio_TinyValueTest/fen_input.h
```

Replace the quoted FEN with any legal position.

### int8 weights → `wio_int8_weights_*.h`

Each script writes the C header into `legacy/pre-sardine/Arduino/Wio_TinyValueTest/`, plus checkpoints/export under `legacy/pre-sardine/models/`.

| Script | Architecture | Default header |
|--------|-------------|----------------|
| `prepare_wio_nano.py` | 768→16→8→1 | `wio_int8_weights_nano.h` |
| `prepare_wio_tiny.py` | 768→32→16→1 | `wio_int8_weights_tiny.h` |
| `prepare_wio_small.py` | 768→64→32→1 | `wio_int8_weights_small.h` |
| `prepare_wio_medium.py` | 768→128→64→1 | `wio_int8_weights_medium.h` |
| `prepare_wio_big.py` | 768→256→64→1 | `wio_int8_weights_big.h` |
| `prepare_wio_huge.py` | 768→512→64→1 | `wio_int8_weights_huge.h` |

```bash
# Default (random init, export int8 header)
py -3.12 legacy/pre-sardine/scripts/prepare_wio_nano.py
py -3.12 legacy/pre-sardine/scripts/prepare_wio_tiny.py
py -3.12 legacy/pre-sardine/scripts/prepare_wio_small.py
py -3.12 legacy/pre-sardine/scripts/prepare_wio_medium.py
py -3.12 legacy/pre-sardine/scripts/prepare_wio_big.py
py -3.12 legacy/pre-sardine/scripts/prepare_wio_huge.py

# Custom checkpoint name
py -3.12 legacy/pre-sardine/scripts/prepare_wio_tiny.py --name tiny_value_wio_tiny_int8

# Train on Lichess CSV, then export
py -3.12 legacy/pre-sardine/scripts/prepare_wio_tiny.py --train --epochs 2 --max-games 600
```

### Switch active model on device

Edit `WEIGHTS_FILE` in `legacy/pre-sardine/Arduino/Wio_TinyValueTest/config.h`, then recompile and upload:

```c
#define WEIGHTS_FILE "wio_int8_weights_nano.h"
```

---

## PC inference & parity check (legacy)

Compare sketch output against the same FEN on the laptop. Use the checkpoint produced by the matching `prepare_wio_*` script.

```bash
# Value net (Wio path)
py -3.12 legacy/pre-sardine/scripts/run_model.py --checkpoint legacy/pre-sardine/models/checkpoints/tiny_value_wio_nano_int8.pt --type value --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"

# Policy net
py -3.12 legacy/pre-sardine/scripts/run_model.py --checkpoint legacy/pre-sardine/models/checkpoints/tiny_chess_policy_lab.pt --type policy --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" --top-k 3
```

Pipeline wrapper (same inference, configurable):

```bash
py -3.12 legacy/pre-sardine/scripts/run_pipeline.py --model-name my_tiny_model --model-type value --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" --stage run
```

---

## Model sizing

Parameter counts and estimated Wio flash usage for all value-net variants:

```bash
py -3.12 legacy/pre-sardine/scripts/count_model_params.py
```

PC-side eval throughput (sanity check, not device):

```bash
py -3.12 legacy/pre-sardine/scripts/benchmark_rate.py
```

---

## Export pipeline (TorchScript / TFLite)

Full walkthrough: [export_pipeline.md](../export_pipeline.md).

### Create a random checkpoint

```bash
py -3.12 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
import torch
from tinymlinternship.models.value import UltraTinyValueMLP

model = UltraTinyValueMLP().eval()
Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), 'models/checkpoints/my_tiny_model.pt')
print('Created models/checkpoints/my_tiny_model.pt')
"
```

### Export to TorchScript

```bash
py -3.12 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
import torch
from tinymlinternship.models.value import UltraTinyValueMLP

model = UltraTinyValueMLP().eval()
model.load_state_dict(torch.load('models/checkpoints/my_tiny_model.pt', map_location='cpu'))
dummy = torch.randn(1, 768)
traced = torch.jit.trace(model, dummy)
Path('models/exported').mkdir(parents=True, exist_ok=True)
traced.save('models/exported/my_tiny_model.ts.pt')
print('Created models/exported/my_tiny_model.ts.pt')
"
```

### Binary → C header (generic embed)

```bash
py -3.12 scripts/bin_to_c_header.py models/exported/my_tiny_model.ts.pt --var-name g_chess_model --out models/arduino/models/my_tiny_model.h
```

### TFLite path (XIAO / TFLM sketches)

```bash
py -3.12 scripts/prepare_for_arduino.py --model-name tiny_policy_v0.1
py -3.12 scripts/prepare_for_arduino.py --model-name tiny_policy_v0.1 --quantize dynamic
py -3.12 scripts/prepare_for_arduino.py --model-name tiny_policy_v0.1 --quantize int8
py -3.12 scripts/prepare_for_arduino.py --from-tflite models/exported/tiny_policy_v0.1.tflite
```

---

## Legacy float32 weights header

Older float32 `wio_weights.h` for hand-written forward pass (superseded by int8 headers):

```bash
py -3.12 legacy/pre-sardine/scripts/generate_wio_weights.py
py -3.12 legacy/pre-sardine/scripts/generate_wio_weights.py --checkpoint models/checkpoints/my_tiny_model.pt --output legacy/pre-sardine/Arduino/Wio_TinyValueTest/wio_weights.h
```

---

## Tests

**SARDINE (active):** see [SARDINE commands.md](SARDINE%20commands.md).

```bash
py -3.12 -m pytest tests/test_features.py -v
py -3.12 -m pytest tests/test_engine.py tests/test_visualization.py -v
py -3.12 -m pytest -v
```

**Legacy policy inference:**

```bash
py -3.12 -m pytest legacy/pre-sardine/tests/test_policy_inference.py -v
```

---

## Arduino IDE (Wio Terminal)

1. Boards Manager (`Ctrl+Shift+B`) → install **Seeed SAMD Boards**
2. **Tools → Board → Seeed Wio Terminal**
3. **Tools → Port →** your COM port
4. Open `legacy/pre-sardine/Arduino/Wio_TinyValueTest/Wio_TinyValueTest.ino` → Verify → Upload

More setup notes: [arduino.md](Arduino.md).

---

[← Back to Notes index](_notes.md)