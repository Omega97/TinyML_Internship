
# Candidate Teacher Models


## Top Recommendation: Lc0 Networks

Lc0 (Leela Chess Zero) is the most natural choice. Its value head is explicitly trained to predict expected game outcome.

**What it outputs**: Since v0.21 (Feb 2019), Lc0's value head predicts **Win/Draw/Loss probabilities** (three numbers summing to 1.0) rather than a single scalar. The MCTS converts this to a single Q value via `Q = Win - Loss`.

**Latest developments**: v0.30.0 (July 2023) introduced WDL rescaling with an Elo-based transformation, making predictions more realistic across different playing strengths. It also added a `WDL_mu` score type that follows the same convention as Stockfish: **+1.00 means 50% white win chance**.

**Where to get networks**: https://training.lczero.org/ — all trained networks are available for download.

**How to use as teacher**:
```python
# Using lc0 Python bindings or UCI interface
# Position → value head → WDL probabilities → expected reward = W - L
```


## Stockfish NNUE (with WDL output)

Stockfish itself doesn't output expected reward directly — its NNUE outputs centipawns. **However**, Stockfish has a built-in `UCI_ShowWDL` option that converts its internal evaluation to WDL probabilities using a **win-rate model**.

**What it outputs**: Win/Draw/Loss percentages via UCI.

**How to use as teacher**:
```python
import chess
import subprocess

engine = subprocess.Popen(["stockfish"], ...)
engine.write(b"uci\n")
engine.write(b"setoption name UCI_ShowWDL value true\n")
engine.write(b"position fen ...\n")
engine.write(b"eval\n")
# Parse WDL from output
```

**Caveat**: The WDL values are a post-hoc conversion, not the network's native output. The conversion is based on Elo differences and may not be as directly calibrated as Lc0's value head.


## Hugging Face Models (PyTorch)

Several open-source PyTorch models output value in [-1, 1] and are ready to use:

### 1. **chess_lite** (satana123/chess_lite)
- **Architecture**: 15-channel CNN with batch normalization
- **Value Head**: Scalar evaluation from -1 (Loss) to +1 (Win), normalized via tanh
- **Training**: Reinforcement learning + Stockfish 16.1 evaluation
- **License**: Apache 2.0
- **Link**: https://huggingface.co/satana123/chess_lite

### 2. **Artoria Zero** (Shinapri/artoria-zero)
- **Architecture**: Decoder-only transformer (LLaMA-style) with dual policy + value heads
- **Value Head**: Position evaluation with tanh output in [-1, 1]
- **Training**: Behavioral cloning on Lichess games
- **Variants**: Small (~19M params), Mid (~100M), Large (~500M)
- **License**: MIT
- **Link**: https://huggingface.co/Shinapri/artoria-zero

### 3. **AlphaZero-style PyTorch implementations**
Several GitHub repos implement AlphaZero with value heads in [-1, 1]:
- https://github.com/ns-1456/AlphaZero-Chess
- https://github.com/lipeeeee/lunachess


## Comparison Table

| Network               | Output          | Local path                                                      | Ruolo                                    |
| --------------------- | --------------- | --------------------------------------------------------------- | ---------------------------------------- |
| **Lc0 791556 (fast)** | WDL permille    | `models/teacher/lc0/791556.pb.gz`                               | ✅ **Default** — bot, demo, smoke         |
| **Lc0 T1-256**        | WDL permille    | `models/teacher/networks/t1-256x10-distilled-swa-2432500.pb.gz` | ✅ Installato — alternativa CPU           |
| **Lc0 BT4**           | WDL permille    | `models/teacher/networks/BT4-1024x15x32h-swa-6147500.pb.gz`     | ✅ Installato — qualità ref, troppo lento |
| **Stockfish + WDL**   | WDL (converted) | —                                                               | Fallback, non installato                 |
| **chess_lite**        | Scalar `[-1,1]` | `models/teacher/hf/chess_lite/chess_lite.pth`                   | ✅ Installato — veloce, **gioco debole** |
| **Artoria Zero**      | Scalar `[-1,1]` | `models/teacher/hf/artoria-zero/small/checkpoint.pt`            | ✅ Installato (small) — più lento di lite |


## Decision (2026-07-06, agg. depth-2 playtest) — teacher v1

**Famiglia: Lc0 value head** via `lc0` UCI + WDL permille. Rete **operativa: `fast` (791556)**; BT4 tenuto come riferimento qualità.

| | |
|---|---|
| Label (se mai custom) | `expected_reward = (W − L) / 1000` · side to move → White POV in code |
| **Dataset training** | **Solo dump pre-etichettati** — labeling live scartato; vedi [Datasets.md](Datasets.md) |
| Bot / baseline (Lc0) | `scripts/record_teacher_game.py` · `--network fast\|t1-256\|bt4` · `--depth 1\|2` |
| Bot / baseline (HF) | `scripts/record_hf_game.py` · `--model auto\|chess_lite\|artoria` · `--depth 1` |
| Codice eval | `eval_lc0.py` (`Lc0Teacher`) · `eval_chess_lite.py` (`ChessLiteEvaluator`) |
| Fallback | Stockfish `UCI_ShowWDL` — **non installato** |

**Rationale:** WDL nativo allineato al blueprint — **teacher per training** resta Lc0. **Depth-2** gioca nettamente meglio di depth-1 (depth-1 ≈ 300 Elo), ma resta lento in partita reale (~1 min/ply con `fast`). **chess_lite** è utile come smoke inference (~2 ms/eval, 400 ply in secondi) ma gioca **molto male** a depth-1 — non sostituisce Lc0 per demo qualità. Per **training NNUE**: dataset pre-etichettato (ChessBench test scaricato; train 1.1 TB opzionale).

**Installazione Lc0:** `py -3.12 scripts/download_teacher.py` (BT4 + lc0) · T1-256 via `scripts/bench_teacher_nets.py` o download manuale · `791556` già nello zip lc0.

**Installazione HF:** `py -3.12 scripts/download_hf_teacher.py` (chess_lite + artoria-small).


## Locally installed models (survey 2026-07-06, agg. HF bench)

Path relativi alla root. Costanti: `settings.py` → `LC0_*`, `CHESS_LITE_WEIGHTS`, `ARTORIA_SMALL_CKPT`.

### Teacher Lc0 — ✅ installato (3 reti + binario)

**Binario condiviso**

| File | Path | Size |
|------|------|------|
| **lc0.exe** | `models/teacher/lc0/lc0.exe` | 2.1 MB |
| OpenBLAS DLL | `models/teacher/lc0/libopenblas.dll` | 19.5 MB |
| Release zip | `models/teacher/lc0/lc0-v0.32.1-windows-cpu-openblas.zip` | 22 MB |
| Manifest | `models/teacher/manifest.json` | 675 B |

**Reti value (preset `--network` in `record_teacher_game.py`)**

| Preset | Path | Size | Eval singola | 1-ply (startpos) | Note |
|--------|------|------|--------------|------------------|------|
| **`fast`** (default) | `models/teacher/lc0/791556.pb.gz` | 18 MB | ~12 ms | ~0.5–1 s/ply | Bot baseline, GIF; ~300 Elo a depth=1 |
| `t1-256` | `models/teacher/networks/t1-256x10-distilled-swa-2432500.pb.gz` | 35 MB | ~36 ms | ~1 s/ply | Alternativa leggermente più forte |
| `bt4` | `models/teacher/networks/BT4-1024x15x32h-swa-6147500.pb.gz` | 320 MB | ~600 ms | ~11–23 s/ply | Qualità max; impraticabile per play/label |

**Integrazione:** `Lc0Teacher(weights=…)` · UCI `UCI_ShowWDL` + `go nodes 1` · default weights = `LC0_NETWORK_FAST`.

### Shallow-search baseline (misurato 2026-07-06)

| Depth | Rete | Tempo/ply | Nodi (startpos) | Forza | Uso |
|-------|------|-----------|-----------------|-------|-----|
| 1 | fast | **~0.7–1 s** | ~20 | ~300 Elo vs random | Demo veloce, sanity |
| 2 | fast | **~1 min** in partita · ~4–5 s startpos | ~190 | **molto migliore** di depth-1 | Demo qualità; impraticabile per labeling |
| 1 | bt4 | ~10–23 s | — | teacher più calibrato | Solo spot-check |

Depth-2 già supportato: `--depth 2` (alpha-beta esistente, no qsearch). Il tempo esplode in posizioni reali (ramificazione × eval lc0 per nodo); startpos è un lower bound. GIF esempio depth-1: `images/teacher_1ply_game.gif`.

### Teacher HF PyTorch — ✅ installato (chess_lite + artoria-small)

Confronto inferenza CPU (eval singola, startpos, 2026-07-06):

| Modello | Path | Size | Eval/pos | 1-ply (40 ply) | Play strength depth-1 |
|---------|------|------|----------|----------------|------------------------|
| **chess_lite** | `models/teacher/hf/chess_lite/chess_lite.pth` | 39 MB | **~1.8 ms** | ~6 s | ❌ **Debole** — mosse ripetitive / anti-tattiche |
| artoria-small | `models/teacher/hf/artoria-zero/small/checkpoint.pt` | 101 MB | ~6.4 ms | ~20 s | Non testato in partita lunga |

**Playtest chess_lite (2026-07-06):** `record_hf_game.py --depth 1 --max-plies 400` — partita completa in pochi secondi, ma qualità di gioco **orribile** (nonostante value head allenata con SF 16.1). Conclusione: velocità ≠ forza; utile solo per **latency smoke**, non per baseline Elo.

**Integrazione:** `ChessLiteEvaluator` in `eval_chess_lite.py` · `record_hf_game.py --model auto` · architettura `BossChessNet` ricostruita da state dict.

GIF esempio: `images/games/hf_chess_lite_depth1_20260706_143456.gif` (40 ply).

**Limiti HF vs Lc0:** eval tanh `[-1,1]` (non WDL); play strength ≪ Lc0 anche a parità depth; **non teacher per training NNUE** (target blueprint = expected reward da WDL).

### Dataset posizioni–evaluations — serve qualcosa di già pronto

| Approccio | Verdetto |
|-----------|----------|
| `label_positions.py` + lc0 UCI | **❌ Scartato** — depth-1 ≈ 1 s/pos, depth-2 ≈ 1 min/pos in partita → giorni/settimane per training set |
| Lc0 chunk `best_q` / `root_q` | ⚠️ Parziale — già nel raw data locale, ma net di generazione ≠ teacher uniforme |
| **ChessBench** (DeepMind, SF 16) | **✅ Scaricato (test)** — vedi sotto; train 15.3B action-values, serve conversione label |
| Altri dump (Lichess/Lc0 HF) | ⚠️ Survey in [Datasets.md](Datasets.md) |

#### ChessBench — ✅ test split locale (2026-07-06)

Fonte: [google-deepmind/searchless_chess](https://github.com/google-deepmind/searchless_chess) · mirror HF: `heidar-an/ChessBench`.

| | |
|---|---|
| **Path locale** | `data/raw/chessbench/test/` |
| **File scaricati** | `state_value_data.bag` (4.4 MB, 62 829 pos) · `action_value_data.bag` (141 MB, ~1.8M action-values test) |
| **Download** | `py -3.12 scripts/download_chessbench.py` · ispezione: `scripts/study_chessbench.py` |
| **Formato Research** | `.bag` (Apache Beam tuple coder) — non Parquet |

**Tre varianti nel formato originale:**

| Variante | Record | Campi |
|----------|--------|-------|
| `state_value` | `(fen, win_prob)` | eval posizione |
| `action_value` | `(fen, move_uci, win_prob)` | eval dopo mossa legale |
| `behavioral_cloning` | `(fen, move_uci)` | solo policy oracle |

**Tipo di valore — analisi (paper §2.1, verificato su test split):**

| Domanda | Risposta |
|---------|----------|
| Centipawns grezzi? | **No** — il dataset espone solo `win_prob` (float64) |
| Expected value WDL (`W−L`)? | **No** — è **probabilità di vittoria** ∈ [0, 1], non WDL completo |
| Normalizzato `[-1,1]`? | **No** — è win% in frazione [0, 1] (0 = 0%, 1 = 100%) |
| Come si ottiene? | SF 16 analizza 50 ms/pos (o /coppia pos-mossa); legge **cp** interni → formula Lichess: `win_prob = 1 / (1 + exp(−0.00368208 · cp))` · **mate → 1.0** |
| Prospettiva | `score.relative` Stockfish → **side to move** |
| Distribuzione test (5k campione) | state: mean ≈ 0.51 · action: mean ≈ 0.35 · molti 0.0 e 1.0 (posizioni decisive) |

**Conversione verso SARDINE (approssimata):** `expected_reward ≈ 2·win_prob − 1` (tratta win% come proxy; ignora massa draw implicita nella formula cp→win). Per allineamento stretto al blueprint Lc0 WDL servirebbe re-label o dataset con WDL nativo.

**Train full:** 15.3B action-values (~1.1 TB `.bag` shard) — non scaricato; test split sufficiente per prototipo pipeline.

### Candidati teacher — ❌ non installati

| Modello | Path previsto | Note |
|---------|---------------|------|
| Stockfish + WDL | `models/teacher/stockfish/stockfish.exe` | Fallback veloce; WDL post-hoc |
| Artoria mid/large | `models/teacher/hf/artoria-zero/{mid,large}/` | Solo small installato |

### SARDINE NNUE (target) — ❌ vuoto

| Path | Stato |
|------|-------|
| `models/` | Root artefatti SARDINE (`SARDINE_MODELS_DIR`) |
| `models/checkpoints/` | Previsto post-nnue-pytorch — **inesistente** |
| `models/exported/` | Previsto int8 + tanh LUT — **inesistente** |

Architettura target: `716 → W (128/256) → 2W → 1` × 8 bucket.

### Legacy pre-SARDINE — archivio (768-dim, centipawn)

**Checkpoints PyTorch** — `legacy/pre-sardine/models/checkpoints/`

| File | Size |
|------|------|
| `tiny_value_wio.pt` | 53 KB |
| `tiny_value_wio_nano_int8.pt` | 53 KB |
| `tiny_value_wio_tiny_int8.pt` | 104 KB |
| `tiny_value_wio_small_int8.pt` | 209 KB |
| `tiny_value_wio_medium_int8.pt` | 430 KB |
| `tiny_value_wio_big_int8.pt` | 857 KB |
| `tiny_value_wio_huge_int8.pt` | 1.7 MB |
| `tiny_value_wio_sparse80_int8.pt` | 53 KB |
| `tiny_chess_policy_lab.pt` | 16.8 MB |
| `tiny_chess_policy_lab_full.pt` | 16.8 MB |

**Export** — `legacy/pre-sardine/models/exported/`

| File | Size |
|------|------|
| `tiny_value_wio_*_int8.bin` | 12–427 KB |
| `tiny_chess_policy_lab.onnx` + `.onnx.data` | ~16.8 MB |
| `tiny_chess_policy_lab.ts.pt` | 16.8 MB |
| `my_tiny_model.ts.pt` | 60 B |

**Headers Wio (int8 embed)** — `legacy/pre-sardine/models/arduino/models/`

| File | Size |
|------|------|
| `tiny_value_wio_int8_model.h` | 79 KB |
| `tiny_value_wio_tiny_int8_model.h` | 156 KB |
| `tiny_chess_policy_model.h` | 100 MB |
| `tiny_chess_onnx_data.h` | 100 MB |
| `my_tiny_model.h` | 374 B |

Sketch: `legacy/pre-sardine/Arduino/Wio_TinyValueTest/`. **Non usare per SARDINE v1** (encoder 716, expected reward).

### Riepilogo

| Categoria | Path root | Installato | Uso SARDINE v1 |
|-----------|-----------|------------|----------------|
| Lc0 teacher (fast default) | `models/teacher/` | ✅ 3 reti + lc0 | Bot depth 1–2, demo GIF; **teacher training** |
| HF teacher (chess_lite) | `models/teacher/hf/` | ✅ 2 checkpoint | Smoke latency; gioco debole a depth-1 |
| ChessBench (SF 16) | `data/raw/chessbench/` | ✅ test split | Label win_prob — conversione per step C |
| Dataset pre-label (full) | esterno / `data/` | ⚠️ parziale | Train ChessBench 1.1 TB o Lc0 WDL dump |
| SARDINE NNUE | `models/` | ❌ | Training step C |
| Legacy value/policy | `legacy/pre-sardine/models/` | ✅ archivio | Riferimento storico |


---
# Dense/Convolutional Models, AlphaZero, Lc0

#### Marvin
- [Marvin](https://huggingface.co/holymolyyy/marvin) is a human-like chess neural network that models human play at a specified Elo rating and time control. 
- **Architecture:** _Not Found_ (Neural network trained to emulate human-like playstyles).
- **Input:** _Not Found_.
- **Output Description:** _Not Found_ (Likely move probabilities matching human play distributions at target ratings).
- **Model Sizes:** Available in three sizes: **large**, **small**, and **tiny**.
- **Parameters:** _Not Found_.
- **File Size:** ~19.3 MB (for the `tiny_2400.pb.gz` checkpoint variant).

#### chess-bot
- [chess-bot](https://huggingface.co/AubreeL/chess-bot) is a policy-value network inspired by AlphaZero, designed to evaluate chess positions and suggest moves. 
- **Input**: 18-plane board representation (12 pieces + 6 metadata planes) 
- **Convolutional backbone**: 32 filters, 1 residual block, ~9,611,202 parameters. 
- **Policy head**: 4,672-dimensional output (one per legal move encoding). 
- **Value head**: Single tanh output (-1 to +1 for position evaluation) 
- **File Size:** 38 MB

#### chessmate-net
- [chessmate-net](https://huggingface.co/victorqueiroz/chessmate-net)
- **Input:** single-position `8×8×22` planes (pieces ×12, castling, side-to-move, en-passant, halfmove, attack maps), side-to-move oriented (board rotated for Black). 
- **Outputs:** 
	- `policy` — 4096 logits over `from*64 + to` (the parity-locked move encoding; promotion folds onto the from→to index, no underpromotion dim); 
	- `value` — scalar `tanh` in `[-1, 1]`, side-to-move POV. 
- **File Size:** 120 MB

#### chess-bot / TinyPCN
- [chess-bot](https://huggingface.co/AubreeL/chess-bot)
- Architecture: Policy-value network ispirata ad AlphaZero.
- Policy + Value: Sì.
- Parameters: circa 9.6M.

#### chessmate-net
- [models](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77983406/9fe7729c-db8d-4f27-b7d0-e93109e825c0/Models.md?AWSAccessKeyId=ASIA2F3EMEYEQTA4RJEK&Signature=2NYsCVeWN80moMz39609lQmhoMc%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEOH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD%2BCt64%2FQVPJxq1kxzwRhDxJWWwlHRJrylXZ2%2BhidK7GgIgTakO%2BrPNrK15t%2B%2BCGW0xtPbx9Rk1O4QATZNjVUg4hFIq%2FAQIqf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDMG6sfk29Xa910jDiCrQBMZQMghdcxJr8GCeYtL72jjE0CiLYTzVkexstFxJrRabhuGlbVHldSBTK40Uej0Gw4T%2FXkTU0PfcZ089tJdx18RGp98DKNNGoQLJ%2F1HWDMztJCQd0syvwpC28l6b6%2BoAbqeHLqqhY3I4r2bS73QahDJ69chKWI1pFGeDnAsV23cE10LqHJcfvOzC9zjskfC7Wv7sIwJcEnUox6fSDP7nuVPc2RHPrxMWHpjGnLAFeVqOhfbAnDZCvxnoLGfu38u%2B2WMi2%2B6Kak6nZCVWpTTcO7Kzuvu0aBpN3SWD%2BXIvwxGFT4xU9bamIinHv2NC163kEzmFKKOAp7Sp5Bl%2FmA3JwRGgnFNojci6ZgiQ%2Bqx5XKU3emlWWZ5h2eSUsA5bGuhWt2IG3iTe49tXI0wYtPMi6dC7uR8ZMfRx%2F6mE4Yz9oxnrkInF6sdS98bJN%2BXawrXKV%2FJEwSOTPwaS9REiZzUjXDxZ3jcbvM7kKz7HlvJQW8uRJAO5wtQa4REgRR1v82MniFlacRa%2FtDZtq5kVQDjdgMS1pT%2FzkUIKPpyP98rJWKLs%2FPSZxgpYGdTtymG5UNDnQaXg8LYry9Xc%2Fm4ye3OEFadNhexfrya%2BpznZCJbSVeTT%2Fb5TGX6WfHWTI7ACrp1kRMTKTjnCP42ZOlyA4Y2DYjw7ZYFTpQL6Rs3UhZcHPmjALH8cB9qR6q9uoaVr2R%2FEc701fnZak7rwTgH2YmRmdU%2BFa0X7tvMiTpvNnG%2F9KMyKGKIbt5zzK6DGdoofheBnf%2FNrNIZxkNEVmVGoglG6eaEw2tSI0gY6mAHfZxswa6e5nbJUl%2Buo0JFJ%2B2mblcanASHLKSH1HNQHBRQDdF3AU40zpiBRh45QQWKv2F2OO6HwkzGcKGuBe2TsbfOcqOOnwPdc6y8ZrEfpAWeiN9aXSNj2d1Ilq1Ym1ODdIfcGdFOValbgohIMjmHmpi9%2FmwOV6YpqX0YFTO4OMrucX18Vh7qpcFsyCJSLyXRVz3YyeE2LBw%3D%3D&Expires=1782724653)
- Architecture: Dense / fully-connected-style, secondo la nota disponibile.
- Policy + Value: Sì.
- Parameters: non indicati.


- Small Lc0 (Leela Chess Zero) Networks on Edge Devices. While the main Leela Chess Zero (Lc0) project relies on massive GPU clusters, the community has developed "personality nets" and smaller network architectures that can run on devices like the Raspberry Pi.
    - **[The Ultimate Guide to Lc0's Human-Like Personality Nets (Google Groups)](https://groups.google.com/g/picochess/c/ap95BZ2JIPg)**: Discusses how smaller Lc0 architectures can run on a Raspberry Pi 3, hitting around ~2400 Lichess Elo with just 8 nodes per move.
    - **[Autonomous chess-playing robotic arm using Raspberry PI (Scribd)](https://www.scribd.com/document/901155249/Autonomous-chess-playing-robotic-arm-using-Raspberry-PI?spm=a2ty_o01.29997173.0.0.267155fbh2DRlp)**: Demonstrates a practical implementation of running a chess engine (Stockfish) on a Raspberry Pi 3B+ paired with a camera for vision processing.
    
- https://github.com/LeelaChessZero/lc0 Lc0 on GitHub 
    
- https://lczero.org/ The Leela Chess Zero community has trained thousands of networks, including highly stripped-down architectures meant for testing or ultra-low-power CPU inference. Because they use a standardized network structure, you can parse them directly using a simple custom script or find community-converted ONNX files

---

## Res-Net

#### chess-alphazero-openenv
- [chess-alphazero-openenv](https://huggingface.co/yogeshwaran13/chess-alphazero-openenv) 
- **Architecture:** ResNet backbone featuring 5 residual blocks and 64 channels.
- **Input** : 19 × 8 × 8 board tensor 
- **Output Description:** Dual-headed output providing a: 
	- Move Policy (4096 moves distribution) 
	- Position Value (scalar from -1 to +1)
- **File Size:** 18.4 MB

#### chess-alphazero-openenv
- [Model](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77983406/9fe7729c-db8d-4f27-b7d0-e93109e825c0/Models.md?AWSAccessKeyId=ASIA2F3EMEYEQTA4RJEK&Signature=2NYsCVeWN80moMz39609lQmhoMc%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEOH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD%2BCt64%2FQVPJxq1kxzwRhDxJWWwlHRJrylXZ2%2BhidK7GgIgTakO%2BrPNrK15t%2B%2BCGW0xtPbx9Rk1O4QATZNjVUg4hFIq%2FAQIqf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDMG6sfk29Xa910jDiCrQBMZQMghdcxJr8GCeYtL72jjE0CiLYTzVkexstFxJrRabhuGlbVHldSBTK40Uej0Gw4T%2FXkTU0PfcZ089tJdx18RGp98DKNNGoQLJ%2F1HWDMztJCQd0syvwpC28l6b6%2BoAbqeHLqqhY3I4r2bS73QahDJ69chKWI1pFGeDnAsV23cE10LqHJcfvOzC9zjskfC7Wv7sIwJcEnUox6fSDP7nuVPc2RHPrxMWHpjGnLAFeVqOhfbAnDZCvxnoLGfu38u%2B2WMi2%2B6Kak6nZCVWpTTcO7Kzuvu0aBpN3SWD%2BXIvwxGFT4xU9bamIinHv2NC163kEzmFKKOAp7Sp5Bl%2FmA3JwRGgnFNojci6ZgiQ%2Bqx5XKU3emlWWZ5h2eSUsA5bGuhWt2IG3iTe49tXI0wYtPMi6dC7uR8ZMfRx%2F6mE4Yz9oxnrkInF6sdS98bJN%2BXawrXKV%2FJEwSOTPwaS9REiZzUjXDxZ3jcbvM7kKz7HlvJQW8uRJAO5wtQa4REgRR1v82MniFlacRa%2FtDZtq5kVQDjdgMS1pT%2FzkUIKPpyP98rJWKLs%2FPSZxgpYGdTtymG5UNDnQaXg8LYry9Xc%2Fm4ye3OEFadNhexfrya%2BpznZCJbSVeTT%2Fb5TGX6WfHWTI7ACrp1kRMTKTjnCP42ZOlyA4Y2DYjw7ZYFTpQL6Rs3UhZcHPmjALH8cB9qR6q9uoaVr2R%2FEc701fnZak7rwTgH2YmRmdU%2BFa0X7tvMiTpvNnG%2F9KMyKGKIbt5zzK6DGdoofheBnf%2FNrNIZxkNEVmVGoglG6eaEw2tSI0gY6mAHfZxswa6e5nbJUl%2Buo0JFJ%2B2mblcanASHLKSH1HNQHBRQDdF3AU40zpiBRh45QQWKv2F2OO6HwkzGcKGuBe2TsbfOcqOOnwPdc6y8ZrEfpAWeiN9aXSNj2d1Ilq1Ym1ODdIfcGdFOValbgohIMjmHmpi9%2FmwOV6YpqX0YFTO4OMrucX18Vh7qpcFsyCJSLyXRVz3YyeE2LBw%3D%3D&Expires=1782724653)
- Architecture: ResNet.
- Policy + Value: Sì.
- Parameters: non indicati.

#### chess-alphazero-pytorch
- [chess-alphazero-pytorch](https://huggingface.co/santoshchandu/chess-alphazero-pytorch) full tree-search
- Residual CNN (6 blocks, 128 filters, 10M parameters)
- Policy head: move probability over 4096 possible moves
- Value head: win probability in [-1, +1]
- MCTS: 100 simulations per move
- Size: ~41MB

---

## Transformers

#### pawn-small
- [pawn-small](https://huggingface.co/thomas-schweich/pawn-small) **PAWN** (Playstyle-Agnostic World-model Network for Chess) 
- a causal transformer trained on random chess games. 
- It learns legal moves, board state representations, and game dynamics purely from uniformly random legal move sequences -- no strategic play, no hand-crafted features, no external game databases. 
- **Architecture**: Transformer
- **File Size:** ~8.9M

#### ChessFormer-SL
- [ChessFormer-SL](https://huggingface.co/kaupane/ChessFormer-SL)
- Architecture: Transformer.
- Policy + Value: Sì, policy su 1.969 mosse e value head.
- Parameters: 100.7M.


 **ChessBot**
- [ChessBot](https://huggingface.co/Maxlegrec/ChessBot)
- Platform: Hugging Face.
- Architecture: Transformer.
- Policy + Value: Sì, policy e value.
- Parameters: 34.7M.

---

## NNUE

Originally invented by Yu Nasu in 2018 for Shogi and ported to computer chess in 2020 via Stockfish, NNUE completely changed the paradigm of embedded board-game AI.

- [**NNUE**](https://beuke.org/nnue/#:~:text=Therefore%2C%20instead%20of%20recomputing%20the,move%20is%20made%20or%20unmade.) is the undisputed gold standard for high-performance chess on low-resource hardware. Originally developed for Shogi and now famously integrated into the Stockfish engine, NNUE is specifically designed to run efficiently on standard CPUs without requiring a GPU:
    - **[Introducing NNUE Evaluation - Stockfish Blog](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)**: The official announcement detailing how NNUE achieves low-latency evaluations on CPUs by only updating parts of the network incrementally
    - **[Efficiently Updatable Neural Network - Grokipedia](https://grokipedia.com/page/Efficiently_updatable_neural_network)**: Breaks down the architecture, noting the overparameterized input layer and the lightweight, updatable intermediate representation
    - **[Why Stockfish is So Good (Dev.to)](https://dev.to/djinn/why-stockfish-is-so-good-and-how-you-could-write-a-chess-engine-2lck)**: A technical deep dive into how NNUE uses int8/int16 quantization and SIMD to achieve massive speeds on consumer CPUs
    
    
- [beuke.org](https://beuke.org/nnue/#:~:text=Therefore%2C%20instead%20of%20recomputing%20the,move%20is%20made%20or%20unmade.) If you are looking for the absolute cutting edge of "small hardware" chess architectures, you must look at the FIDE & Google Efficient Chess AI Challenge hosted on Kaggle. This competition explicitly forbids brute-force computation and forces participants to build engines under extreme hardware constraints, such as allocating a maximum of just 5MiB of RAM.

---

### JEPA

[Yann LeCun, June 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)


---

## Other Pages
   
- The absolute cutting edge of "small hardware" chess architectures, the FIDE & Google Efficient Chess AI Challenge hosted on [Kaggle](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/overview). This competition explicitly forbids brute-force computation and forces participants to build engines under extreme hardware constraints, such as allocating a maximum of just 5MiB of RAM
	- **[FIDE and Google create the Efficient Chess AI Challenge (FIDE.com)](https://www.fide.com/fide-and-google-create-the-efficient-chess-ai-challenge-hosted-on-kaggle/)**: The official announcement challenging developers to create smart, resource-light chess programs
	- **[FIDE & Google Efficient Chess AI Challenge (Kaggle)](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/557921)**: The competition forums and discussion boards are a goldmine for seeing the exact lightweight architectures (like heavily pruned NNUE variants or tiny custom networks) that top competitors used to maximize Elo per byte
    
- https://huggingface.co/spaces/karthickajan/chess PHOTO -> FEN
    
- https://huggingface.co/atamano/whisper-chess-tiny reading move notation
    
- https://github.com/undera/chess-engine-nn chess-engine-nn
    

---
