
# Project Blueprint

## **Scope & Milestones**

1. Get a minimal NNUE-like policy network running inference on XIAO (no search). Measure latency/power on random positions.
2. Add shallow search 
3. distillation/pruning + Elo testing (power optimization)
4. Full system (input/output).

---

## **Data & Training**

- Large chess datasets for distillation.
  -  [Lichess database](https://database.lichess.org/)
    
- Small chess datasets for testing.
  - [small Kaggle database](https://www.kaggle.com/datasets/datasnaek/chess)

---

## **Models**

### Value functions

#### Minimal Model
- **Architecture**: `768 → 32 → 32 → 1` or even smaller like `768 → 16 → 16 → 1` or `768 → 8 → 8 → 1` for ultra-tiny baselines.
- **Input**: Simple piece-square tables or sparse 768-bit "A-style" features; HalfKP possible but overkill here. 
- **Output**: Position evaluation in centipawns (float scalar).
- **Size**: ~25K params for `768 → 32 → 32 → 1`; <10K params only with narrower layers (e.g. ~8K for `768 → 8 → 8 → 1`). 
- **Why**: Fits on MCUs like XIAO/ESP32, fast 4-bit quantized inference, ideal first prototype or distillation baseline. 
- **Use case**: Baseline eval for shallow search; beats random play minimally.
- **Links**: [NNUE docs](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), [Chessprogramming NNUE](https://www.chessprogramming.org/NNUE), [MCU-Max](https://chessengines.blogspot.com/2024/02/chess-engine-mcu-max-10.html). [chessengines.blogspot](https://chessengines.blogspot.com/2024/02/chess-engine-mcu-max-10.html)

#### Small NNUE
- **Architecture**: `768 → 64 → 32 → 1` or `HalfKP → 64 → 32 → 1` (classic compact style). 
- **Input**: HalfKP (king + piece relative positions) — sparse, efficient incremental updates. 
- **Output**: Position evaluation in centipawns (float scalar).
- **Size**: ~50K dense params; effective size small due to sparse input (~0.1% active features).
- **Why**: Sweet spot for TinyML/MCUs; strong vs random, runs with shallow search on low-power hardware. 
- **Use case**: Practical embedded chess engine prototype.
- **Links**: [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch), [Sunfish NNUE](https://github.com/kennyfrc/sunfishNNUE), [Casanchess](https://github.com/casanche/casanchess). [github](https://github.com/kennyfrc/sunfishNNUE)

#### Good / Medium NNUE (Target for good performance)
- **Architecture**: `768 → 256 → 32 → 1`, `1024 → 128 → 32 → 1`, or Stockfish-like `1024x2 → 8 → 32 → 1`. 
- **Input**: HalfKP or improved bucketed features for better accuracy. 
- **Output**: Position evaluation in centipawns (float scalar).
- **Size**: 100K–few million params; still CPU/MCU viable when quantized.
- **Why**: Matches strong open-source tiny engines; 1600–2000+ Elo with light search. 
- **Use case**: Competitive embedded engine with decent strength.
- **Links**: [Bullet trainer](https://github.com/jw1912/bullet), [Stockfish NNUE](https://www.chessprogramming.org/Stockfish_NNUE), [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch). [github](https://github.com/official-stockfish/nnue-pytorch)

_Note: there is no direct "policy NNUE" equivalent in chess engines for tiny/MCU use._

---

### Policy Functions

#### Policy-chess
- **Architecture**: Conv net (8x8x8 board input → policy softmax over legal moves).
- **Input**: Board state as 8x8x8 channels.
- **Output**: Probability distribution over ~4k possible moves (softmax).
- **Size**: ~100K–500K parameters (simple conv layers).
- **Why**: Smallest full-chess policy net; TF Lite convertible for edge chips.
- **Edge fit**: Good for ESP32/RPi; quantize to 4/8-bit.
- **Links**: [GitHub](https://github.com/Zeta36/Policy-chess).

#### Reddit Small NN
- **Architecture**: Direct move inference policy-style net.
- **Input**: Board state → move distribution.
- **Output**: Policy over legal moves.
- **Size**: 15M params (quantizable to ~4M at 4-bit).
- **Why**: Plays full chess, 2ms/move on CPU.
- **Edge fit**: Moderate; export to ONNX/TFLite needed.
- **Links**: [Reddit](https://www.reddit.com/r/chess/comments/1rv86hw/i_trained_a_small_neural_network_to_play_chess_on/).

#### Minichess DRL Agent
- **Architecture**: Shared value/policy net for 5x5 minichess.
- **Input**: 5x5 board → policy distro.
- **Output**: Move probabilities.
- **Size**: <100K parameters.
- **Why**: MCU-ready but mini variant only.
- **Edge fit**: Excellent for tiny chips.
- **Links**: [arXiv](https://arxiv.org/pdf/2112.13666.pdf).

#### Tiny DRL Chess (Thesis)
- **Architecture**: Tiny NN with policy head for constrained agents.
- **Input**: Board state → policy distribution.
- **Output**: Move probabilities.
- **Size**: <50K parameters.
- **Why**: Designed specifically for edge computing.
- **Edge fit**: Perfect for MCU deployment.
- **Links**: [TU Wien PDF](https://repositum.tuwien.at/bitstream/20.500.12708/227472/1/Tayari%20Hakim%20-%202026%20-%20Tiny%20Deep%20Reinforcement%20Learni...).

---
