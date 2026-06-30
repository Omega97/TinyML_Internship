# NNUE — Efficiently Updatable Neural Network Evaluation

> **NNUE** (ƎUИИ, *Efficiently Updatable Neural Network*) is a neural-network architecture designed to replace hand-crafted evaluation (HCE) in alpha-beta chess engines running on CPUs. It keeps search fast by updating only a small part of the network after each move, instead of re-running a full forward pass.

Primary reference: [NNUE — Chessprogramming wiki](https://www.chessprogramming.org/NNUE)

---

## History

| Year | Event |
|------|-------|
| 2018 | **Yu Nasu** introduces NNUE for Shogi, inspired by Kunihito Hoki's king-indexed piece-square tables in Bonanza |
| 2020 | **Nodchip** ports NNUE into Stockfish 10 as a proof of concept → **Stockfish NNUE** |
| 2020 | Stockfish 12 adopts NNUE as default eval; strength jumps despite ~2× slower nodes/sec |
| 2020+ | Widespread adoption (Igel, Halogen, Komodo Dragon, Fat Fritz 2, …) |

NNUE changed the paradigm for embedded chess AI: a compact neural net can outperform decades of hand-tuned heuristics, as long as inference stays cheap enough to search millions of nodes.

Further reading: [Introducing NNUE Evaluation — Stockfish Blog](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)

---

## Why "Efficiently Updatable"?

In a standard dense MLP, every move changes the input vector and forces a full recompute through all layers.

NNUE exploits **sparse binary inputs**: each active feature is an independent on/off neuron. The first hidden layer (the **accumulator**) is a linear sum of weight columns for active features. When a piece moves:

- Turn off 1–2 input features (old square / captured piece)
- Turn on 1–2 input features (new square)

So a quiet move updates the accumulator with **add/sub on two weight columns**, not a full matmul. Captures touch at most 3 features; castling, 4.

This incremental trick works **only on the first hidden layer**. Deeper layers (if any) must be recomputed from the refreshed accumulator — which is why the accumulator is kept wide (512–3072 neurons) and deeper stacks are used sparingly in fast time controls.

==Accumulators are not entirely re-computed at every move==, saving lots of calculations (lazy updates).

---

## Basic Architecture

The simplest NNUE is a **3-layer** network with **dual perspective**:

```
Input (sparse, 768 binary features)
    ↓  linear sum of active weight columns
Accumulator (hidden layer, e.g. 256–1024 int16)
    ×2 perspectives (White POV, Black POV) — board flipped for Black view
    ↓  concat → vector of length 2 × HL_SIZE
Activation (CReLU or SCReLU)
    ↓  dot product
Output (1 neuron → centipawn eval, side-to-move POV)
```

### Input layer — 768 features

One binary neuron per `(piece_type, square, color)`:

```
index = side × 384 + piece_type × 64 + square
      = side × 64 × 6 + piece_type × 64 + square
```

`piece_type`: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5  
`side`: White=0, Black=1  
`square`: 0–63 (A1=0 … H8=63)

Only ~32 features are active in a typical position (one per occupied square).

### Dual perspective

Two accumulators are maintained — one per side's point of view. For the Black perspective, the board is **flipped vertically** (square `^ 0b111000`, colors swapped). At inference, the **side-to-move** accumulator is concatenated first with the opponent's.

### Accumulator update (core idea)

```c
void accumulatorAdd(const Network* net, Accumulator* acc, size_t index) {
    for (int i = 0; i < HL_SIZE; i++)
        acc->values[i] += net->accumulator_weights[index][i];
}

void accumulatorSub(const Network* net, Accumulator* acc, size_t index) {
    for (int i = 0; i < HL_SIZE; i++)
        acc->values[i] -= net->accumulator_weights[index][i];
}
```

On `e2e4`: subtract weights for White pawn on e2, add weights for White pawn on e4 — for **both** perspective accumulators.

### Output layer

Accumulator values pass through an activation, then a dot product with output weights plus bias. Training uses sigmoid on the target; **inference skips sigmoid** and scales by `SCALE` (typically 400) for centipawn output.

| Activation | Formula | Notes |
|------------|---------|-------|
| ReLU | `max(x, 0)` | Rare; overflow risk on int16 accumulators |
| CReLU | `clamp(x, 0, QA)` | Common; easy to vectorize |
| SCReLU | `clamp(x, 0, 1)²` | Strongest eval quality; needs hand-written SIMD |

---

## Quantization

Training uses float32 weights; deployment uses **int16 accumulators** and **int8/int16 weight tables** for speed and deterministic incremental updates (float rounding errors would compound across plies).

Typical constants (2024 engines):

```c
#define SCALE 400   // centipawn scaling
#define QA  255     // accumulator quant
#define QB  64      // output-weight quant
```

Quantization flow:

1. Multiply float weights by `QA` (input→hidden) and `QB` (hidden→output), round to int
2. Accumulator biases scaled by `QA`; output bias by `QA × QB` (or `QA² × QB` for SCReLU)
3. At inference end: `eval = dot_product / (QA × QB) × SCALE`

This is the same int8 philosophy we use on the Wio pipeline — NNUE was designed for quantized CPU inference from the start.

---

## Advanced Techniques

Used in top engines and especially in the [FIDE & Google Efficient Chess AI Challenge](FIDE%20%26%20Google%20Efficient%20Chess%20AI%20Challenge.md) to shrink networks without losing Elo:

| Technique | Idea | Trade-off |
|-----------|------|-----------|
| **Output buckets** | Separate output weights per piece-count band | Better endgame vs middlegame specialization |
| **Horizontal mirroring** | King always on left half; flip board if king crosses centre | Halves king input space; king crossing = full refresh |
| **King input buckets** | Different accumulator weights per king region | Mixture-of-experts; boundary crossing = refresh |
| **Feature factorizer** | Shadow bucket during training, merged into buckets after | Faster training for rare king regions |
| **Pairwise multiplication** | Element-wise product of accumulator halves before output | Faster, smaller hidden; slightly weaker |
| **Multiple hidden layers** | e.g. Stockfish `L1 → L2 → L3` | Stronger eval; slower nodes/sec |
| **LayerStacks** | Switch post-L1 weights by material count | Generalization of output buckets |

### Performance tricks (engine integration)

- **Fused updates** — merge multiple add/sub into one loop (less overhead)
- **Copy-make** — store per-ply accumulators; update once on make, not on unmake
- **Lazy updates** — defer accumulator refresh until eval is actually needed (TT cuts skip eval)
- **Finny tables** — cache king-bucket accumulators to avoid expensive full refresh
- **SIMD** — vectorize accumulator forward (especially SCReLU dot products)

---

## NNUE vs Other Chess Networks

| | NNUE | AlphaZero-style CNN | Transformer (HF) |
|---|------|---------------------|------------------|
| **Role** | Evaluation in alpha-beta search | Policy + value for MCTS | Policy + value |
| **Input** | Sparse 768 binary features | Dense 8×8 planes | Token / plane sequence |
| **Update** | Incremental (add/sub) | Full forward pass | Full forward pass |
| **Typical size** | 50K–few M (quantized) | 10M–100M | 35M–100M |
| **MCU fit** | Excellent (designed for CPU) | Poor (needs GPU / big flash) | Poor |
| **Policy** | No native policy head | Yes (4096 logits) | Yes |

There is no standard "policy NNUE" — engines derive move ordering from search + move ordering heuristics, not a policy head. For TinyML policy output, AlphaZero or a custom small transformer is the usual path; for **position evaluation inside search**, NNUE is the gold standard.

---

## Relevance to This Project (Wio Terminal)

| Wio constraint | NNUE implication |
|----------------|------------------|
| 192 KB RAM | No large TT; accumulator + small net only |
| ~500 KB flash | Micro-NNUE fits (FIDE 1st place: 64 KiB total binary; nagiss: `768→16→1`) |
| SAMD51 120 MHz, FPU | int8 weights + int16 accumulators match our export path; SIMD limited vs x86 |
| No search yet | NNUE shines with alpha-beta; standalone eval is still useful as value net |

**Actionable ideas from NNUE literature:**

1. **Sparse binary input** — our `768`-dim piece-square featurizer is already NNUE-compatible
2. **Geometric pruning** — zero weights for impossible states (pawns on rank 1/8, mirrored kings) to shrink compressed size
3. **Incremental updates** — worth implementing only once we add search; for single-position inference, full forward is fine
4. **Small hidden layer** — `768 → 16 → 1` or `768 → 64 → 1` as stepping stone before wider nets
5. **Dual-head extension** — our value MLP family could gain a policy head separately; true NNUE remains eval-only

See also: [PROJECT.md](../PROJECT.md) (Small NNUE target), [Models.md](Models.md) (NNUE section), [chess.md](chess.md) (FEN / promotion).

---

## Tools & Source Code

- [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) — official Stockfish NNUE trainer
- [Stockfish NNUE docs](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
- [Sunfish NNUE](https://github.com/kennyfrc/sunfishNNUE) — minimal Python reference
- [minifish](https://github.com/linrock/minifish) — 1st place FIDE efficiency challenge (micro-NNUE)

---

[← Back to Notes index](_notes.md)