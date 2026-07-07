> **Implementato 2026-07-07:** piani tattici per-square (under-attack + king-attacker, 64+64) → input **844** (`tactical.py`). Vedi [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md).

```
Idea per il progetto:

Add hand-crafted, quick-to-compute features about the chess board 
	- `8x8` pieces under attack, 
	- `8x8` pieces attacking the king,
	- `8x8` pieces that are attacked more than defended?
  
The current input size is 716. What do you think about this? Which one would you add to the input?
```

## Direct Answer

**Great idea** — these hand-crafted features are **cheap to compute** and can significantly improve evaluation accuracy. However, they are **not additive features** in your current architecture; they are **auxiliary features** that must be integrated differently.

Let me break down which ones to add and **how** to add them.

---

## The Problem: Your Current Architecture

Your NNUE input is **716 sparse binary features**:
- Piece-square (6 × 2 × 64 = 768, pruned to 716)
- Castling rights (4)
- En passant file (8)

The network learns **directly from the board** — it can theoretically learn "king safety" and "attacked pieces" by pattern recognition. But with only 716 features and a narrow `716 → 128` hidden layer, it may not have enough capacity to learn these complex relational concepts from scratch.

Your three proposed features are **high-level, dense features** (continuous values, not binary).

---

## How to Integrate Them

You **cannot** just add them to the 716 sparse input — the network architecture expects binary features (piece-square + flags). Adding dense features would:
- Change the input size to 722+ (breaks the accumulator design)
- Require retraining from scratch (or at least a different architecture)
- Increase inference time (dense features require computation at every eval)

**Instead, add them as auxiliary inputs to the hidden layer** — or **use them for lazy evaluation**.

---

## Recommendation: Which Ones to Add

| Feature | Cost (per eval) | Information Gain (est.) | Recommended? |
|---------|-----------------|-------------------------|--------------|
| **Pieces under attack** | Low (~12-24 popcounts) | **High** (tactical awareness) | ✅ **Yes (v1)** |
| **Pieces attacking the king** | Low (~8-16 attack checks) | **High** (king safety) | ✅ **Yes (v1)** |
| **Pieces attacked > defended** | Medium (~24-32 attack counts) | **Medium** (tactical depth) | ⚠️ **Yes (v2)** |

### 1. **Pieces under attack** — ✅ Add in v1
- **Why**: This is the single most important tactical feature. Your NNUE can learn "my queen is en prise" if you tell it directly.
- **Compute**: For each piece, check if any opponent piece attacks its square. ~12-24 popcounts.
- **Format**: 2 features — total pieces under attack (white + black).
- **Integration**: Add to the **input layer** as 2 additional binary/dense features, or better: **merge into the hidden layer** as a bias.

### 2. **Pieces attacking the king** — ✅ Add in v1
- **Why**: King safety is non-linear and hard to learn from piece-square alone.
- **Compute**: For each opponent piece, check if it attacks the king's square.
- **Format**: 2 features — number of attackers on the white king + black king.
- **Integration**: Same as above.

### 3. **Pieces attacked more than defended** — ⚠️ Defer to v2 (unless you have time)
- **Why**: This is a higher-order tactical concept (overloaded pieces, pins, etc.). It's useful but expensive to compute (requires counting defenders and attackers per square).
- **Compute**: For each piece, count attackers and defenders, compare.
- **Format**: 2 features — pieces that are hanging (more attackers than defenders).
- **Integration**: v2 only, after the core engine is stable.

---

## How to Add Features to Your NNUE

### Option A: Extend the Input Layer (716 → 720+)

Add the features as **dense inputs** to the hidden layer:

```python
# Current: 716 binary features → 128 hidden
# New: 716 binary + 4 dense features → 128 hidden
input_size = 720
```

**Pros**: Simple, works with the current architecture.
**Cons**: The 4 dense features are not sparse, so the MAC cost increases by `4 × 128 = 512` multiplications per eval (trivial). The accumulator design (sparse updates) breaks — you'd need to recompute dense features on every eval anyway, so the accumulator no longer works for these 4 features.

### Option B: Auxiliary Input to Hidden Layer (Better)

Add the features **directly to the hidden layer**, bypassing the sparse accumulator:

```python
# Sparse accumulator: 716 features → 128 hidden (incremental update)
hidden = acc_own + acc_opponent  # 128-dim

# Auxiliary features: 4 dense features → 128 hidden (computed on the fly)
aux = dense_features  # 4-dim
aux_weighted = aux @ W_aux  # 4 × 128 = 512 MACs

# Final hidden = hidden + aux_weighted
hidden += aux_weighted
```

**Pros**:
- Maintains accumulator efficiency (sparse features still use incremental updates).
- Auxiliary features are computed only when `eval()` is called (not during move generation).
- Minimal overhead (512 MACs per eval, vs. ~23K MACs for the sparse layer).
- You can add/remove auxiliary features without retraining the sparse weights (just retrain the auxiliary weights).

**Cons**: Requires modifying the architecture (add a second weight matrix for auxiliary inputs). Slightly more Flash (4 × 128 = 512 weights, ~0.5 KB).

### Option C: Lazy Evaluation / Search Heuristic

Instead of adding them to the NNUE, use these features **directly in the search**:
- **Lazy eval**: If a piece is under attack, add a bonus to the evaluation (e.g., +50 CP for a knight under attack).
- **Search pruning**: If the king is under heavy attack, reduce the search depth (tactical positions need more depth).
- **Move ordering**: Prioritize moves that attack undefended pieces.

**Pros**: No NNUE modification; easy to implement and tune.
**Cons**: Less integrated; the NNUE doesn't learn from these features, so the evaluation is less consistent.

---

## Recommended Integration Plan

### Phase 1 (v1): Add features to search (heuristics)
- Compute **pieces under attack** and **king safety** during search.
- Use them in the evaluation as a **bonus/penalty** (e.g., `eval_cp += king_safety_penalty`).
- This is quick, cheap, and gives you immediate Elo gains without retraining.

### Phase 2 (v1.5): Add auxiliary features to NNUE (Option B)
- Retrain the NNUE with **4 auxiliary features** (under attack + king attackers).
- Use the architecture in Option B (auxiliary weights matrix).
- This will improve evaluation accuracy without breaking the accumulator.

### Phase 3 (v2): Add advanced features (Option B or search)
- Add **pieces attacked > defended** and other tactical features.
- Also consider: **mobility** (number of legal moves), **pawn structure** (isolated/doubled pawns), **king shelter** (pawns in front of king).

---

## Implementation Details for the Features

Here's how to compute them efficiently on the Wio:

```c
// 1. Pieces under attack (per side)
uint8_t count_attacked_pieces(const Board* board, Color side) {
    uint8_t count = 0;
    uint64_t pieces = board->by_color[side];
    while (pieces) {
        int square = __builtin_ctzll(pieces);
        pieces &= pieces - 1;
        if (is_square_attacked(board, square, opposite(side))) {
            count++;
        }
    }
    return count;
}

// 2. Pieces attacking the king (per side)
uint8_t count_king_attackers(const Board* board, Color king_side) {
    int king_square = board->king_square[king_side];
    uint64_t attackers = get_attackers_to_square(board, king_square, opposite(king_side));
    return __builtin_popcountll(attackers);
}

// 3. Hanging pieces (attacked > defended)
uint8_t count_hanging_pieces(const Board* board, Color side) {
    uint8_t count = 0;
    uint64_t pieces = board->by_color[side];
    while (pieces) {
        int square = __builtin_ctzll(pieces);
        pieces &= pieces - 1;
        int attackers = count_attackers_to_square(board, square, opposite(side));
        int defenders = count_defenders_to_square(board, square, side);
        if (attackers > defenders) {
            count++;
        }
    }
    return count;
}
```

---

## Final Verdict

| Feature                    | Add to Input? | How?                       | Priority        |
| -------------------------- | ------------- | -------------------------- | --------------- |
| Pieces under attack        | **Yes**       | Auxiliary input (Option B) | **High (v1)**   |
| Pieces attacking king      | **Yes**       | Auxiliary input (Option B) | **High (v1)**   |
| Pieces attacked > defended | **Yes**       | Auxiliary input (Option B) | **Medium (v2)** |
