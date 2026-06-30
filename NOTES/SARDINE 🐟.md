# SARDINE

```
Small Artificial RAM-restricted Deep Intelligent Neural Engine
```

![SARDINE logo](../images/SARDINE-logo-dark-small.png)

Chess engine for the **Wio Terminal** — neural evaluation + alpha-beta search, maximizing **Elo per byte** under 192 KB RAM / ~500 KB flash.

*Locked 2026-06-30. Full option catalogue: [SARDINE design options.md](SARDINE%20design%20options.md).*

---

## Mission

Playable chess bot on-device: no cloud, no GPU. v1 uses search + NNUE eval; separate policy net deferred.

**Non-goals (v1):** human-like play, Stockfish parity, photo/voice input.

---

## Targets

| Parameter | Decision |
|-----------|----------|
| **Elo** | ≥ **2000** (reference: FIDE Kaggle bots ~2500 Elo at 5 MiB RAM) |
| **Move time** | Best move within **~1 s** |
| **MCTS** | Feasible on-device (1–50 ms/eval); **v2 only** — v1 uses alpha-beta |
| **Nets** | **Separate nets** — NNUE for eval; distinct net for search guidance later |

---

## Architecture

```
FEN → pruned 704 features → NNUE accumulators → bucketed eval (8 piece-count heads)
                                    ↑
              alpha-beta + quiescence → LMR/NMP → iterative deepening
                                    ↑
                         TT (128–160 KB RAM) + MVV-LVA move order
                                    ↓
                         best move → TFT + Serial
```

---

## Design Decisions

### Runtime — R-A

Pure **C** engine core (Cfish-style). Minimal C++ only for TFT/Serial if the Arduino toolchain requires it.

---

### Input features — F-B

**Pruned 704** features: zero impossible pawn ranks + mirrored king coordinates. HalfKP deferred.

---

### Evaluation — E-E (E-C)

**Bucketed micro NNUE:** `768 → 16 → 1`, dual-perspective, **8 output weight sets** selected by piece-count bucket.

---

### Output buckets — B-C

Balanced training buckets (equalized position mass per bucket):

| Bucket | Piece count |
| ------ | ----------- |
| 0      | <12         |
| 1      | 13–17       |
| 2      | 18–21       |
| 3      | 22–24       |
| 4      | 25–27       |
| 5      | 28–29       |
| 6      | 30–31       |
| 7      | 32          |

Informed by `piece_count_distribution_10k.xlsx` (~48% of positions have 28–32 pieces).

> Alternatively, we can try to find a completely different way to bucket positions. For example, the MoE approach could benefit from dividing the positions in buckets by leveraging the metrics:
- opening (28+ pieces)
- middlegame
- endgame (<=12 pieces, including kings)
- bishop pair
- rook pair
- queens
- ...
> How can we optimize for the best Experts?

---

### Policy — P-A (v1)

**Search-only** for v1. Upgrade to history heuristics (**P-B**) once tables are in.

---

### Incremental updates — U-B → U-C

1. **Add/sub** accumulator updates on each move
2. **Lazy updates** when TT cutoffs make eval skippable
3. Copy-make + fused add/sub optional later

---

### Geometric optimizations — K-A, K-B, K-C

- Horizontal king mirroring
- Hard-zero weights for impossible states (pawns on rank 1/8)
- Magnitude pruning (~80% sparsity post-training)

---

### Quantization — Q-A

**int8** weights, **int16** accumulators, **CReLU** activation.

---

### Search — S-B → S-C → S-D

1. Alpha-beta + **quiescence**
2. **LMR + null-move pruning**
3. **Iterative deepening** once TT is stable

---

### Memory

| Resource | Split | Allocation |
|----------|-------|------------|
| **Flash (M-F2)** | Balanced | ~10% weights; rest search code + tables |
| **RAM (M-R1)** | TT-dominant | TT **128–160 KB**; accumulators + stack ~16 KB; scratch ~16 KB |

---

### Move ordering — O-A → O-B

**MVV-LVA + captures** for v1; **killer moves** when search depth > 4.

---

### Training data — D-B + D-D

- Primary: **Lc0** high-quality games
- **Bucket-stratified** resampling for B-C buckets
- Stockfish self-play labels optional augment

---

### Training pipeline — T-B + T-D

- **nnue-pytorch** for network training
- **SPSA** post-hoc for search/heuristic tuning

---

### I/O — I-B

**TFT + Serial**; hardcoded FEN input for now.

---

## Build Order

1. Feature encoder F-B — PC + device parity
2. Train bucketed micro NNUE → int8 export
3. Incremental accumulators on device
4. Search skeleton (quiescence) on PC → port to C
5. TT + LMR/NMP + iterative deepening; SPSA tuning
6. Elo gate vs ≥ 2000 baseline

---

## Open Questions

- [ ] Port to C (**R-A**) before or after first playable search in C++?

---

[← Design options](SARDINE%20design%20options.md) · [← Notes index](_notes.md)