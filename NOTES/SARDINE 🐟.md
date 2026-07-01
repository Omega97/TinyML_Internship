# SARDINE

```
Small Artificial RAM-restricted Deep Intelligent Neural Engine
```

![SARDINE logo](../images/SARDINE-logo-dark-small.png)

Chess engine for the **Wio Terminal** — neural evaluation + alpha-beta search, maximizing **Elo per byte** under 192 KB RAM / ~500 KB flash.

*Locked 2026-06-30. Alternative designs considered: [SARDINE design options.md](SARDINE%20design%20options.md).*

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
| **Nets** | **Separate nets** — NNUE for eval; distinct policy head deferred until after v1 Elo gate |

---
	
## Build Pipeline

```mermaid
flowchart TD
    A["Step 1 — Feature encoder<br/>PC + device parity"] --> B["Step 2 — Search skeleton in C++ on PC<br/>perft · eval hooks · TT format benchmark"]
    B --> C["Step 3 — Train bucketed NNUE<br/>Lc0 data · bucket-stratified sampling · games ≥ 16<br/>→ calibrated int8 export"]
    C --> D["Step 4 — Queen-split ablation<br/>per-bucket eval MSE vs piece-count baseline"]
    D --> E["Step 5 — Incremental accumulators<br/>on device"]
    E --> F["Step 6 — Port search + NNUE to C"]
    F --> G["Step 7 — Full search stack + tuning<br/>LMR · null-move · iterative deepening · SPSA"]
    G --> H{"Step 8 — Elo gate<br/>≥ 2000?"}
    H -- "No" --> I["Iterate:<br/>SCReLU · quantization-aware training ·<br/>TT format · bucket scheme"]
    I --> G
    H -- "Yes" --> J["Step 9 — v2 scope<br/>Policy guidance head · SCReLU if needed ·<br/>Grapheus/QAT if post-training quant gap too large"]

    style H fill:#f9d,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
```

---

## MoE + NNUE Architecture

```mermaid
flowchart TD
    FEN["FEN position"] --> FEAT["Pruned 704 features<br/>(zero impossible pawn ranks +<br/>mirrored king coordinates)"]

    FEAT --> ACC["Shared Accumulator<br/>768 → 16, dual-perspective<br/>int8 weights / int16 accumulators<br/>CReLU (SCReLU fallback) <br/> + Incremental Accumulator Update"]

    ACC --> BUCKET{"Bucket selector<br/>piece count + queen presence"}

    BUCKET -->|"p < 12"| E0["Expert 0<br/>endgame"]
    BUCKET -->|"..."| E1["..."]
    BUCKET -->|"p = 32"| E7["Expert 7<br/>early opening"]

    E0 & E1 & E7 --> OUT["Scalar eval score<br/>(16 → 1 per expert)"]

    OUT --> SEARCH["Alpha-beta + quiescence<br/>LMR / null-move / iterative deepening"]
    SEARCH <--> TT[("TT: 128–160 KB<br/>MVV-LVA move ordering")]
    SEARCH --> BEST["Best move"]
    BEST --> IO["TFT + Serial output"]

    style ACC fill:#bbf,stroke:#333,stroke-width:2px
    style BUCKET fill:#fdb,stroke:#333,stroke-width:2px
```

The accumulator (blue) is computed once per position and shared across all 8 experts — only the output head selected by the bucket router (orange) changes. Incremental add/sub updates on the accumulator are bucket-agnostic.

---

## Design Decisions

### Runtime (phased)

Pure **C** engine core (Cfish-style) is the target, but **port after** the first playable search exists in C++ on PC.

Rationale: debugging alpha-beta, quiescence, and TT interactions is far faster with a PC toolchain (debugger, sanitizers, perft/eval unit tests) than on Wio hardware. Minimal C++ remains acceptable for TFT/Serial glue on-device.

---

### Input features

**Pruned 704** features: zero impossible pawn ranks + mirrored king coordinates. HalfKP deferred.

---

### Evaluation

**Bucketed micro NNUE:** `768 → 16 → 1`, dual-perspective, **8 output weight sets** (experts) selected by position bucket.

**Shared accumulator:** all experts share the same first hidden layer. The accumulator depends only on board features, not on which output bucket is active — compute it once per position, then route to the correct output head. Matches Stockfish-style bucketed NNUE; incremental add/sub updates stay bucket-agnostic.

**Autoencoder warm-start:** skip for v1.

---

### Output buckets

Balanced training buckets with **queen-split** in middlegame/opening bands:

| Bucket | Condition | Phase |
|--------|-----------|-------|
| 0 | $p < 12$ | endgame |
| 1 | $p \in [13,21]$, no queen | late middlegame |
| 2 | $p \in [13,21]$, queen present | late middlegame |
| 3 | $p \in [22,27]$, no queen | middlegame |
| 4 | $p \in [22,27]$, queen present | middlegame |
| 5 | $p \in [28,31]$, no queen | opening |
| 6 | $p \in [28,31]$, queen present | opening |
| 7 | $p = 32$ | early opening |

Queen presence is high-leverage in buckets 1–4. Buckets 0 and 7 barely need the split.

**Ablation plan:** train queen-split vs pure piece-count buckets once pipeline exists. Compare **per-bucket eval MSE** on a Stockfish-labeled validation set (stratified like training) — not pooled MSE alone, which could hide bucket-level regressions. Escalate to playing-strength tests only if per-bucket results are ambiguous or contradictory; uniform improvement or regression across buckets is decisive enough to skip the expensive test.

**Future axes (v1.x/v2):** no king side, bishop/rook pair, or tactical flags in v1. `inCheck` is the first candidate (8 → 16 buckets) if an axis is added later.

Informed by `piece_count_distribution_10k.xlsx` (games ≥ 16 moves). Training uses bucket-stratified resampling.

---

### Policy (v1)

**Search-only** for v1. Add killer-move and countermove history heuristics once tables are in.

**Policy guidance net (v2):** defer until after v1 Elo gate. Lightweight head off the **shared accumulator** (16 → move-ranking); watch per-node latency against the ~1 s budget.

---

### Incremental updates

1. **Add/sub** accumulator updates on each move (shared layer — bucket-independent)
2. **Lazy updates** when TT cutoffs make eval skippable
3. Copy-make + fused add/sub optional later

---

### Geometric optimizations

- Horizontal king mirroring
- Hard-zero weights for impossible states (pawns on rank 1/8)
- Magnitude pruning (~80% sparsity post-training)

---

### Quantization

| Tensor | Precision |
|--------|-----------|
| Weights | **int8** |
| Biases | **int16** |
| Accumulators | **int16** |
| Activation | **CReLU** (v1) |

**Scale calibration:** train fp32 first, histogram post-training weights, set per-tensor scales onto $[-127, 127]$ with minimal clipping.

**SCReLU fallback** — first upgrade if CReLU plateaus below ≥2000 Elo:

1. **Clip** int16 accumulator to quantized activation range (e.g. $[0, 127]$) **before** squaring — load-bearing; unclipped $32767^2$ overflows even int32.
2. **Square** in **int16** (max $127^2 = 16{,}129$ fits).
3. **Multiply-accumulate** with int8 weights in **int32** (product up to ~2M; sum across terms needs int32 accumulation).

Mirrors Stockfish SCReLU practice: moderate width after square, promote before weighted sum.

**Grapheus / quantization-aware training:** skip for v1. Stay on **nnue-pytorch** + calibrated post-training quantization. Measure fp32→int8 accuracy gap after first training run; only investigate Grapheus or in-pipeline QAT if gap threatens the Elo gate.

---

### Search

Phased rollout:

1. Alpha-beta + **quiescence**
2. **Late-move reduction + null-move pruning**
3. **Iterative deepening** once TT is stable

---

### Memory

| Resource | Philosophy | Allocation |
|----------|------------|------------|
| **Flash** | Balanced | ~10% weights; rest search code + tables |
| **RAM** | TT-dominant | TT **128–160 KB**; accumulators + stack ~16 KB; scratch ~16 KB |

#### Transposition table

Design entry format first; slot count follows from 128–160 KB budget. Prototype on PC (build step 2), then benchmark on **Wio**.

**Candidate entry:** truncated zobrist, best move (~16 bit), score (16 bit), depth (8 bit), bound type (2 bit).

| Format | Size | Slots @ 128 KB | Trade-off |
|--------|------|----------------|-----------|
| Tight pack | ~10 B | ~13,100 | More entries; unaligned loads on SAMD51 cost extra shift/merge per probe |
| Byte-aligned | 16 B | ~8,200 | Fewer entries; faster probes |

**Decision metric:** wall-clock **nodes/sec** and **depth reached in ~1 s** on Wio at both formats — not hit-rate alone. Millions of probes per move compound per-probe CPU cost; the format with better hit-rate can still lose on net search depth.

---

### Move ordering

**MVV-LVA + captures** for v1; **killer moves** when search depth > 4.

---

### Training data

- Primary: **Lc0** high-quality games
- **Games ≥ 16 moves** (consistent with distribution analysis)
- **Bucket-stratified** resampling (queen-split rules)
- Stockfish self-play labels optional augment

---

### Training pipeline

- **nnue-pytorch** for v1 network training (not Grapheus)
- Calibrated post-training int8 export; measure fp32→int8 gap before considering QAT
- **SPSA** post-hoc for search/heuristic tuning

---

### I/O

**TFT + Serial**; hardcoded FEN input for now.

---

## Open Questions

### Validation & quantization (current pipeline)

- [ ] **Per-bucket ablation thresholds** — what MSE delta (per bucket or aggregate) counts as "decisive" vs "ambiguous" enough to warrant playing-strength tests?
- [ ] **TT format on Wio** — run the 10 B vs 16 B benchmark once the PC search skeleton can drive realistic node counts on hardware.
- [ ] **PTQ acceptance criterion** — maximum allowable fp32→int8 eval error (or Elo proxy) before escalating to SCReLU or quantization-aware training?

### Features & optimizations (from [Ideas 💡.md](Ideas%20💡.md) See Also)

Inspired by **Dog** (NNUE on ESP32, ~320 KB RAM, ~2000+ Elo on-device):

- [ ] **Dog RAM budget study** — how does Dog split flash/RAM between NNUE weights, TT, and book? Can we adopt a similar layout on Wio (~192 KB RAM, ~500 KB flash) without sacrificing our TT-dominant plan?
- [ ] **UCI over Serial** — expose SARDINE as a UCI engine on the Wio (like Dog on Lichess) for Elo testing against standard opponents, instead of hardcoded FEN only?
- [ ] **Opening book** — Dog ships a book alongside NNUE; is a minimal embedded book worth the flash cost for v1, or defer until post–Elo gate?

Inspired by **MicroChess** (<2 KB RAM, stack surfing):

- [ ] **Stack surfing for search depth** — instead of a fixed max ply, grow depth until free stack/RAM hits a safety threshold (dynamic depth within the ~1 s budget). Applicable on Wio with 192 KB RAM, or too risky next to TT + accumulators?
- [ ] **Bare-metal C patterns** — which MicroChess techniques (move generation, board representation, stack discipline) are worth porting into the C search core beyond what a standard alpha-beta skeleton provides?

Inspired by **ESP32 Chess Engine** (Urusov, ~20 kNps heuristics-only, ~2023 Elo):

- [ ] **Node budget model** — at measured Wio eval latency (ms/node) + move-gen overhead, how many nodes fit in ~1 s? Compare against Urusov's 20 kNps baseline to estimate reachable depth with our NNUE eval vs his pure HCE.
- [ ] **Futility pruning** — Urusov relies on it heavily; add to our search stack alongside LMR/NMP, or skip for v1 to keep the skeleton simple?
- [ ] **Lazy evaluation** — skip full NNUE eval when a cutoff is already provable from a previous ply's score; pairs naturally with lazy accumulator updates — worth implementing together?

From general Ideas not yet in the locked pipeline:

- [ ] **NNUE + pre-computed patterns** — combine the neural net with small handcrafted pattern tables (linrock-style geometric zeros go partway; is a separate pattern cache worth flash)?
- [ ] **Tactical MoE axis** — Ideas note that bucket switches are rare if piece count mostly decreases; should `inCheck` / capture-threat heads be revisited earlier than v2 given infrequent switching cost?
- [ ] **Compiler `-O3` on device** — Ideas flag speed optimizations; trade flash size vs nodes/sec on SAMD51 once the C port exists (FIDE 9th place used `-O3` + UPX after gutting features)?
- [ ] **Compact transformer as guidance net** — if the v2 policy head off the shared accumulator underperforms, is the ~210K transformer design ([chess transformer.md](chess%20transformer.md)) a fallback, or too heavy for per-node move ordering?

---

[← Design options](SARDINE%20design%20options.md) · [← Notes index](_notes.md)