
![SARDINE logo](../images/SARDINE-logo-dark-small.png)

Chess engine for the **Wio Terminal** — neural evaluation + alpha-beta search, maximizing **Elo per byte** under 192 KB RAM / ~500 KB flash.

*Alternative designs considered: [SARDINE design options.md](SARDINE%20design%20options.md).*

---

## Mission

> Playable chess bot on a tiny device: no cloud, no GPU. Extreme optimization and efficiency. 

---

## Targets

| Parameter     | Decision                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------- |
| **Elo**       | **~ 1700** (gate minimo; se superiamo, meglio)                                           |
| **Move time** | Best move within **~1 s**                                                                |
| **MCTS**      | Feasible on-device (1–50 ms/eval); **v2 only** — v1 uses alpha-beta                      |
| **Nets**      | **Separate nets** — NNUE for eval; distinct policy head deferred until after v1 Elo gate |

**Node budget reference:** Urusov's ESP32 engine (~20 kNps, heuristics-only, ~2023 Elo) sets a baseline for search throughput without NNUE. SARDINE's reachable depth in ~1 s depends on measured eval latency + move-gen overhead on Wio — model empirically once the search skeleton exists.

---

## Repo status (2026-07-02)

Riferimento onesto rispetto al blueprint (vedi anche [ai-feed.md](../ai-feed.md)):

| Area | Stato |
|------|--------|
| **Feature encoder (step 1)** | In corso — `src/tinymlinternship/features/`: 716 sparse, dual-perspective, bucket router; test su PC |
| **Search + TT** | Non iniziato — prossimo: skeleton C++ su PC (`engine/`) |
| **NNUE training** | Non iniziato — serve subset Lc0 + nnue-pytorch |
| **Wio runtime** | Solo legacy value-MLP in `legacy/pre-sardine/`; nuovo engine non ancora portato |
| **Dati** | Kaggle `games.csv` (~100 MB) per test; **Lc0 non ancora nel repo** |

Il percorso attivo è `src/` + `scripts/download_data.py` / `plot_piece_count_distribution.py`. Tutto il vecchio export MLP → `legacy/pre-sardine/`.

---

## Build Pipeline

```mermaid
flowchart TD
    A["Feature encoder<br/>PC + device parity"] --> B["Search skeleton in C++ on PC<br/>perft · eval hooks · TT format benchmark"]
    B --> C["Train bucketed NNUE<br/>Lc0 data · bucket-stratified sampling · games ≥ 16<br/>→ calibrated int8 export"]
    C --> D["Queen-split ablation<br/>per-bucket eval MSE vs piece-count baseline"]
    D --> E["Incremental accumulators<br/>on device"]
    E --> F["Port search + NNUE to C<br/>benchmark -O3 vs -Os"]
    F --> G["Full search stack + tuning<br/>quiescence · futility · LMR · null-move ·<br/>lazy eval · iterative deepening · SPSA"]
    G --> H{"Elo gate<br/>≥ 1700?"}
    H -- "No" --> I["Iterate:<br/>SCReLU · quantization-aware training ·<br/>TT format · bucket scheme"]
    I --> G
    H -- "Yes" --> J["v2 scope<br/>minimal UCI polish · policy head ·<br/>SCReLU / QAT / transformer only if needed"]

    style H fill:#f9d,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
```

---

## MoE + NNUE Architecture


```mermaid
flowchart TD
    subgraph A_row["Feature encoding"]
        A1["Own-side features<br/>716-dim, own king mirrored"]
        A2["Opponent-side features<br/>716-dim, board mirrored for opp"]
    end

    B["Shared sparse FFNN <br/>(white POV)<br/>716 → 16<br/>same weights, called twice"]

    subgraph C_row["Accumulators"]
        C1["My accumulator<br/>16 values, own POV"]
        C2["Opponent accumulator<br/>16 values, their POV"]
    end

    D["Concatenate<br/>own ‖ opponent → 32-dim"]
    E{"Bucket Router<br/>routes to 1 of 8 experts"}
    F["Selected NNUE expert head<br/>32 → 1"]
    G["Eval score in [-1,+1]"]

    A1 --> B
    A2 --> B
    B --> C1
    B --> C2
    C1 --> D
    C2 --> D
    D --> E
    E --> F
    F --> G

    style A1 fill:#e6f1fb,stroke:#185fa5,stroke-width:1px
    style C1 fill:#e6f1fb,stroke:#185fa5,stroke-width:1px
    style A2 fill:#e1f5ee,stroke:#0f6e56,stroke-width:1px
    style C2 fill:#e1f5ee,stroke:#0f6e56,stroke-width:1px
    style B fill:#eeedfe,stroke:#534ab7,stroke-width:1px
    style D fill:#eeedfe,stroke:#534ab7,stroke-width:1px
    style E fill:#eeedfe,stroke:#534ab7,stroke-width:1px
    style F fill:#eeedfe,stroke:#534ab7,stroke-width:1px
    style G fill:#f1efe8,stroke:#5f5e5a,stroke-width:1px
```


The accumulator (green) is computed once per position and shared across all 8 experts — only the output head selected by the bucket router (orange) changes. Incremental add/sub updates on the accumulator are bucket-agnostic.

---

## Design Decisions

### Runtime (phased)

Pure **C** engine core (Cfish-style) is the target, but **port after** the first playable search exists in C++ on PC.

Rationale: debugging alpha-beta, quiescence, and TT interactions is far faster with a PC toolchain (debugger, sanitizers, perft/eval unit tests) than on Wio hardware. Minimal C++ remains acceptable for TFT/Serial glue on-device.

**Compiler flags:** once the C port lands (build step 6), benchmark `-O3` vs `-Os` on Wio — a free recompile experiment; no decision needed upfront. FIDE 9th place gained significant speed from `-O3` after gutting unused features.

**MicroChess bare-metal patterns:** skip — not worth diverging from a standard alpha-beta skeleton for v1.

---

### Input features

**Pruned 716** features: zero impossible pawn ranks + mirrored king coordinates. HalfKP deferred.

Input structure:

$$6 \times 2 \times 64 - 2\times16 - 32 + 4 + 8 = 716$$

- $768$ raw piece-square (6 types × 2 colors × 64 squares)
- $-32$ pawn ranks 1/8 omitted from index map
- $-32$ perspective-king plane compressed 64→32 (enemy king keeps full 64-square resolution)
- $+4$ castling rights · $+8$ en passant file

Implementazione: `index_map.py`, `encoder.py`, `mirror.py`, `bucket.py` — test: `tests/test_features.py`.

**Separate pattern tables:** skip for v1. Geometric zeros (impossible pawn ranks, king mirroring, magnitude pruning) already capture the cheapest compression wins; a handcrafted pattern cache adds flash complexity for uncertain gain until the Elo gap is measured.

---

### NNUE Architecture

```mermaid
flowchart TD
    subgraph INPUT["📥 INPUT LAYER"]
        X["716 sparse features<br/>(piece-square + castling + en passant)"]
    end

    subgraph ACC["⚡ SHARED ACCUMULATOR L1 (same weights, called twice)"]
        W["Weights: 716 × 16<br/>dtype: int8<br/>update: lazy add/sub"]
        A1["Own-POV accumulator<br/>16 × int16"]
        A2["Opponent-POV accumulator<br/>16 × int16"]
        C1["CReLU: clamp 0..127<br/>own POV, int8[16]"]
        C2["CReLU: clamp 0..127<br/>opp POV, int8[16]"]
        W --> A1 --> C1
        W --> A2 --> C2
    end

    subgraph CONCAT["🔗 DUAL PERSPECTIVE"]
        D["Concat: 16 + 16 = 32<br/>(own POV + opponent POV,<br/>by side to move)"]
    end

    subgraph EXPERT["🧠 EXPERT HEAD (selected bucket)"]
        E["Weights: 32 × 1<br/>dtype: int8"]
        MAC["Multiply-accumulate<br/>dtype: int32"]
        E --> MAC
    end

    subgraph OUT["📊 EVALUATION"]
        Y["Scalar eval score<br/>(side to move perspective)"]
    end

    X -->|"own-side features"| W
    X -->|"opponent-side features"| W
    C1 --> D
    C2 --> D
    D --> EXPERT
    MAC --> Y

    style INPUT fill:#e8f4f8,stroke:#2196F3,stroke-width:2px
    style ACC fill:#e8f5e9,stroke:#4CAF50,stroke-width:2px
    style CONCAT fill:#fff3e0,stroke:#FF9800,stroke-width:2px
    style EXPERT fill:#ede7f6,stroke:#673AB7,stroke-width:2px
    style OUT fill:#e0f2f1,stroke:#009688,stroke-width:2px
```

---

### Evaluation

**Bucketed micro NNUE:** `716 → 16 → 1`, dual-perspective, **8 output weight sets** (experts) selected by position bucket.

**Activations:** **CReLU** on the shared hidden layer (716 → 16); **tanh** on the final scalar output (16 → 1 per expert). The tanh is never computed at runtime — apply a **precomputed lookup table** indexed by the clipped int16 dot-product, mapping to a fixed-range centipawn score. LUT lives in flash (~1–2 KB for 256–512 entries); training in `nnue-pytorch` uses tanh so export matches inference.

**Shared accumulator:** all experts share the same first hidden layer. The accumulator depends only on board features, not on which output bucket is active — compute it once per position, then route to the correct output head. Matches Stockfish-style bucketed NNUE; incremental add/sub updates stay bucket-agnostic.

**Autoencoder warm-start:** skip for v1.

**Tactical MoE (`inCheck`, capture threat):** defer to v1.x/v2. Bucket switches are already infrequent along a typical game (piece count mostly decreases), so the current 8-bucket scheme is not leaving large gains on the table — no urgency to add tactical heads earlier.

---

### Output buckets

Balanced training buckets (based on number of pieces $p$, kings included) with **queen-split** (for now) in middlegame/opening bands:

| Bucket | Condition                      | Phase           |
| ------ | ------------------------------ | --------------- |
| 0      | $p \le 12$                     | endgame         |
| 1      | $p \in [13,21]$, no queen      | late middlegame |
| 2      | $p \in [13,21]$, queen present | late middlegame |
| 3      | $p \in [22,27]$, no queen      | middlegame      |
| 4      | $p \in [22,27]$, queen present | middlegame      |
| 5      | $p \in [28,31]$, no queen      | opening         |
| 6      | $p \in [28,31]$, queen present | opening         |
| 7      | $p = 32$                       | early opening   |

Queen presence is high-leverage in buckets 1–4. Buckets 0 and 7 barely need the split.

**Ablation plan:** train queen-split vs pure piece-count buckets once pipeline exists. Compare **per-bucket eval MSE** on a Stockfish-labeled validation set (stratified like training) — not pooled MSE alone. Escalate to playing-strength tests only if per-bucket results are ambiguous or contradictory.

Informed by `piece_count_distribution_10k.xlsx` (games ≥ 16 moves). Training uses bucket-stratified resampling.

---

### Policy (v1)

**Search-only** for v1. Add killer-move once tables are in.

**Policy guidance net (v2):** defer until after v1 Elo gate. Lightweight head off the **shared accumulator** (16 → move-ranking); watch per-node latency against the ~1 s budget.

**Compact transformer fallback:** the ~210K design in [chess transformer.md](chess%20transformer.md) stays in reserve — evaluate only if the lightweight policy head underperforms post-gate. Too heavy for per-node move ordering to pursue in parallel; last resort, not a parallel track.

***Killer moves** are a complementary heuristic for non-capture moves. The idea: if a particular quiet move (not a capture) caused a beta cutoff (i.e., it was so strong the search stopped exploring further alternatives at that depth) in one branch of the tree, it's often also a strong move in sibling branches at the same depth — even though the board position is slightly different. Chess tactics are often tied to squares and piece maneuvers rather than the exact position, so a move like "knight jumps to a strong central square" that worked well once tends to work well again nearby in the tree.*

---

### Incremental updates & lazy evaluation

1. **Add/sub** accumulator updates on each move (shared layer — bucket-independent)
2. **Lazy accumulator updates** — defer refresh until eval is actually needed (TT cutoffs skip work)
3. **Lazy evaluation** — skip full NNUE forward when a beta cutoff is already provable from a prior ply score; implement **together** with lazy accumulator updates (same "skip work when cutoff makes it moot" principle)
4. Copy-make + fused add/sub optional later

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
| Hidden activation | **CReLU** (v1) |
| Output activation | **tanh** — **LUT** at inference (no runtime `tanh`) |

**Scale calibration:** train fp32 first, histogram post-training weights, set per-tensor scales onto $[-127, 127]$ with minimal clipping.

**Tanh LUT:** after int16 output-head dot-product, clip to LUT index range, fetch centipawn score from table. Generate LUT offline from training scale + desired eval range; verify fp32 tanh vs LUT max error on validation set before Wio export.

**SCReLU fallback** — first upgrade if hidden CReLU plateaus below ≥1700 Elo: clip before square (int16), multiply-accumulate in int32 (avoid overflow). Output tanh + LUT unchanged.

**Grapheus / quantization-aware training:** skip for v1. Stay on **nnue-pytorch** + calibrated post-training quantization; investigate QAT only if the fp32→int8 gap threatens the Elo gate.

---

### Search

Phased rollout:

1. Alpha-beta + **quiescence**
2. **Futility pruning** + **late-move reduction** + **null-move pruning** (futility is cheap, well-proven at this speed class — include in v1, not deferred)
3. **Lazy evaluation** (paired with lazy accumulator updates)
4. **Iterative deepening** once TT is stable

**Stack surfing (MicroChess-style dynamic depth):** rejected for v1. With TT already claiming 128–160 KB of 192 KB RAM, probing free stack at runtime to extend depth is too risky alongside accumulators and search stack.

**Fixed depth / iterative deepening** within the ~1 s budget replaces dynamic stack-based depth.

---

### Memory

| Resource | Philosophy | Allocation |
|----------|------------|------------|
| **Flash** | Balanced | ~10% weights; rest search code + tables; **no opening book v1** |
| **RAM** | TT-dominant | TT **128–160 KB**; accumulators + stack ~16 KB; scratch ~16 KB |

**RAM risk:** 192 KB − 160 KB TT ≈ 32 KB margine — **TFT_eSPI buffer (~30 KB)** può consumare quasi tutto. Per partite reali: TFT off durante search, Serial/UCI per debug; TFT solo bring-up.

**Opening book:** defer until after Elo gate. Dog ships one, but flash is better spent on search + NNUE for v1.

**Dog (ESP32) reference:** Dog fits NNUE + TT + book in ~320 KB RAM at ~1700+ Elo on-device — proof the target is feasible. RAM layout study still useful (see Open Questions) but TT-dominant plan stands.

#### Transposition table

Design entry format first; slot count follows from 128–160 KB budget. Prototype on PC (build step 2), then benchmark on **Wio**.

**Candidate entry:** truncated zobrist, best move (~16 bit), score (16 bit), depth (8 bit), bound type (2 bit).

| Format | Size | Slots @ 128 KB | Trade-off |
|--------|------|----------------|-----------|
| Tight pack | ~10 B | ~13,100 | More entries; unaligned loads on SAMD51 cost extra shift/merge per probe |
| Byte-aligned | 16 B | ~8,200 | Fewer entries; faster probes |

**Decision metric:** wall-clock **nodes/sec** and **depth reached in ~1 s** on Wio — not hit-rate alone.

---

### Move ordering

**MVV-LVA + captures** for v1; **killer moves** when search depth > 4.

***Move ordering** is about the order in which a chess engine tries moves at each node in the search tree — not which moves are legal, but which ones get examined first. This matters enormously because of alpha-beta pruning: alpha-beta can skip whole branches of the search tree once it proves a move is "good enough" that the opponent would never let you reach it, but how much it can skip depends entirely on whether the best moves get searched early. If you search the best move first, alpha-beta prunes aggressively and the search is fast. If you search it last, you've wasted time fully exploring worse alternatives before finding it, and pruning barely helps. Good move ordering can be the difference between searching a few thousand nodes and a few million to reach the same depth.*

---

### Training data

**Primary (v1): subset Lc0 — target ~1–2 GB in repo**, non il corpus completo (centinaia di GB).

| Fonte | Ruolo | Note |
|-------|--------|------|
| **Lc0 training games** | Addestramento principale | Scaricare subset filtrato (es. partite forti, formato training-data / `.bin` Lc0); obiettivo **~1–2 GB** locale — sufficiente per prima NNUE bucketed |
| **Stockfish labels** | Target eval (centipawn) | Self-play o analisi SF su posizioni campionate dal subset Lc0 |
| **Kaggle `games.csv`** | Smoke test / statistiche | Già via `scripts/download_data.py`; **non** per training NNUE (partite deboli, label outcome) |
| **Filtro** | Games **≥ 16 mosse** | Allineato a `piece_count_distribution_10k.xlsx` |
| **Resampling** | Bucket-stratified | Queen-split (`bucket_id()`); vedi tabella Output buckets |

**Prossimo script:** `scripts/download_lc0.py` (o equivalente) — download incrementale, checksum, path sotto `data/raw/lc0/`. Valutare mirror ufficiali Lc0 / subset pre-processati della community prima di scaricare TB interi.

**Etichette:** nnue-pytorch si aspetta eval in centipawn da engine forte (Stockfish su subset Lc0 è l’opzione pragmatica se non si hanno le eval Lc0 native nel dump).

---

### Training pipeline

- **nnue-pytorch** for v1 network training (not Grapheus)
- Hidden **CReLU** + output **tanh** in the training graph; emit tanh LUT alongside int8 weights at export
- Calibrated post-training int8 export; measure fp32→int8 gap before considering QAT
- **SPSA** post-hoc for search/heuristic tuning

---

### I/O

**TFT + Serial** for on-device display and debug output.

**Minimal UCI over Serial:** yes — required for standardized Elo testing against the ≥1700 gate. Full UCI spec not needed; implement enough of the protocol for engine-vs-engine tools (cutechess-cli, etc.). Hardcoded FEN remains fine for early bring-up; UCI lands before or during Elo gate testing.

*TFT = Thin-Film Transistor LCD on the Wio Terminal (2.4" onboard screen).*

---

## Near-term roadmap (allineato al repo)

Ordine consigliato dopo la review [ai-feed.md](../ai-feed.md):

1. **Chiudere step 1** — golden FEN test, gate encoder 716 + bucket
2. **Search skeleton PC** — alpha-beta + quiescence, perft, eval hook (score costante), benchmark nodi/s
3. **Dati** — scaricare **~1–2 GB** subset Lc0; preprocessing verso nnue-pytorch
4. **Train bucketed NNUE** — shared accumulator 716→16, 8 teste; hidden CReLU, output tanh; int8 export + tanh LUT
5. **TT + search tuning** — formato entry, MVV-LVA, iterative deepening, SPSA
6. **Port Wio** — solo dopo parity PC; incremental accumulators; TFT opzionale in gioco

**Non fare ora:** policy head, MCTS, opening book, QAT — dopo gate **≥ 1700 Elo**.

---
