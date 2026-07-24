# SARDINE Blueprint — Progress

_Checkpoint map vs NOTES/SARDINE Engine Blueprint.md only. Last updated: 2026-07-22 (reassessment cleanup; buckets stay **8** until §D)._

---

- Ispezionare link del supervisore Zennaro su [Systematic Pruning](https://ieeexplore.ieee.org/abstract/document/11603432)
    
- Idee con Ansuini:
	Task vectors (in weight space / representation space) T1, T2, …, T8 have specific vectors in the weight space. Is it possible / useful to define task vectors that bring you from one head to another with minimal information/performance loss? This may be the case if the head vectors live in a linear submanifold of low dimensionality (are we so lucky? Maybe not!)

---

## Build pipeline

Stesso ordine del diagramma _Build Pipeline_ nel blueprint.

### A · Feature encoder (PC + device parity)

- [x] Pruned **716** base features (pawn prune, king mirror, castling, EP) — `index_map.py`, `encoder.py`, `mirror.py`
- [x] Tactical extension **+128** (under-attack + king-attacker, 64+64) → **844** total — `tactical.py`
- [x] Dual-perspective sparse input — `encode_dual()`
- [x] Enemy-king full 64-square resolution (only the perspective king is mirror-compressed to 32 slots)
- [x] **Castling coordinate-frame fix** — rights from `base` + K/Q swap if mirrored (EP from `view`); golden startpos + `test_castling_*`
- [x] 8-bucket router (piece count + queen-split), boundaries closed at **p ≤ 12** (no gap)
- [x] Gate test (`tests/test_features.py`, `test_tactical.py`, golden FEN) — **33** tests
- [ ] Encoder parity on **device** (C, with search) — step F

---

### B · Search skeleton on PC

- [x] perft — `engine/perft.py`, `tests/test_perft.py` (d5 = 4 865 609)
- [x] HCE eval bring-up — `evaluate_hce` (`tests/test_engine.py`)
- [x] 1-ply search bring-up — `search_best_move` (= `search(..., depth=1)`)
- [x] Alpha-beta — fixed-depth negamax (`search(board, depth)`), engine **v0.2**
- [x] Capture quiescence — depth-0 leaves, MVV-LVA noisy moves, `quiescence=False` opt-out; engine **v0.3**
- [x] MVV-LVA move ordering — main search + qsearch (killers: step G)
- [x] `record_engine_game.py --depth` + GIF depth-2 — `images/nnue_d2_game.gif`, `images/hce_d2_game.gif` (`--no-quiescence` per HCE)
- [x] Eval hook at depth-search nodes — `evaluate_nnue` + `--eval nnue` (`eval_nnue.py`, `make_eval_fn`)
- [x] Gate depth-1 ACPL (§Bot Evaluation) — NNUE **121** · HCE **275** · Sunfish **818** cp (`eval_bot_acpl.py`, `sunfish_selfplay_pgn.py`, `tests/test_bot_eval_acpl.py`)
- [ ] TT entry format prototype + PC benchmark
- [ ] Nodes/s benchmark on PC
- [ ] Node-budget model vs Urusov ESP32 baseline (~20 kNps, heuristics-only) — estimate reachable depth once eval latency is measured

---

### C · Train bucketed NNUE

Architecture: shared L1 **844 → W** with $W \in \{128, 256\}$ (dense train + **gradual prune** 70–80% in training; **non-zero only in flash**; dual call, same weights twice — own-POV + opponent-POV) → concat **2W** → router → expert **2W → 1** × 8 · **CReLU** hidden · **tanh** output → expected-reward LUT in $[-1,+1]$.

- [x] Lc0 subset **~1–2 GB** (`data/raw/lc0/`, `scripts/download_lc0.py`)
- [x] Games **≥ 16** moves; pilot preprocess — `lc0_preprocess.py`, `stats_lc0_processed.py`, `prepare_lc0_dataset.py` (ply≥32 global, bucket7≥8)
- [x] Survey pre-labeled datasets — nessun dump riutilizzabile end-to-end; vedi [NOTES/Datasets.md](NOTES/Datasets.md)
- [x] Lichess PGN → FEN script — `scripts/lichess_pgn_to_fen.py` (streaming, `bucket_id` / piece_count / has_queen); smoke 50 games → `data/processed/lichess/positions.parquet` (all 8 buckets)
- [ ] Full Lichess dump download + large-scale FEN extract (production volume; natural bucket mix) + Lc0 supplement at scale
- [x] Teacher scelto: **Lc0** (`expected_reward` White POV from WDL, on-the-fly UCI); fallback Stockfish WDL — [NOTES/Models.md](NOTES/Models.md) · [ASSETS.md](ASSETS.md) §Uniformity
- [x] Teacher installato — `models/teacher/` (lc0 v0.32.1 + reti); `scripts/download_teacher.py`, `smoke_test_teacher.py` OK
- [x] `label_positions.py` — scaffold + smoke; **uniform target = `expected_reward` only** (same formula for every source block)
- [x] Label **mini** production blocks (Lichess **2371** + Lc0 **3149**) with teacher **`791556`** → `lichess_labeled` / `lc0_labeled`; no `best_q` as train target
- [ ] Label **full-volume** production blocks after Lichess dump / large extract (same teacher; multi-session)
- [x] Schema/merge tooling — `schema.py`, `merge_training_sets.py` (game-level split + single-teacher check)
- [x] Merge mini set → `data/processed/labeled/{train,val}.parquet` + `manifest.json` (5306 / 214, seed 42, 2026-07-20)
- [ ] **nnue-pytorch** fork/adapt — 844-dim bucketed, gradual L1 prune, 100 ep fixed
- [x] PyTorch **smoke** pilot (shared L1 + 8 expert heads) — ChessBench `pilot_W128_844` val_mse **0.056**; production mini `smoke_prod_W128_844` val_mse **0.247** (2 ep, encode-from-fen)
- [ ] Calibrated **PTQ int8** export + tanh LUT (histogram weights, scale onto [-127,127])
- [ ] Measure fp32→int8 gap — QAT fallback if MSE > 0.01 or Elo drop > 30
- [ ] L1 gradual pruning 70–80% **during training**; sparse flash export
- [ ] Teacher-only **depth=1** playing-strength baseline (after first net)
- [x] Kaggle `games.csv` smoke only (not NNUE training) — `scripts/download_data.py`
- [x] Piece-count distribution for bucket design — `plot_piece_count_distribution.py`, `excel/piece_count_distribution_10k.xlsx`

---

### D · Bucket ablation (decides scheme)

_Interim (reassessment G3, 2026-07-22): **keep 8-bucket queen-split in code** until this section decides. Blueprint default table lists 4 piece-count-only as a candidate baseline — do not silent-migrate._

- [ ] Per-bucket eval MSE: **8 queen-split** vs **4 piece-count-only** (and other splits if useful) on teacher-labeled val (natural mix)
- [ ] Define "decisive vs ambiguous" threshold (e.g. >5% relative MSE change per bucket = decisive; 2–5% or mixed-direction buckets = ambiguous → escalate)
- [ ] Playing-strength test — **only if** per-bucket results are ambiguous or contradictory
- [ ] Lock final `NUM_BUCKETS` + router; sync blueprint, `bucket.py`, train/export, docs

---

### E · Incremental accumulators (device)

- [ ] Lazy add/sub on shared layer (bucket-agnostic)
- [ ] Full refresh on king centre-file crossing
- [ ] Lazy accumulator updates (TT cutoffs)
- [ ] Castling-bit add/sub (rare — only on king/rook moves or rook capture)
- [ ] En-passant-bit add/sub (frequent — flips near every ply, but cheap: single bit)

---

### F · Port search + NNUE to C (Wio)

- [ ] C engine core (after playable PC search)
- [ ] Benchmark **`-O3` vs `-Os`** on Wio
- [ ] Sparse L1 int8 weights + tanh LUT in flash; int16 accumulators ($W$ per POV) in RAM

---

### G · Full search stack + tuning

Phased rollout dal blueprint (tutto v1, non rinviato salvo dove indicato):

- [x] Alpha-beta + **quiescence** (PC minima — catture/promozioni in foglia, `search.py` v0.3)
- [ ] **Futility** + **LMR** + **null-move**
- [ ] **Lazy evaluation** (paired with lazy accumulators)
- [ ] **Iterative deepening** (TT stable)
- [ ] TT **128–160 KB** — format decision: ~10 B tight pack vs 16 B byte-aligned entry, decided by wall-clock nodes/sec + depth reached on Wio, **not** hit-rate alone
- [ ] Move ordering: **killer moves** (depth > 4) — MVV-LVA ✅ in main/qsearch
- [x] Countermove history — **out of v1** per blueprint (killers only at depth > 4)
- [ ] **SPSA** search/heuristic tuning

---

### H · Elo gate

- [ ] **≥ 1700 Elo** (blueprint gate) — e.g. cutechess-cli
- [ ] Minimal **UCI over Serial**
- [ ] TFT **off** during search; Serial for debug

---

### I · Iterate (if gate missed)

- [ ] SCReLU fallback (hidden layer) — clip int16 accumulator to activation range **before** squaring (load-bearing, avoids overflow); square in int16; multiply-accumulate with int8 weights in **int32**
- [ ] Quantization-aware training (only if int8 gap too large) — stay on nnue-pytorch first; only evaluate Grapheus or in-pipeline QAT if PTQ gap threatens the Elo gate
- [ ] TT format / bucket scheme revision

---

### J · v2 (after gate)

- [ ] Minimal UCI polish
- [ ] Policy guidance head (off shared accumulator, $W \rightarrow$ move-ranking; watch per-node latency vs ~1 s budget)
- [ ] Opening book
- [ ] SCReLU / QAT / compact transformer fallback (~210K design) — only if needed
- [ ] Tactical MoE axis (`inCheck`, capture threat) — only if switching cost analysis shows it's worth it earlier than assumed

**Explicitly deferred in blueprint (v1):** MCTS · tactical MoE heads · autoencoder warm-start · separate pattern tables · opening book · Grapheus/QAT · MicroChess stack surfing · MicroChess bare-metal patterns.

---

## Open questions / research (not blocking, but untracked otherwise)

- [x] Dog (ESP32) RAM budget study — feasibility reference in blueprint §Memory; TT-dominant plan unchanged
- [ ] Compact-transformer fallback evaluation criteria — define what "underperforms" means for the v2 policy head before deciding to invoke this fallback

---
