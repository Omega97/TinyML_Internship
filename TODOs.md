# SARDINE — Eng micro-gates

**Product progress lives only in [PROJECT.md](PROJECT.md)** (decision **A1**). This file expands eng detail under the same section names.  
Architecture: [blueprint](NOTES/SARDINE%20Engine%20Blueprint.md) (**B1**). Labels/data: [ASSETS.md](ASSETS.md) (**C1/E1**). Authority: [ai-feed.md](ai-feed.md).  
_Last updated: 2026-08-05 — F3 single-head path + mini train + ACPL gate; **F3** single head until §D._

| PROJECT section                  | This file            |
| -------------------------------- | -------------------- |
| Run Cfish                        | §1                   |
| First NNUE                       | §2                   |
| Dataset                          | §3                   |
| Train the Network                | §4                   |
| On the Hardware                  | §5 (+ §8 device eng) |
| For the Thesis                   | §6                   |
| Stretch Goals (encoder / search) | §7                   |

When a checkbox conflicts, **PROJECT wins** for product status; update both when closing a gate.

---

## Parking (research, not product)

- Ispezionare link del supervisore Zennaro su [Systematic Pruning](https://ieeexplore.ieee.org/abstract/document/11603432)
- Idee con Ansuini (task vectors / multi-head): **after** Elo path (**J1**) — see §6 / [NOTES/Thesis.md](NOTES/Thesis.md)

---

## 1 · Run Cfish

_PROJECT: Cfish smoke [x] · HCE/Cfish eval with Stockfish [ ]._

- [x] Cfish binary + launcher — `src/cfish/cfish.exe`, `run-cfish.bat`, `NOTES/Cfish.md`
- [x] UCI smoke — `uciok` / `readyok` (2026-07-31)
- [x] Formal UCI recipe — `go depth 5` → `bestmove` + archived **nps** (2026-08-04; depth-12 ~1.8M nps Hybrid)
- [x] Fix `scripts/cfish.py` stale `./cfish` path → `src/cfish/cfish.exe` + cwd `src/cfish/`
- [ ] **Cfish classical / HCE-style baseline ACPL** (if distinct from Python HCE gate) — `eval_bot_acpl.py` / UCI self-play; PROJECT open item
- [x] Python HCE ACPL gate already archived — `images/plots/PGN_and_JSON/hce_d1_gate*` (not a substitute for Cfish UCI baseline)

---

## 2 · First NNUE (stock net in Cfish)

_PROJECT: download / wire / hybrid smoke [x] · Cfish ACPL eval [ ]._

- [x] Stock NNUE on disk — `src/cfish/nn-62ef826d1a6d.nnue` (`make net` in `src/cfish/`)
- [x] Default EvalFile / INCBIN — `evaluate.h`, `nnue.c`, `ucioption.c`
- [x] Hybrid path available with net present (dedicated hybrid-only log optional)
- [x] **Cfish self-play + Stockfish ACPL** (D1 judge) — 2026-08-04 Hybrid d5, 3 games: ACPL **42.0** · Elo **2435**; `images/plots/PGN_and_JSON/cfish_hybrid_d5_gate*`; CLI `scripts/cfish_selfplay_pgn.py`
- [x] In-process ACPL stack ready — `scripts/eval_bot_acpl.py`, `src/tinymlinternship/bot_eval/` (used for HCE / pilot NNUE / Sunfish / Lc0 / Cfish)
- *Stock net ≠ SARDINE student. Custom SARDINE weights in Cfish = later / optional; device path is own C port.*

---

## 3 · Dataset

_PROJECT: raw download / teacher labels / uniform mini set [x] · dedup [ ] · full Lichess volume open._

- [x] Lc0 raw subset — `data/raw/lc0/`, `scripts/download_lc0.py` (~1–2 GB)
- [x] Lichess **smoke** PGN + FEN extract — `data/raw/lichess_smoke50.pgn`, `scripts/lichess_pgn_to_fen.py` → `data/processed/lichess/positions.parquet`
- [ ] Full **Lichess monthly dump** → `data/raw/lichess/` + large-scale FEN extract (E1 production)
- [x] Kaggle `games.csv` — smoke/stats only (`scripts/download_data.py`) — **not** NNUE train
- [x] ChessBench bags — encoder smoke only
- [ ] **Dedup + multiplicity** — no script yet (`src/tinymlinternship/data/` + CLI)
- [x] Teacher = **Lc0** → `expected_reward` only (C1) — `label_positions.py`, `models/teacher/lc0/`
- [x] Mini labeled blocks + merge — `lichess_labeled` / `lc0_labeled` → `data/processed/labeled/{train,val}.parquet` + `manifest.json`
- [ ] Label **full-volume** after Lichess dump (same teacher net)
- [x] Schema / merge tooling — `schema.py`, `merge_training_sets.py`
- *Stockfish = ACPL judge only (D1/I1: PATH / `STOCKFISH_PATH`), not train labels.*

---

## 4 · Train the Network

_PROJECT: pilot/smoke [x] · F3 single-head **code + mini train** [x] · ACPL pilot [x] · ACPL single-head mini recorded (strength fail) · full volume / Elo path still open · thesis MoE later [ ]._

Architecture (**F3** production): L1 `844 → W` dual POV → concat `2W` → **one** head `2W → 1` · CReLU · tanh → expected-reward LUT.  
*Legacy 8-head pilots = experimental only.*

- [x] Pilot / smoke train — `scripts/train_nnue.py`, `nnue/model.py` (legacy multi-head checkpoints under `models/checkpoints/nnue/`)
- [x] **Single-head** model + train path (F3) — `SingleHeadNNUE` + `--architecture single_head` default; `eval_nnue` loads both arches (2026-08-05)
- [x] Single-head train on **mini** labeled set — `single_W128_mini_ep30` · data `data/processed/labeled/{train,val}.parquet` · best val_mse **0.196** @ ep6
- [ ] Single-head train on **full volume** (Lichess dump + re-label after scale)
- [ ] **nnue-pytorch** adapt — 844-dim, single head, gradual L1 prune, 100 ep
- [ ] L1 gradual pruning 70–80% during training; sparse flash export
- [ ] PTQ int8 + tanh LUT; measure fp32→int8 gap (QAT only if MSE/Elo gap too large)
- [x] ACPL on pilot NNUE (D1) — `images/plots/PGN_and_JSON/nnue_d1_gate*`; d2 collapse known
- [x] ACPL on single-head mini (D1) — `images/plots/PGN_and_JSON/single_W128_mini_d1_gate*` · ACPL **~583** / Elo **400** (worse than random ~276; **not** a strength win)
- [ ] Playing-strength student ≥ pilot ladder (need more data / better train before Elo gate)
- [ ] **nps** microbench for student / search (still open; PROJECT notes)
- [ ] Teacher-only depth=1 playing-strength baseline (after a *playable* single-head net)
- [x] Piece-count distribution study — `scripts/plot_piece_count_distribution.py`, `data/excel/piece_count_distribution_10k.xlsx` (PROJECT Stretch [x])

### 4b · Bucket ablation (PROJECT Train / blueprint §D — after single-head baseline)

_**F3:** no multi-head ship until this finishes. Compare single-head vs candidate partitions only after a **playable** single-head baseline exists (mini MSE alone insufficient — need ladder not at Elo floor)._

- [x] Single-head val MSE on mini teacher-labeled set (best **0.196**; overfit train MSE ~0.006 @ ep30)
- [ ] Single-head baseline **playing strength** on teacher-labeled / ladder (open — mini gate failed)
- [ ] Per-partition MSE: single head vs 4 piece-count vs 8 queen-split (natural mix)
- [ ] Decisive vs ambiguous threshold; playing-strength only if ambiguous
- [ ] Lock scheme; sync PROJECT, blueprint, `bucket.py`, train/export

---

## 5 · On the Hardware

_PROJECT: Wio smoke / device ACPL / Lichess [ ] all open._

- [ ] Wio Terminal smoke test (no active device tree; pre-SARDINE sketches removed)
- [ ] Device eval vs Stockfish (ACPL / match) — ~30 games + nps; host SF judge
- [ ] Connect Wio to Lichess and play
- *Eng detail: §8 (accumulators, C port, UCI/Serial).*

---

## 6 · For the Thesis

_PROJECT: compare techniques [ ] — **J1** after base NNUE + Elo path._

- [ ] Task-vector bucketing vs embedding clustering — [NOTES/Thesis.md](NOTES/Thesis.md), blueprint §Later
- [ ] Compact-transformer fallback criteria (v2 policy) — define “underperforms” before invoking

---

## 7 · PC SARDINE stack (PROJECT Stretch Goals — detail)

_PROJECT marks encoder / αβ+qsearch / MVV-LVA / dual-POV [x]. Eng depth for remaining search/TT work._

### 7a · Feature encoder

- [x] 716 base + 128 tactical → **844** — `features/index_map.py`, `encoder.py`, `mirror.py`, `tactical.py`
- [x] Dual-perspective `encode_dual()`; enemy-king full 64; castling frame fix
- [x] Bucket **metadata** helpers (`bucket.py`) — analysis / future §D only; **not** production multi-head (**F3**)
- [x] Tests — `tests/test_features.py`, `test_tactical.py`
- [ ] Encoder parity on **device** (with §8 C port)

### 7b · Search skeleton on PC

- [x] perft — `engine/perft.py`, `tests/test_perft.py`
- [x] HCE + 1-ply + alpha-beta + capture quiescence + MVV-LVA — `engine/search.py` v0.3
- [x] `record_engine_game.py` + GIF demos; NNUE eval hook `eval_nnue.py`
- [x] Depth-1 ACPL ladder (HCE / NNUE / Sunfish) — see PROJECT artifacts
- [ ] TT entry format prototype + PC benchmark
- [ ] Nodes/s benchmark on PC
- [ ] Node-budget model vs Urusov ESP32 (~20 kNps) once eval latency known

### 7c · Full search stack (v1 remaining)

- [x] Alpha-beta + quiescence (PC)
- [ ] Futility + LMR + null-move
- [ ] Lazy evaluation (with lazy accumulators)
- [ ] Iterative deepening (TT stable)
- [ ] TT **128–160 KB** entry format (Wio metric: nodes/s + depth @ ~1 s)
- [ ] Killer moves (depth > 4)
- [x] Countermove / full history suite — **out of v1**
- [ ] SPSA on search params only

---

## 8 · Device eng (supports PROJECT §On the Hardware)

### 8a · Incremental accumulators

- [ ] Lazy add/sub on shared L1 (no bucket router while F3)
- [ ] Full refresh on king centre-file crossing
- [ ] Lazy accumulator updates (TT cutoffs)
- [ ] Castling-bit / EP-bit add/sub

### 8b · Port search + NNUE to C (Wio)

- [ ] C engine core after playable PC search
- [ ] Benchmark `-O3` vs `-Os` on Wio
- [ ] Sparse L1 int8 + tanh LUT in flash; int16 accumulators in RAM
- [ ] Minimal UCI over Serial; TFT off during search

### 8c · Elo ship gate

- [ ] ≥ 1700 **`elo_match`** under frozen protocol (ACPL heuristic alone insufficient — blueprint)
- [ ] PC↔device parity on golden FENs

### 8d · Iterate if gate missed

- [ ] SCReLU fallback (hidden)
- [ ] QAT only if PTQ gap too large
- [ ] TT / scheme revision after §D if multi-head enabled

### 8e · v2 (after gate)

- [ ] UCI polish · policy head · opening book · tactical MoE · transformer fallback — as needed

**v1 non-goals (aligned with PROJECT Cfish strip list):** MCTS · multi-head before §D · book · TB · SMP/NUMA · full history move ordering · HalfKP student · desktop-only SIMD as sole NNUE kernel · autoencoder warm-start · MicroChess stack surfing.

---

## Cross-check: open in PROJECT → eng follow-ups

| PROJECT open item | Eng follow-up here |
| ----------------- | ------------------ |
| Cfish / HCE eval with SF | §1 |
| Cfish stock-NNUE ACPL | §2 |
| Dedup + multiplicity | §3 |
| Full Lichess + re-label | §3 |
| Single-head code + mini train [x]; full volume + strength | §4 |
| Thesis MoE / task vectors | §6 (after Elo) |
| Wio smoke / device eval / Lichess | §5 + §8 |
| nps microbench | §2 / §4 / §7b |
| Killers / TT / futility / LMR | §7c |

---

## Done reference (do not re-open without cause)

- Dog ESP32 RAM feasibility note (blueprint §Memory)
- Stretch: piece-count study, 844 encoder, αβ+qsearch, MVV-LVA, dual-POV (PROJECT [x])

---

#core
