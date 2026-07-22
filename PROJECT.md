# SARDINE Project

**SARDINE** — *Small Artificial RAM-restricted Deep Intelligent Neural Engine*

Chess engine for the **Wio Terminal**: neural evaluation + alpha-beta search, maximizing **Elo per byte** under **192 KB RAM** / **~500 KB flash**. No cloud, no GPU. Target: playable bot (ideally on *Lichess*).

| Doc | Role |
| --- | ---- |
| [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md) | Spec, architecture, pipeline, design decisions |
| [TODOs.md](TODOs.md) | Checkpoint checklist vs blueprint |
| [NOTES/Thesis.md](NOTES/Thesis.md) | Later research: task vectors / optimal bucketing |
| [ASSETS.md](ASSETS.md) | Paths, teachers, uniformity of labels |

_Last progress sync: 2026-07-21 (from TODOs + daily notes)._

---

## Targets

| Parameter | Decision |
| --------- | -------- |
| **Elo** | ≥ **1700** (minimum gate; higher is better) |
| **Move time** | Best move within **~1 s** |
| **Search** | Alpha-beta (not MCTS — no policy net in v1) |
| **Eval** | Bucketed micro **NNUE** (policy head maybe v2) |
| **Device** | Wio Terminal · pure **C** core after PC bring-up |
| **RAM / Flash** | TT-dominant 128–160 KB · sparse int8 weights in flash |

Node-budget reference: Urusov ESP32 (~20 kNps, heuristics-only) for search throughput without NNUE.

---

## Overall progress

| Scope | Bar | % | Note |
| ----- | --- | - | ---- |
| **v1 → Elo gate** | `████░░░░░░` | **~38%** | Encoder + PC search skeleton + mini train path; no device, no full search stack, no gate |
| **Device ship** | `░░░░░░░░░░` | **0%** | Wio port not started |

Rough weights: A–C heavy early wins; E–H still open.

---

## Build pipeline progress

Same order as the blueprint *Build Pipeline*. Detail in [TODOs.md](TODOs.md).

| Step                                         | Bar          | %       | Status                                                                                        |
| -------------------------------------------- | ------------ | ------- | --------------------------------------------------------------------------------------------- |
| **A** · Feature encoder (PC + device parity) | `█████████░` | **88%** | 844-dim dual POV + 8 buckets ✅ · device parity with F                                         |
| **B** · Search skeleton on PC                | `████████░░` | **75%** | Engine v0.3 (αβ + qsearch + MVV-LVA + NNUE hook + d1 ACPL) ✅ · TT / nodes/s ❌                 |
| **C** · Train bucketed NNUE                  | `██████░░░░` | **55%** | Teacher Lc0, mini labels + merge + smoke train ✅ · full volume / nnue-pytorch / prune / PTQ ❌ |
| **C1** · Teacher-only depth=1 baseline       | `██░░░░░░░░` | **20%** | Tooling exists · systematic teacher@d1 ladder not done                                        |
| **D** · Queen-split ablation                 | `░░░░░░░░░░` | **0%**  | After production train + decent val                                                           |
| **D2** · Optimal bucketing (task vectors)    | `░░░░░░░░░░` | **0%**  | Later research — [Thesis.md](NOTES/Thesis.md); off critical path                              |
| **E** · Incremental accumulators             | `░░░░░░░░░░` | **0%**  | Device / search path                                                                          |
| **F** · Port search + NNUE to C (Wio)        | `░░░░░░░░░░` | **0%**  | After playable PC stack                                                                       |
| **G** · Full search stack + tuning           | `███░░░░░░░` | **25%** | αβ + capture qsearch + MVV-LVA ✅ · futility/LMR/null/ID/TT/killers/SPSA ❌                     |
| **H** · Elo gate ≥ 1700                      | `░░░░░░░░░░` | **0%**  | Minimal UCI + match protocol                                                                  |
| **I** · Iterate if gate missed               | `—`          | **—**   | SCReLU · QAT · TT / buckets (on demand)                                                       |
| **J** · v2 after gate                        | `░░░░░░░░░░` | **0%**  | UCI polish · policy head · book · fallbacks if needed                                         |

```text
Pipeline (v1 critical path)

  A █████████░ 88%  ──►  B ████████░░ 75%  ──►  C ██████░░░░ 55%
                                                      │
                                                      ▼
                                               C1 ██░░░░░░░░ 20%
                                                      │
                                                      ▼
                                               D  ░░░░░░░░░░  0%
                                                      │
                            (D2 later · research)     ▼
                                               E  ░░░░░░░░░░  0%
                                                      │
                                                      ▼
                                               F  ░░░░░░░░░░  0%
                                                      │
                                                      ▼
                                               G  ███░░░░░░░ 25%
                                                      │
                                                      ▼
                                               H  ░░░░░░░░░░  0%   Elo ≥ 1700?
                                                  yes ──► J v2
                                                  no  ──► I iterate ──► G
```

---

## Architecture (v1)

Canonical diagrams and decisions: [SARDINE Engine Blueprint](NOTES/SARDINE%20Engine%20Blueprint.md).

```text
844 sparse features (own POV)  ──┐
                                 ├──► shared L1  844 → W   (W ∈ {128, 256})
844 sparse features (opp POV)  ──┘         │
                                    dual accumulators (int16)
                                           │
                                    concat → 2W
                                           │
                              bucket router (piece count + queen-split)
                                           │
                              expert head i:  2W → 1   (×8)
                                           │
                              tanh LUT → expected reward ∈ [-1, +1]
```

| Piece | Choice |
| ----- | ------ |
| **Features** | **844** = 716 base (piece-square, king mirror, castling, EP) + **128** tactical (under-attack + king-attackers) |
| **L1** | Dense train → **gradual prune 70–80%** → sparse **int8** in flash; shared across experts |
| **Experts** | 8 output heads; router by piece count + queen presence (see blueprint bucket table) |
| **Activations** | CReLU hidden · **tanh LUT** output (no runtime tanh) |
| **Search (target)** | αβ + quiescence · futility · LMR · null-move · lazy eval · iterative deepening · killers (d>4) · SPSA on search only |
| **Search (now)** | **v0.3** fixed-depth αβ + capture quiescence + MVV-LVA; HCE default; NNUE via `--eval nnue` |
| **Policy** | **Search-only v1**; lightweight head off accumulator deferred to **v2** |
| **Runtime** | PC bring-up first (Python/C++ skeleton) → pure **C** on Wio; TFT + Serial; minimal UCI for Elo tests |

---

## Data & training

| Item | Decision |
| ---- | -------- |
| **Target label** | Teacher **`expected_reward`** only (WDL → \(W-L\)), White POV — not centipawns, not chunk `best_q` |
| **Teacher** | **Lc0** latest best net via UCI; fallback Stockfish WDL |
| **Positions** | **Lichess** human games (diversity) + **Lc0** training games (volume); games ≥ 16 moves |
| **Buckets** | Natural distribution — no stratified resampling |
| **Framework (target)** | **nnue-pytorch** adapted to 844-dim bucketed MoE |
| **Framework (now)** | Custom PyTorch (`scripts/train_nnue.py`) — pilot + mini production smoke |
| **Quantization** | **PTQ int8** first; QAT only if MSE gap > 0.01 or Elo drop > 30 |
| **Smoke data** | Kaggle `games.csv` / ChessBench — wiring only, not production train |

**Current mini production set** (2026-07-20): Lichess 2371 + Lc0 3149 labeled with teacher `791556` → merge **5306 / 214** train/val · smoke `smoke_prod_W128_844` val_mse **0.247** (2 ep). Pilot ChessBench `pilot_W128_844` val_mse **0.056**.

Sources: [database.lichess.org](https://database.lichess.org/) · Lc0 training data · [NOTES/Datasets.md](NOTES/Datasets.md) · [NOTES/Models.md](NOTES/Models.md).

---

## Models

### Production target (SARDINE NNUE)

| | |
| --- | --- |
| **Arch** | Shared L1 `844 → W` + 8 experts `2W → 1`, dual perspective, bucket router |
| **W** | Empirically **128** or **256** (latency vs strength on Wio) |
| **Output** | Expected reward in \([-1,+1]\) via tanh LUT |
| **Size goal** | Sparse L1 (~20–30% non-zero) + 8 dense heads — fit flash budget |
| **Use** | Leaf eval under alpha-beta on Wio |

Training / export path: dense train → gradual prune → calibrated int8 + LUT. See blueprint §Evaluation / §Quantization.

### Eval / search baselines (now)

| Recipe | Role |
| ------ | ---- |
| **HCE** | Default handcrafted eval for engine bring-up |
| **NNUE pilot** | `pilot_W128_844` — ChessBench smoke, wired to search |
| **NNUE mini-prod** | `smoke_prod_W128_844` — teacher-labeled mini set (pipeline proof) |
| **Sunfish / ladder** | Opponents for ACPL and match tracks (blueprint §Benchmark) |

Depth-1 ACPL (16g, heuristic Elo): NNUE pilot ~**1465** · HCE / Sunfish weak — not the ≥1700 match claim.

### Deferred / reserve (not v1 critical path)

| Track | When |
| ----- | ---- |
| Policy guidance head off shared accumulator | v2, after Elo gate |
| Compact transformer (~210K) | Last-resort policy fallback |
| Task-vector optimal bucketing | After queen-split ablation ([Thesis.md](NOTES/Thesis.md)) |
| Tactical MoE (`inCheck`, threats) | v1.x / v2 if switching cost warrants it |

Historical survey of external value/policy nets lives in notes/archive; **SARDINE is eval-NNUE + search**, not a standalone policy MCU net.

---

## Explicitly deferred (v1)

MCTS · tactical MoE heads · autoencoder warm-start · separate pattern tables · opening book · Grapheus/QAT by default · MicroChess stack surfing / bare-metal patterns · full UCI polish.

---

## Fallback ladder (if Elo slips)

| Level | Trigger | Action |
| ----- | ------- | ------ |
| 1 | Eval > 3 ms | Reduce \(W\) 256→128 or prune harder |
| 2 | Depth < 6 | TT 128→64 KB |
| 3 | Elo < 1600 | Aggressive pruning / SPSA |
| 4 | Still < 1600 | Remove MoE → single expert |
| 5 | Still < 1500 | Heuristics-only eval |
| 6 | Timeboxed | Material-only |

L1–2 autonomous; L3+ supervisor sign-off (blueprint).

---

## Related research (not blocking)

- Supervisor link: [Systematic Pruning](https://ieeexplore.ieee.org/abstract/document/11603432) (Zennaro)
- Task vectors / low-dim expert subspace (Ansuini) — [Thesis.md](NOTES/Thesis.md)

---

#core
