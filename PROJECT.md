# Project: SARDINE

**SARDINE** — *Small Artificial RAM-restricted Deep Intelligent Neural Engine*

Chess engine for the **Wio Terminal**: neural evaluation + alpha-beta search, maximizing **Elo per byte** under **192 KB RAM** / **~500 KB flash**. No cloud, no GPU. Target: playable bot (ideally on *Lichess*).

| Doc | Role |
| --- | ---- |
| [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md) | Spec, architecture, pipeline, design decisions |
| [TODOs.md](TODOs.md) | Checkpoint checklist vs blueprint (**progress source of truth**) |
| Daily notes | Session plan + execution log under `DAILY-NOTES/` (**local / gitignored**) |
| [ASSETS.md](ASSETS.md) | Paths, teachers, label uniformity |
| [Goal.md](Goal.md) | Short mission statement |
| [ai-feed.md](ai-feed.md) | Slim code map for agents |
| [NOTES/Thesis.md](NOTES/Thesis.md) | Later research: task vectors / optimal bucketing |

_Last reassessment: 2026-07-22 (decisions A4,B3,C2,D1,E3,F1,G3,H2,I1,J1)._

---

## Overview

Build a complete, playable chess bot that runs **entirely on-device** on the Seeed Wio Terminal. Primary path: PC bring-up (Python search + training) → pure **C** core on device. Spec: blueprint.

## Goals

- **Primary:** ≥ **1700 Elo** on-device (match gate), best move within **~1 s**
- **Secondary:** Reproducible train → export → device pipeline; minimal UCI for engine-vs-engine tests
- **Non-goals (v1):** MCTS · policy net · opening book · full UCI polish · tactical MoE · Grapheus/QAT by default · MicroChess stack surfing

## Key design decisions

| Topic | Decision |
| ----- | -------- |
| **Search** | Alpha-beta (not MCTS); PC skeleton first, then C on Wio |
| **Eval** | Bucketed micro NNUE: shared L1 `844 → W` ($W \in \{128,256\}$), dual POV, expert heads `2W → 1` |
| **Buckets (interim G3)** | **Code ships 8** (piece count + queen-split). Blueprint default table is **4** piece-count-only. **Keep 8 until §D ablation** decides; do not silent-migrate |
| **Labels** | Teacher **`expected_reward`** only (Lc0 WDL → White POV $[-1,+1]$) — never mix `best_q` / game result |
| **Train framework (target)** | nnue-pytorch adapted; **now:** `scripts/train_nnue.py` (pilot + mini prod) |
| **Quantization** | PTQ int8 first; QAT only if MSE/Elo gap too large |
| **Stockfish (ACPL judge)** | **System PATH / `STOCKFISH_PATH` only** — no in-repo binary |
| **HF study nets** | Code kept; **weights not shipped** (re-download if needed) |
| **Legacy pre-SARDINE** | **Removed** (2026-07-22) |
| **ChessBench** | Smoke / pilot wiring only — not production train |
| **Daily notes** | `DAILY-NOTES/` only, **gitignored** |

## Development guidelines

- **Python:** 3.12; `pip install -e ".[train,viz]"` as needed
- **Tests:** `py -3.12 -m pytest tests/ -q` after code changes
- **Progress:** update `TODOs.md` checkboxes; write a daily note under `DAILY-NOTES/YYYY-MM/` each work session
- **Labels:** follow [ASSETS.md](ASSETS.md) uniformity rule
- **No scope creep:** defer blueprint non-goals until Elo gate

## Current status & roadmap

| Scope | Bar | % | Note |
| ----- | --- | - | ---- |
| **v1 → Elo gate** | `████░░░░░░` | **~38%** | Encoder + PC search + mini train path; no device, no full search stack, no match gate |
| **Device ship** | `░░░░░░░░░░` | **0%** | Wio port not started |

| Step | Bar | % | Status |
| ---- | --- | - | ------ |
| **A** · Feature encoder | `█████████░` | **88%** | 844 dual POV + 8 buckets ✅ · device parity with F |
| **B** · Search skeleton PC | `████████░░` | **75%** | v0.3 αβ + qsearch + MVV-LVA + NNUE hook · TT / nodes/s ❌ |
| **C** · Train bucketed NNUE | `██████░░░░` | **55%** | Mini labels + merge + smoke train ✅ · full volume / nnue-pytorch / prune / PTQ ❌ |
| **C1** · Teacher@d1 baseline | `██░░░░░░░░` | **20%** | Tooling exists · systematic ladder not done |
| **D** · Bucket ablation | `░░░░░░░░░░` | **0%** | Decides 8 queen-split vs 4 piece-count (and peers) |
| **D2** · Optimal bucketing | `░░░░░░░░░░` | **0%** | Later — Thesis.md |
| **E–F** · Accumulators / C port | `░░░░░░░░░░` | **0%** | After stronger PC net + search |
| **G** · Full search stack | `███░░░░░░░` | **25%** | qsearch + MVV-LVA ✅ · rest open |
| **H** · Elo gate ≥1700 | `░░░░░░░░░░` | **0%** | ACPL heuristic only today |

**Critical path (next):** longer mini-set train → d1 ACPL → full Lichess volume + re-label → nnue-pytorch/prune/PTQ → TT/nodes/s → search polish → device.

**Known issues:** NNUE d2 ACPL collapse vs d1 (2026-07-20 notes); val set thin for per-bucket metrics; blueprint 4-bucket table vs code 8-bucket until D.

## Deliverables

| Priority | Deliverable |
| -------- | ----------- |
| **P0** | Playable PC engine + production-labeled train set + NNUE checkpoint |
| **P0** | Search stack + TT; Elo **match** path toward ≥1700 |
| **P0** | C port on Wio @ ~1 s/move with parity |
| **P1** | Minimal UCI; bucket ablation (D); PTQ export |
| **P2** | Policy head / book / thesis bucketing (after gate) |

## Reassessment cleanup (2026-07-22)

| Decision | Action |
| -------- | ------ |
| A4 | Hard-deleted `legacy/pre-sardine/` |
| B3 | Stockfish via PATH / env only |
| C2 | Removed `models/teacher/hf/` weights; code kept |
| D1 | Cleared local `terminals/` |
| E3 | Daily notes only under `DAILY-NOTES/` (gitignored); moved 07-20/07-21 there |
| F1 | Kept ChessBench smoke path |
| G3 | Stay on **8** buckets until ablation D |
| H2 | PROJECT = status; Goal short; Report archived; ai-feed slimmed |
| I1 | Removed empty `core/`, `evaluation/`, `datasets/`, `models/` packages + stale pyc |
| J1 | Kept all `images/games/` demos |

---

#core
