# SARDINE — Mission

**SARDINE** — *Small Artificial RAM-restricted Deep Intelligent Neural Engine*

Build a playable chess bot that runs **entirely on-device** on the Seeed **Wio Terminal** (~120 MHz, **192 KB RAM**, **~500 KB flash**): neural evaluation + alpha-beta search, maximizing **Elo per byte**. No cloud, no GPU. Ideally playable on *Lichess*.

| Target | Value |
| ------ | ----- |
| Elo | ≥ **1700** (match gate) |
| Move time | ~**1 s** |
| Eval | Micro **NNUE**: shared L1 `844 → W`, dual POV, **single** head `2W → 1` until bucket ablation §D (**F3**) |
| Labels | Teacher **`expected_reward`** (Lc0 WDL → White POV ∈ \([-1,+1]\)) — not Stockfish centipawns (**C1**) |
| Search | Alpha-beta + quiescence (+ pruning in v1); **Python** PC first, then pure **C** on Wio (**G1**) |
| Baseline | **Cfish** + stock NNUE as strong reference; student nets on the SARDINE path |
| Strength review | Stockfish **ACPL** (`eval_bot_acpl.py`) — not Cfish as judge (**D1**) |

## Path (high level)

Same spine as [PROJECT.md](PROJECT.md) §Progress Overview (**A1** — sole progress map):

1. **Run Cfish** — UCI baseline on PC  
2. **First NNUE** — stock net in Cfish; ACPL / nps baselines  
3. **Dataset** — Lichess primary + Lc0 supplement → uniform \((s, v)\) with Lc0 labels (**E1**)  
4. **Train the network** — small single-head student NNUE; ACPL vs Stockfish  
5. **On the hardware** — Wio smoke → strength/speed gates → Lichess  

**Thesis (later, J1):** task-vector / embedding-based bucketing vs hand partitions — after base NNUE + Elo path. See [NOTES/Thesis.md](NOTES/Thesis.md).

## Where we are (~2026-08-03)

| Stage | Status |
| ----- | ------ |
| Cfish smoke + stock NNUE | Done enough to build on |
| Mini labeled set + pilot / smoke train | Done (not production volume; pilots were multi-head — next train single-head) |
| Cfish self-play + ACPL; full Lichess dump; dedup | Open |
| Wio port / on-device gate | Not started |

**Progress:** [PROJECT.md](PROJECT.md)  
**Eng micro-gates:** [TODOs.md](TODOs.md) (mapped to PROJECT sections)  
**Architecture:** [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md)  
**Labels / paths:** [ASSETS.md](ASSETS.md)  
**Which file wins:** [ai-feed.md](ai-feed.md)

Session work is logged under local `DAILY-NOTES/` (gitignored).

**SARDINE** — *Small but mighty.* 🐟
