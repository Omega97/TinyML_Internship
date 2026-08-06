# Demo — pure NNUE self-play GIF

**One command** from the **repo root**: Python SARDINE **pilot NNUE** vs itself → PGN + GIF under `images/games/`.

```powershell
pip install -e ".[viz]"
# needs PyTorch for the checkpoint load
pip install torch

py -3.12 scripts/record_engine_game.py --eval nnue --nnue-checkpoint models/checkpoints/nnue/pilot_W128_844/best.pt --depth 1 --no-quiescence --max-plies 80 --headless --frame-ms 450 --output images/games/nnue_d1_demo.gif
```

(`--nnue-checkpoint` is optional if the default is still `pilot_W128_844/best.pt`.)

Also writes **`images/games/nnue_d1_demo.pgn`** (same stem). Open the GIF in a browser or image viewer.

---

## Engine used in this demo

In-process **SARDINE** (Python) with a **trained student value net** — not Cfish Hybrid, not pure HCE.

| Spec | This demo |
| ---- | --------- |
| **Stack** | `src/tinymlinternship/engine/` + `nnue/` + `features/` — pure Python |
| **Script** | `scripts/record_engine_game.py` (self-play + GIF in one pass) |
| **Sides** | Same policy White & Black |

### Value function

| Spec | Value |
| ---- | ----- |
| **Backend** | **`nnue`** — neural value only (no HCE blend) |
| **Code** | `src/tinymlinternship/engine/eval_nnue.py` · model `nnue/model.py` |
| **Checkpoint** | `models/checkpoints/nnue/pilot_W128_844/best.pt` (default) |
| **Encoder** | Dual-POV sparse features, dim **844** (`features/` — 716 base + 128 tactical) |
| **Net shape** | Hidden width **W=128** (pilot `BucketedNNUE`; multi-head pilots are experimental) |
| **Train target** | Teacher `expected_reward` ∈ **[-1, +1]** (ChessBench pilot for this ckpt) |
| **At search time** | Tanh value → **centipawn-like** score for αβ (White POV) |
| **Not used** | Cfish stock HalfKP NNUE · Python HCE material/PST as the leaf eval |

### Tree search

| Spec | Value |
| ---- | ----- |
| **Framework** | Thin **alpha-beta** (negamax) v0.3 — `engine/search.py` |
| **Depth** | **1** (`--depth 1`) — isolates **value quality** with almost no horizon |
| **Quiescence** | **Off** (`--no-quiescence`) — matches agent `nnue-w128-844-d1` |
| **Move ordering** | MVV-LVA when qsearch is on |
| **Not included** | TT, null-move, LMR, futility, iterative deepening (Cfish-class stack) |

### Runtime config

| Spec | Value |
| ---- | ----- |
| **Max plies** | 80 half-moves |
| **GIF** | `gifpgn`, 450 ms/frame, headless |
| **Prereqs** | `pip install -e ".[viz]"` + **torch**; checkpoint must exist on disk |

### Strength snapshot (optional)

Depth-1 pilot NNUE sits **above** random/HCE on the ACPL ladder (historical multi-game gate ~**ACPL 139** / Elo heur. **~1465**). Artifact: `plots/PGN_and_JSON/nnue_d1_gate_acpl.json`.  
**Depth 2** with this pilot often **collapses** (known issue) — prefer d1 for demos unless you intentionally stress search.

---

## Variants

```powershell
# Depth 2 + qsearch (slower; play can be unstable — agent nnue-w128-844-d2)
py -3.12 scripts/record_engine_game.py --eval nnue --depth 2 --max-plies 80 --headless `
  --output images/games/nnue_d2_demo.gif

# Other local pilot run
py -3.12 scripts/record_engine_game.py --eval nnue `
  --nnue-checkpoint models/checkpoints/nnue/smoke_prod_W128_844/best.pt `
  --depth 1 --no-quiescence --max-plies 80 --headless `
  --output images/games/nnue_smoke_d1_demo.gif
```

Existing reel files: `images/games/nnue_d1_game.gif`, `images/games/nnue_d2_game.gif`,  
`images/games/nnue_w128_844_d1_vs_nnue_w128_844_d1_2026-07-10.gif`.

---

## Related

| Path | Role |
| ---- | ---- |
| [demo-hce-gif.md](demo-hce-gif.md) | Pure HCE value (same thin search) |
| [demo-cfish-hybrid-gif.md](demo-cfish-hybrid-gif.md) | Cfish Hybrid + full SF-class search |
| `NOTES/agents/nnue-w128-844-d1.md` | Frozen bot recipe |
| `scripts/run_engine.py --eval nnue --depth 1` | Single-position best move |
| `scripts/eval_bot_acpl.py --eval nnue --depth 1 --no-quiescence` | ACPL gate (needs Stockfish) |
| [ASSETS.md](../ASSETS.md) | Checkpoint + train-data inventory |
