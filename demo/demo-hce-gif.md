# Demo — pure HCE self-play GIF

**One command** from the **repo root**: Python SARDINE **HCE** vs itself → PGN + GIF under `images/games/`.

```powershell
pip install -e ".[viz]"

py -3.12 scripts/record_engine_game.py `
  --eval hce `
  --depth 1 `
  --no-quiescence `
  --max-plies 80 `
  --headless `
  --frame-ms 450 `
  --output images/games/hce_d1_demo.gif
```

Also writes **`images/games/hce_d1_demo.pgn`** (same stem). Open the GIF in a browser or image viewer.

---

## Engine used in this demo

In-process **SARDINE** (Python), **not** Cfish / stock NNUE.

| Spec | This demo |
| ---- | --------- |
| **Stack** | `src/tinymlinternship/engine/` — pure Python |
| **Script** | `scripts/record_engine_game.py` (self-play + GIF in one pass) |
| **Sides** | Same policy White & Black |

### Value function

| Spec | Value |
| ---- | ----- |
| **Backend** | **`hce`** — hand-crafted evaluation only |
| **Code** | `src/tinymlinternship/engine/eval_hce.py` |
| **Terms** | Material (classic piece values) + PeSTO-style **piece-square tables** |
| **Scale** | Centipawns, White POV (`+` = better for White) |
| **Neural net** | **None** — no NNUE weights, no checkpoint |

### Tree search

| Spec | Value |
| ---- | ----- |
| **Framework** | Thin **alpha-beta** (negamax) v0.3 — `engine/search.py` |
| **Depth** | **1** (`--depth 1`) — one ply of main search |
| **Quiescence** | **Off** (`--no-quiescence`) for this recipe (fast, matches d1 ladder) |
| **Move ordering** | MVV-LVA on captures (when qsearch is on) |
| **Not included** | TT, null-move, LMR, futility, iterative deepening (Cfish-class stack) |

### Runtime config

| Spec | Value |
| ---- | ----- |
| **Max plies** | 80 half-moves |
| **GIF** | `gifpgn`, 450 ms/frame, headless |
| **Prereqs** | `pip install -e .` and `pip install -e ".[viz]"` |

### Strength snapshot (optional)

Depth-1 HCE self-play is a **weak** baseline on the ACPL ladder (often near Elo floor ~400). Artifact example: `plots/PGN_and_JSON/hce_d1_gate_acpl.json`. Stronger HCE demo: depth **2** + qsearch (see below).

---

## Variants

```powershell
# Stronger / slower — depth 2 + capture quiescence (agent: NOTES/agents/hce-d2-qsearch.md)
py -3.12 scripts/record_engine_game.py --eval hce --depth 2 --max-plies 80 --headless `
  --output images/games/hce_d2_demo.gif

# Cap qsearch if d2 blows up
py -3.12 scripts/record_engine_game.py --eval hce --depth 2 --max-qsearch-depth 6 --max-plies 80 --headless `
  --output images/games/hce_d2_q6_demo.gif
```

Existing reel files: `images/games/hce_d1_game.gif`, `images/games/hce_d2_game.gif`.

---

## Related

| Path | Role |
| ---- | ---- |
| [demo-cfish-hybrid-gif.md](demo-cfish-hybrid-gif.md) | Cfish Hybrid NNUE + full SF-class search (contrast) |
| `scripts/run_engine.py --eval hce --depth 1` | Single-position best move |
| `scripts/eval_bot_acpl.py --eval hce --depth 1 --no-quiescence` | ACPL gate (needs Stockfish) |
