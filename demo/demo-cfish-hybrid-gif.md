# Demo — Cfish self-play GIF (depth 5)

Generate a **Cfish Hybrid** vs **Cfish Hybrid** game at **depth 5**, save the PGN, and export an animated GIF under `images/games/` (same style as the other engine demos).

Run all commands from the **repo root**.

---

## Engine used in this demo

This demo does **not** use the Python SARDINE student (`engine/search.py` + HCE/pilot NNUE). It uses the **Cfish** binary already in the tree — a pure-**C** port of **Stockfish**-class search with a stock NNUE value head.

| Spec                | This demo                                                                 |
| ------------------- | ------------------------------------------------------------------------- |
| **Engine**          | **Cfish** — Stockfish algorithms in C (bitboards, UCI)                    |
| **Binary**          | `src/cfish/cfish.exe` (Windows) / `src/cfish/cfish`                       |


### Value function (evaluation)

| Spec                   | Value                                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| **Mode**               | **Hybrid NNUE** (UCI `Use NNUE` = `Hybrid`)                                                          |
| **Weights**            | Stock net `src/cfish/nn-62ef826d1a6d.nnue` (`DefaultEvalFile` in `evaluate.h`)                       |
| **Architecture class** | Desktop **HalfKP-style** NNUE (SIMD-capable kernels in Cfish), **not** the SARDINE 844-dim student   |
| **Output scale**       | Centipawn-style score for search (UCI `score cp` / mate), not SARDINE’s `expected_reward ∈ [-1, +1]` |
| **Not used here**      | Python `eval_hce` / `eval_nnue` / pilot `pilot_W128_844`                                             |
**What Hybrid means**: Leaf evaluation blends / uses the **NNUE** network together with Cfish’s **classical (HCE)** terms — not Pure-NNUE-only and not classical-only.

On load you should see a log line like:  
`Hybrid NNUE evaluation using nn-62ef826d1a6d.nnue enabled.`

### Tree search

| Spec | Value |
| ---- | ----- |
| **Framework** | Full Cfish/Stockfish-class **alpha-beta** search (PVS), not the thin Python αβ in `engine/search.py` |
| **Typical extras** (engine defaults) | Null-move pruning, late-move reductions (LMR), futility-style pruning, **quiescence** on captures, killers / history-style ordering, etc. |
| **Board rep** | **Bitboards** (64-bit masks) |
| **Transposition table** | Yes — size set by UCI `Hash` (demo default **64 MB**) |
| **Depth control** | Fixed **`go depth 5`** every move (UCI limit depth = **5 plies** of main search, plus internal extensions / qsearch as the engine implements them) |
| **Time control** | None in this recipe — pure depth limit (no `movetime` / clock) |

### Runtime / self-play config (as set by the scripts)

| Spec | Value |
| ---- | ----- |
| **Sides** | Same Cfish process for **White and Black** (identical policy self-play) |
| **Threads** | `1` |
| **Hash** | `64` MB (CLI `--hash-mb`) |
| **Large pages** | `false` (avoids Windows TT allocation noise) |
| **Contempt** | Mild per-game diversity (`24 + 12×game_index`) in multi-game runs |
| **Game length cap** | `--max-plies 80` half-moves |
| **Working directory** | **Must** be `src/cfish/` so `EvalFile` resolves the `.nnue` (scripts set this) |

### Strength snapshot (optional context)

Same Hybrid depth-5 policy was gated with Stockfish ACPL (not required to build the GIF):

| Metric | Approx. |
| ------ | ------- |
| **ACPL** (3 games, SF 100 ms/move) | **~42** |
| **Elo heuristic** (`≈ 2855 − 10×ACPL`, floor 400) | **~2435** |
| Artifact | `images/plots/PGN_and_JSON/cfish_hybrid_d5_gate_acpl.json` |

More detail: [NOTES/Cfish.md](../NOTES/Cfish.md) · [PROJECT.md](../PROJECT.md) (Cfish / First NNUE) · [ASSETS.md](../ASSETS.md).

---

## 1. Self-play → PGN

Play **one** Hybrid game at **depth 5** (max 80 half-moves) and write PGN under `images/games/` (and a copy under `images/plots/` if you leave the default secondary path):

```powershell
py -3.12 scripts/cfish_selfplay_pgn.py `
  --games 1 `
  --depth 5 `
  --max-plies 80 `
  --hash-mb 64 `
  --output images/games/cfish_hybrid_d5_demo.pgn `
  --also-games-dir images/games
```

What you get:

- `images/games/cfish_hybrid_d5_demo.pgn` — main output (`--output`)
- `images/games/Cfish-hybrid-d5_vs_Cfish-hybrid-d5_YYYY-MM-DD.pgn` — dated copy (`--also-games-dir`)

Defaults if you omit flags: 3 games, depth 5, output `images/plots/PGN_and_JSON/cfish_hybrid_d5_gate.pgn`.

For a multi-game gate batch (no GIF):

```powershell
py -3.12 scripts/cfish_selfplay_pgn.py --games 3 --depth 5 --max-plies 80
```

---

## 2. PGN → GIF (`images/games/`)

`cfish_selfplay_pgn.py` writes **PGN only**. Export a GIF with the same helper used by other demos (`gifpgn` via `write_game_gif`):

```powershell
py -3.12 -c @"
from pathlib import Path
import chess.pgn
from tinymlinternship.visualization import write_game_gif

pgn_path = Path('images/games/cfish_hybrid_d5_demo.pgn')
gif_path = Path('images/games/cfish_hybrid_d5_demo.gif')

with pgn_path.open(encoding='utf-8') as f:
    game = chess.pgn.read_game(f)
if game is None:
    raise SystemExit(f'No game in {pgn_path}')

write_game_gif(game, gif_path, frame_ms=450)
print(f'GIF written: {gif_path} ({gif_path.stat().st_size:,} bytes)')
"@
```

Open `images/games/cfish_hybrid_d5_demo.gif` in a browser or image viewer.

**Tips:**

- First game only is used if the PGN file has several games (`read_game` once).
- Slower / faster animation: change `frame_ms` (e.g. `300` or `600`).
- Existing dated PGN example from the 2026-08-04 gate:  
  `images/games/Cfish-hybrid-d5_vs_Cfish-hybrid-d5_2026-08-04.pgn`  
  Point `pgn_path` / `gif_path` at that stem to re-render without replaying Cfish.

---

## One-shot recipe (copy-paste)

```powershell
# from repo root
pip install -e ".[viz]"

py -3.12 scripts/cfish_selfplay_pgn.py --games 1 --depth 5 --max-plies 80 `
  --output images/games/cfish_hybrid_d5_demo.pgn --also-games-dir images/games

py -3.12 -c "from pathlib import Path; import chess.pgn; from tinymlinternship.visualization import write_game_gif; p=Path('images/games/cfish_hybrid_d5_demo.pgn'); g=chess.pgn.read_game(p.open(encoding='utf-8')); write_game_gif(g, p.with_suffix('.gif'), frame_ms=450); print('OK', p.with_suffix('.gif'))"
```

---
