# Design options: §Benchmark Infrastructure

## LOCKED (2026-07-16)

```text
A2, B3, C2, D2+D4, E2+E3, F4, G2+G3+G4, H2+H4, I2+I3, J3, K2
```

Written into [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md) §**Benchmark Infrastructure** (supersedes old Bot Evaluation Tool Selection table).

| Pick | Meaning |
| ---- | ------- |
| A2→A3 | Strength + PC perf now; device later, same schema |
| B3 | ACPL for iteration; match Elo for gates |
| C2 | Stockfish **fixed depth** as ACPL judge |
| D2+D4 | Opponent ladder + sparse teacher@d1 checks |
| E2+E3 | Time/move **and** node-budget protocols |
| F4 | Unified PC/Wio metric JSON schema |
| G2+G3+G4 | Goldens, PC↔device parity, teacher correlation |
| H2+H4 | On-demand manifested runs + weekly/pre-release full gate |
| I2+I3 | JSON/PGN + generated dashboard plots |
| J3 | Dual gate: ACPL “not broken” + match ≥1700 claim |
| K2 | `bot_eval` library + thin scripts |

---

_Original options below (for history). Context: expands full benchmarking (strength + speed + parity + gates)._  
_Related: [ASSETS.md](ASSETS.md) · `scripts/eval_bot_acpl.py` · agent recipes in `NOTES/agents/`._

---

## What “benchmark infrastructure” must cover

SARDINE needs more than a one-shot Elo guess. A robust bench stack should answer:

1. **Is the bot stronger than last week?** (playing strength)
2. **Is search/eval fast enough for ~1 s/move on target hardware?** (throughput / latency)
3. **Do PC and device agree?** (parity — later, once C port exists)
4. **Did training labels / NNUE actually improve leaves?** (eval quality vs teacher)
5. **When do we ship vs iterate?** (gates vs blueprint ≥1700)

Design choices below fix *how* we measure those, not the NNUE architecture itself.

---

## Decision A — Scope of the benchmark suite

### A1: Strength-only
- **Description**: Only playing-strength metrics (ACPL and/or match Elo). No nodes/s or latency suite.
- **Pros**: Simple; matches current scripts.
- **Cons**: Misses Wio feasibility; strong-but-slow bots look fine until port.
- **Notes**: Too thin for “infrastructure.”

### A2: Strength + PC performance (Recommended)
- **Description**: Playing strength **and** PC benchmarks: nodes/s, evals/s, time-to-depth, optional TT hit stats. Device parity deferred until C port.
- **Pros**: Covers day-to-day iteration now; performance budgets inform search/NNUE width.
- **Cons**: PC numbers ≠ Wio; must re-measure on device later.
- **Notes**: Fits current Python engine + future C port.

### A3: Full stack (strength + PC + device + parity)
- **Description**: A2 plus Wio latency, `-O3`/`-Os`, encoder/NNUE parity tests, Serial UCI gate.
- **Pros**: Complete vs blueprint.
- **Cons**: Much of this is blocked on C port; writing it as “must have now” stalls progress.
- **Notes**: Good as **v1 end-state**; implement in phases.

### A4: Eval-quality only (teacher MSE / correlation)
- **Description**: Bench = val MSE / correlation vs teacher `expected_reward`; no games.
- **Pros**: Fast regression for training.
- **Cons**: MSE ≠ Elo; ignores search.
- **Notes**: Useful as a **sub-suite**, not the whole infrastructure.

---

## Decision B — Primary playing-strength method

### B1: ACPL heuristic only (current default)
- **Description**: Self-play PGN → Stockfish analyzes moves → ACPL → `Elo ≈ 2855 − 10×ACPL` (floor/cap as now).
- **Pros**: Already implemented (`eval_bot_acpl.py`); cheap; no pairing schedule.
- **Cons**: Not real match Elo; high variance; self-play style can distort ACPL; not comparable to Lichess/CCRL.
- **Notes**: Blueprint table already chose something like this (A1).

### B2: Match Elo only (cutechess / UCI round-robin)
- **Description**: Engine-vs-engine games with fixed time or depth; Elo from W/D/L (BayesElo / Ordo / cutechess).
- **Pros**: Standard; credible ≥1700 gate story.
- **Cons**: Needs stable UCI (or adapter); more games for tight CI; slower iteration.
- **Notes**: Blueprint already wants minimal UCI before formal gate.

### B3: Hybrid — ACPL for iteration, matches for gates (Recommended)
- **Description**: Day-to-day: ACPL (+ optional short match vs Sunfish/HCE). Milestone / Elo gate: fixed match protocol vs reference engines.
- **Pros**: Fast feedback + honest ship criterion.
- **Cons**: Two pipelines to maintain; must not confuse “ACPL Elo” with match Elo in reports.
- **Notes**: Label outputs clearly (`elo_acpl_heuristic` vs `elo_match`).

### B4: Human / Lichess bot rating
- **Description**: Deploy as Lichess bot or play rated humans for rating.
- **Pros**: Real-world signal.
- **Cons**: Slow, noisy, not reproducible CI; needs network & account.
- **Notes**: Optional v2 demo, not v1 infrastructure core.

---

## Decision C — ACPL judge configuration (if B1 or B3)

### C1: Stockfish fixed movetime (current)
- **Description**: e.g. `--sf-movetime-ms 100` per move analysis.
- **Pros**: Simple, already used.
- **Cons**: Strength of analysis depends on machine; less reproducible across PCs.
- **Notes**: Document SF version + movetime in every JSON.

### C2: Stockfish fixed depth (Recommended if keeping ACPL)
- **Description**: e.g. depth 12–16 for every move; same on all machines (slower but stabler).
- **Pros**: More comparable across hardware.
- **Cons**: Wall time scales with positions; still not “truth.”
- **Notes**: Pair with pinned SF binary path (`models/teacher/stockfish/…`).

### C3: Dual judge (SF + second engine)
- **Description**: Report ACPL under two judges; flag large disagreement.
- **Pros**: Detects judge noise.
- **Cons**: 2× cost; little gain early.
- **Notes**: Overkill until gate season.

---

## Decision D — Match opponents / ladder (if B2 or B3)

### D1: Weak ladder only (Sunfish, random, HCE)
- **Description**: Only engines clearly below target; prove “not broken.”
- **Pros**: Easy wins; good smoke.
- **Cons**: Cannot claim ≥1700.
- **Notes**: Matches blueprint early baseline + Sunfish calibration.

### D2: Ladder to gate (weak → mid → gate peers) (Recommended)
- **Description**: Fixed ladder, e.g. Sunfish → HCE@d2 → pilot NNUE@d1 → Stockfish level / limited-strength → (later) Dog-class if available. Gate = score vs a defined reference set under fixed TC.
- **Pros**: Interpretable progression; gate is a protocol, not a vibe.
- **Cons**: Need frozen opponent recipes (like `NOTES/agents/*.md`).
- **Notes**: Never move opponent strength silently between runs.

### D3: Self-play generations only
- **Description**: New net vs previous checkpoint only.
- **Pros**: Good for RL-style iteration.
- **Cons**: Drift; no absolute Elo.
- **Notes**: Add as **extra** track, not sole track.

### D4: Teacher-eval 1-ply as permanent opponent
- **Description**: Always include “Lc0 teacher @ depth 1” (or frozen teacher net) as a reference bot.
- **Pros**: Ties strength to label ceiling discussion (`ai-feed` teacher notes).
- **Cons**: Slow if teacher is Lc0; use only for sparse checkpoints.
- **Notes**: Valuable for “are we near teacher leaf quality?”

---

## Decision E — Search protocol under test (fair comparison)

### E1: Fixed depth only
- **Description**: All bots compared at depth 1 and/or 2 (current demos).
- **Pros**: Reproducible; no time variance.
- **Cons**: Ignores real ~1 s budget; favors cheap evals unfairly at same depth.
- **Notes**: Keep for **smoke** and agent cards.

### E2: Fixed time per move (Recommended for strength gates)
- **Description**: e.g. 100 ms or 1 s/move on PC; same TC for all engines in a match.
- **Pros**: Closer to product goal; mixes eval speed + search skill.
- **Cons**: PC-dependent unless capped nodes; need careful process isolation.
- **Notes**: Log CPU model in manifest.

### E3: Fixed node budget
- **Description**: e.g. 20k nodes/move (Urusov-class reference).
- **Pros**: Hardware-fairer than wall time; good for search-stack comparison.
- **Cons**: Needs accurate node counting; NNUE vs HCE cost differs in wall time.
- **Notes**: Excellent **secondary** metric alongside E2.

### E4: Recipe matrix (depth × qsearch × eval)
- **Description**: Publish a small grid: `{hce,d1,no-q}`, `{nnue,d1}`, `{nnue,d2,q}`, … as frozen agent files.
- **Pros**: Avoids apples-to-oranges; already started in `NOTES/agents/`.
- **Cons**: Many cells; pick a **default gate recipe** explicitly.
- **Notes**: Combine with E2: matrix for research, one recipe for the gate.

---

## Decision F — Performance / systems benchmarks

### F1: Ad-hoc timing prints
- **Description**: Occasional `time.perf_counter` around search.
- **Pros**: Zero infra.
- **Cons**: Not comparable; easy to lie to yourself.

### F2: Scripted PC microbench suite (Recommended now)
- **Description**: Fixed FEN set → report: evals/s (HCE, NNUE, teacher), nodes/s @ depth d, time-to-depth, game ply/s for self-play smoke.
- **Pros**: Cheap regression; informs W vs 128/256 and qsearch blowups (seen with HCE).
- **Cons**: Still PC-only.
- **Notes**: Store JSON under `plots/` or `bench/` with git-ignored large logs.

### F3: Device-first (Wio only)
- **Description**: Defer all perf until on-device.
- **Pros**: Measures what matters for ship.
- **Cons**: Blocks optimization loop on PC for months.
- **Notes**: Wrong primary choice today.

### F4: PC suite now + device suite later (same metrics schema) (Recommended end-state)
- **Description**: Same JSON schema (`metric`, `value`, `platform`, `commit`, `recipe`); fill `platform=wio` when ready.
- **Pros**: Continuous history across port.
- **Cons**: Slight schema design upfront.
- **Notes**: Best with A2→A3 evolution.

---

## Decision G — Parity & correctness gates

### G1: Unit tests only (current)
- **Description**: perft, encoder tests, tactical features, etc.
- **Pros**: Already strong (~100 tests).
- **Cons**: Does not catch PC↔device drift or silent eval scale bugs.

### G2: Golden FEN eval vectors (Recommended)
- **Description**: Freeze N FENs → expected encoder indices + NNUE fp32 score (+ later int8). Fail CI if drift.
- **Pros**: Catches training export and port bugs early.
- **Cons**: Must regenerate goldens when net changes (version goldens with checkpoint id).
- **Notes**: Separate goldens per checkpoint.

### G3: PC↔device bitwise / tolerance parity
- **Description**: Same as G2 but device must match PC within atol/rtol (int8 may be exact if identical math).
- **Pros**: Required for trustworthy on-device Elo claims.
- **Cons**: After C port only.
- **Notes**: Part of F port checklist.

### G4: Teacher correlation suite
- **Description**: On labeled val set, report MSE / Spearman of student vs `expected_reward`.
- **Pros**: Direct train-quality metric; uniform with data policy.
- **Cons**: Not playing strength.
- **Notes**: Run after every production train.

---

## Decision H — Automation & frequency

### H1: Manual only (current blueprint D1/E1 style)
- **Description**: Run scripts by hand after big changes.
- **Pros**: No CI cost.
- **Cons**: Drift; “we forgot to bench.”

### H2: On-demand CLI + saved manifests (Recommended)
- **Description**: One entry script or documented commands; every run writes JSON/PGN under `plots/` or `bench/runs/<timestamp>/` with config hash.
- **Pros**: Reproducible without full CI.
- **Cons**: Discipline required.
- **Notes**: Extends what you already do with `*_gate_acpl.json`.

### H3: CI on every PR (unit + microbench smoke)
- **Description**: pytest + tiny ACPL (1 game) or evals/s smoke in CI.
- **Pros**: Guards regressions.
- **Cons**: SF/Lc0 in CI is heavy on Windows runners; may skip heavy engines in CI.
- **Notes**: Keep heavy SF ACPL **nightly / manual**.

### H4: Scheduled full gate (weekly / pre-release)
- **Description**: Longer match + ACPL multi-game + microbench.
- **Pros**: True milestone signal.
- **Cons**: Needs stable machine or cloud.
- **Notes**: Pair with B3.

---

## Decision I — Artifacts & reporting

### I1: Print to terminal only
- **Pros**: Fast. **Cons**: Lost history.

### I2: JSON + PGN per run (Recommended)
- **Description**: Machine-readable metrics + games; optional markdown table in daily notes.
- **Pros**: Diffable; already partial (`plots/*_gate_acpl.json`).
- **Cons**: Need naming convention.
- **Notes**: Include: `commit`, `recipe`, `teacher/net`, `sf_version`, `n_games`, `metrics`.

### I3: Dashboard (HTML/plots always generated)
- **Description**: Auto plots for Elo/ACPL over time, nodes/s over time.
- **Pros**: Nice for report/ICTP.
- **Cons**: Extra maintenance.
- **Notes**: Optional; generate from I2 JSON later.

### I4: Single “leaderboard.md” checked into git
- **Description**: Human-updated or script-updated table of best configs.
- **Pros**: Visible.
- **Cons**: Merge conflicts; keep generated section or append-only.
- **Notes**: Good for internship report.

---

## Decision J — Elo gate definition (≥1700)

### J1: ACPL-mapped Elo ≥ 1700
- **Description**: Ship if heuristic Elo from ACPL crosses 1700.
- **Pros**: Easy with current tools.
- **Cons**: Heuristic can lie (you already saw huge σ / self-play quirks).
- **Notes**: Weak sole criterion.

### J2: Match Elo ≥ 1700 vs defined references (Recommended for “real” gate)
- **Description**: Protocol: N games, TC, opponents list, sprt or fixed games; BayesElo ≥ 1700 with CI.
- **Pros**: Defensible.
- **Cons**: Needs UCI + time; more work.
- **Notes**: Aligns with cutechess-cli vision in blueprint.

### J3: Dual gate — ACPL smoke + match ship
- **Description**: Block train merge if ACPL collapses; **ship** only on match protocol.
- **Pros**: Best of B3.
- **Cons**: Two thresholds to document.
- **Notes**: Recommended if B3 chosen.

### J4: Device gate only
- **Description**: ≥1700 only counts on Wio under 1 s/move.
- **Pros**: True product goal.
- **Cons**: Late signal; need PC proxy gates earlier.
- **Notes**: Final bar; keep PC proxies before port.

---

## Decision K — Repo layout for bench code

### K1: Keep scripts only (`scripts/eval_*.py`)
- **Pros**: Minimal. **Cons**: Grows messy.

### K2: Library + scripts (Recommended)
- **Description**: `src/tinymlinternship/bot_eval/` (exists) expands; scripts stay thin CLIs; optional `bench/` for run outputs (gitignored bulk).
- **Pros**: Testable; reusable.
- **Cons**: Small refactor.

### K3: External tool-only (cutechess + engines)
- **Pros**: Standard. **Cons**: Weak for ACPL/microbench/encoder parity; still need Python glue.

---

## Recommended configuration (starting point)

| Decision | Pick | One-line why |
| -------- | ---- | ------------ |
| **A** Scope | **A2** → evolve to **A3** | Strength + PC perf now; device later |
| **B** Strength method | **B3** | ACPL iterate; matches for real Elo |
| **C** ACPL judge | **C2** | Fixed SF depth more portable than movetime |
| **D** Opponents | **D2** (+ optional **D4** sparse) | Ladder + teacher ceiling checks |
| **E** Search protocol | **E4** recipes + **E2** for gates | Fair product-like TC; fixed depth for smokes |
| **F** Perf | **F2** / **F4** schema | Scripted PC suite; same JSON later on Wio |
| **G** Parity | **G1+G2** now; **G3** at port; **G4** after train | Goldens + teacher MSE |
| **H** Automation | **H2** (+ **H4** pre-release) | Manifested runs without heavy CI |
| **I** Artifacts | **I2** (+ **I4** summary) | JSON/PGN history |
| **J** Gate | **J3** | ACPL don’t-ship-broken; match for ≥1700 claim |
| **K** Layout | **K2** | bot_eval lib + scripts |

---

## How to reply

Send a line like:

```text
A2, B3, C2, D2, E4, F2, G2, H2, I2, J3, K2
```

Optional notes per letter (e.g. `E2 for gates only, E1 for daily smoke`).  
After you choose, the next step is to draft **§Benchmark Infrastructure** in `NOTES/SARDINE Engine Blueprint.md` from those decisions (not before).

---

## Explicitly out of scope for this decision set

- Changing NNUE architecture or training loss  
- Replacing the **uniform `expected_reward`** data policy  
- Full SPSA search tuning (uses bench outputs later, but is not the bench design itself)
