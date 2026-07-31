# Structural integrity assessment (post-hygiene)

_Snapshot for agents. Date context: 2026-07-31. Not a full product roadmap — layout & coherence only._

## Verdict

| Dimension              | Grade  | Note                                                                        |
| ---------------------- | ------ | --------------------------------------------------------------------------- |
| **Overall structure**  | **B+** | Clear SARDINE spine; recent hygiene removed the worst duplicates            |
| Python package layout  | **A−** | Single installable tree under `src/tinymlinternship/`                       |
| Data layout            | **A−** | Matches ASSETS mental model (`raw` → `processed` → `labeled`)               |
| Docs / agent maps      | **B−** | Several policies still lag hygiene (daily notes, Cfish path, legacy ghosts) |
| Binary / build hygiene | **C+** | Cfish build products checked in; large local data gitignored correctly      |
| Duplicate trees        | **A**  | Root `Cfish/` gone; single engine tree at `src/cfish/`                      |

**Bottom line:** The repo is **structurally sound for PC-side SARDINE work**. Integrity gaps are mostly **doc drift**, **tracked C build artifacts**, and a few **stale path references** — not a broken package layout.

---

## Canonical layout (what “good” looks like)

```text
TinyMLInternship/
├── PROJECT.md, Goal.md, TODOs.md, README.md, ASSETS.md, ai-feed.md
├── YYYY-MM-DD.md              ← active daily note (root only)
├── DAILY-NOTES/YYYY-MM/       ← archived daily notes
├── run-cfish.bat
├── src/
│   ├── tinymlinternship/      ← Python package (encoder, engine, data, nnue, …)
│   └── cfish/                 ← external C engine (baseline), not part of py package
├── scripts/                   ← CLIs
├── tests/
├── data/{raw,processed,excel} ← local datasets (mostly gitignored)
├── models/{checkpoints,teacher}
├── NOTES/, plots/, images/, presentations/
├── AI-SKILLS/                 ← project agent skills (gitignored)
└── PRIVATE/                   ← personal (gitignored)
```

| Layer | Path | Integrity |
| ----- | ---- | --------- |
| Spec authority | `NOTES/SARDINE Engine Blueprint.md` | Present |
| Status dashboard | `PROJECT.md` | Present; some fields stale (see below) |
| Progress checklist | `TODOs.md` | Present |
| Path/label contract | `ASSETS.md` | Present; aligns with `data/` |
| Python API | `src/tinymlinternship/` | Coherent modules + tests |
| C baseline | `src/cfish/` + `run-cfish.bat` | Single location after hygiene |
| Pipeline scripts | `scripts/` (~30) | Named by role; production vs smoke still mixed in one folder |

---

## Hygiene wins (verified)

| Item | Status |
| ---- | ------ |
| Root `Cfish/` duplicate clone | **Removed** |
| Active Cfish tree | **`src/cfish/` only** (exe + NNUE + sources) |
| `run-cfish.bat` | Points at `src\cfish` |
| Empty Python packages (`core/`, `evaluation/`, `datasets/`) | **Gone** (reassessment I1) |
| `legacy/pre-sardine/` | **Gone** |
| Active daily note | **`2026-07-31.md` at root** (matches intended daily-notes policy) |
| Archived notes | Under `DAILY-NOTES/2026-06/`, `DAILY-NOTES/2026-07/` |
| Large data | `data/raw/`, `data/processed/` **gitignored** |
| Settings paths | `DATA_DIR`, Lc0/ChessBench raw+processed, teacher binary, pilot checkpoint — **exist on disk** |

---

## Structural strengths

1. **Separation of concerns:** features / engine / data / nnue / bot_eval / visualization are real packages, not a scripts soup.
2. **Data pipeline topology is correct:**  
   `raw/{lc0,chessbench,…}` → `processed/{lc0,lichess,chessbench}` → `processed/labeled/{train,val}.parquet`.
3. **Tests mirror domains:** parser, preprocess, schema, engine, features, ACPL — good structural coverage map.
4. **External teachers isolated** under `models/teacher/` (Lc0, networks; HF weights not shipped).
5. **Agent surface area is intentional:** root trackers + `NOTES/` + `AI-SKILLS/` + `ai-feed.md`.

---

## Integrity issues (ordered by impact)

### P1 — Doc / policy drift

| Location | Problem |
| -------- | ------- |
| `PROJECT.md` | Still says daily notes live **only** under `DAILY-NOTES/` and “write under `DAILY-NOTES/YYYY-MM/` each session”. **New policy:** active note at **root** `YYYY-MM-DD.md`, archive under `DAILY-NOTES/YYYY-MM/`. |
| `AI-SKILLS/repo-map/SKILL.md` | Still lists `legacy/pre-sardine/` as “do not use” (tree is **deleted**); “Project Report” path outdated vs `NOTES/archive/`. |
| `NOTES/Cfish.md` | Build path still `cd Cfish/src` — should be `src/cfish`. |
| `AI-SKILLS/` | Project copy has **no** `daily-notes` skill folder (may live only under user `~/.grok/skills/daily-notes`). Repo agents that only read `AI-SKILLS/` miss the archive policy. |

### P1 — Cfish as a “foreign” tree inside `src/`

| Observation | Risk |
| ----------- | ---- |
| `src/cfish/` is **not** a Python package; sits beside `tinymlinternship/` | Fine if intentional; confusing for “everything under src is installable” |
| **24× `.o`**, `cfish.exe`, **~20 MB** `.nnue` live in-tree | Repo bloat / noisy diffs / platform-specific binaries if committed |
| `scripts/cfish.py` uses `./cfish` | **Broken path** relative to cwd; should be `src/cfish/cfish.exe` (or env) |
| Upstream `LICENSE` / top-level README of Cfish clone | Dropped when flattening to `src/cfish` — legal/attribution should stay in `NOTES/Cfish.md` or a short `src/cfish/README` |

### P2 — Clutter & oddities

| Path | Note |
| ---- | ---- |
| `terminals/` | Empty directory still present (reassessment wanted it cleared; dir shell remains) |
| `data/processed/lc0.md` | Lone markdown next to `lc0/` folder — ambiguous (doc vs data) |
| `models/teacher/sunfish/` | Large third-party tree; study/baseline only — OK but heavy |
| `images/games/` | Many dated GIF/PGN demos; intentional archive, not structural rot |
| `scripts/` | Production (`label_positions`, `merge_training_sets`, …) mixed with smoke (`study_chessbench`, HF record, presentation builders) — no subfolders |

### P2 — `.gitignore` sharpness

| Pattern | Effect |
| ------- | ------ |
| `*.h` | Global ignore of headers — risky if SARDINE C-port adds headers; Cfish headers only stay if force-added |
| `*.pgn`, `*.csv` | Correct for data dumps; also hides intentional small smoke PGN unless force-added |
| Root `YYYY-MM-DD.md` | **Not** gitignored — active daily notes may be committed accidentally; archives under `DAILY-NOTES/` are ignored |
| `src/cfish` build products | **Not** ignored — `.o` / exe / nnue can enter git (several already tracked) |

### P3 — Naming / case

On Windows, `src/cfish` and `src/Cfish` resolve to the **same** directory. Prefer always **`src/cfish`** (matches `run-cfish.bat`) to avoid case-only path bugs on Linux CI later.

---

## Dependency graph (structural)

```text
ASSETS.md ──schema──► tinymlinternship.data
        │                    │
        ▼                    ▼
 scripts/*  ──parquet──►  data/processed/labeled
        │                    │
        ▼                    ▼
 train_nnue.py ──► nnue.dataset ──► features.encoder (844)
        │
        ▼
 engine (hce|nnue|lc0) ◄── bot_eval (Stockfish ACPL, PATH only)
        │
        ▼
 scripts/record_*, run_engine.py

Cfish baseline (parallel, not in graph above):
  run-cfish.bat → src/cfish/cfish.exe + nn-*.nnue
```

No circular package imports expected at the layout level; Cfish is a **sibling baseline**, not imported by the Python package.

---

## Structural checklist (agents)

- [x] One Python package root: `src/tinymlinternship/`
- [x] One Cfish location: `src/cfish/`
- [x] No root `Cfish/` clone
- [x] No `legacy/pre-sardine/`
- [x] Data dirs present for configured paths
- [ ] Docs agree on **daily note root vs archive**
- [ ] Cfish build artifacts ignored or documented as vendored
- [ ] `scripts/cfish.py` path fixed or deleted
- [ ] `NOTES/Cfish.md` paths updated to `src/cfish`
- [ ] `repo-map` skill drops dead `legacy/` pointer
- [ ] Optional: project-local `AI-SKILLS/daily-notes` synced with user skill

---

## Recommended next hygiene (optional, not blocking)

1. **Align docs:** `PROJECT.md` daily-note lines; `NOTES/Cfish.md`; `AI-SKILLS/repo-map`.
2. **Gitignore Cfish build:** `src/cfish/*.o`, `src/cfish/cfish.exe` (keep sources; keep or LFS the `.nnue` intentionally).
3. **Fix or remove** `scripts/cfish.py` (broken `./cfish`).
4. **Delete empty** `terminals/` or stop recreating it.
5. **Move** `data/processed/lc0.md` → `NOTES/` if it is documentation.
6. **Subfolder scripts** only if noise grows (`scripts/data/`, `scripts/eval/`, `scripts/viz/`) — not required now.

---

## Agent reading order (structure-first)

1. This file (`ai-feed.md`) — integrity snapshot  
2. `PROJECT.md` — goals / % (trust TODOs more for checkboxes)  
3. `ASSETS.md` — data contract  
4. `src/tinymlinternship/` + `scripts/` — implementation  
5. Root `YYYY-MM-DD.md` — today’s session  
6. `NOTES/SARDINE Engine Blueprint.md` — architecture authority  

#structure #hygiene
