# Project: SARDINE

**SARDINE** — *Small Artificial RAM-restricted Deep Intelligent Neural Engine*

Chess engine for the **Wio Terminal**: neural evaluation + alpha-beta search, maximizing **Elo per byte** under **192 KB RAM** / **~500 KB flash**. No cloud, no GPU. Target: playable bot (ideally on *Lichess*).

| Doc                                                                        | Role                                                             |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **[PROJECT.md](PROJECT.md)** (this file)                                   | **Progress source of truth** (decision A1) + Cfish inventory     |
| [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md) | Architecture / device design (decision B1)                       |
| [ASSETS.md](ASSETS.md)                                                     | Labels + data contract + paths (C1, E1, H1)                      |
| [ai-feed.md](ai-feed.md)                                                   | Locked “which file to follow” table                              |
| [TODOs.md](TODOs.md)                                                       | Eng micro-gates **mapped to §Progress Overview** (not product status) |
| [Goal.md](Goal.md)                                                         | Short mission statement                                          |
| [NOTES/Thesis.md](NOTES/Thesis.md)                                         | Later research: task vectors / optimal bucketing (J1)            |

Inspiration from [Kaggle Challenge](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/linrock-my-solution-cfish-nnue-data-1st), [repo](https://github.com/linrock/minifish)

---

## Progress Overview


_Progress vs repo as of ~2026-08-03. **This checklist is the sole product progress map (A1).** Architecture: [blueprint](NOTES/SARDINE%20Engine%20Blueprint.md) (B1). Labels/data: [ASSETS.md](ASSETS.md) (C1/E1). Doc authority: [ai-feed.md](ai-feed.md). Italics = caveats; if a title conflicts with italics/ASSETS/blueprint design, prefer the latter (K1)._

### Run Cfish

- [x] Cfish smoke test 
	- binary: `src/cfish/cfish.exe` · launcher: `run-cfish.bat` · notes: `NOTES/Cfish.md`
	- UCI `uciok` / `readyok` verified (2026-07-31)
	- Formal recipe + nps archived (2026-08-04): cwd **must** be `src/cfish/`; Hybrid NNUE loads `nn-62ef826d1a6d.nnue`; `go depth 5` → `bestmove e2e4`, nps ~**224k** (depth 12 ~**1.8M** nps). See daily note `2026-08-04.md` §Execution log.
	- `scripts/cfish.py` resolves `src/cfish/cfish.exe` + correct cwd (no longer `./cfish`).
    
- [x] Selecting evaluation criterion
	- **Primary judge: Stockfish** (strong external engine) — game review via average centipawn loss (ACPL) and heuristic Elo (`Elo ≈ 2855 − 10×ACPL`, floor 400)
	- CLI: `scripts/eval_bot_acpl.py` · per-side from PGN: `scripts/eval_game_elo.py` · lib: `src/tinymlinternship/bot_eval/`
	- Stockfish binary: PATH / `STOCKFISH_PATH` / `--stockfish` (optional local under `tools/stockfish/`; not shipped in git)
	- **Other criteria used in this project:**
		- **nps / nodes** — search cost model (UCI `info nps` on Cfish; microbench still open for Python student)
		- **Self-play ladder** — same judge protocol across policies (HCE, NNUE, random floor, Cfish Hybrid, teachers)
		- **val MSE / MAE** — train-time fit to teacher `expected_reward` (not a playing-strength metric alone)
		- **Match Elo** (blueprint ship gate) — head-to-head under a frozen protocol; preferred for the final ≥1700 claim over ACPL alone
	- *ACPL is the day-to-day gate; match Elo is the product ship gate (see blueprint / On the Hardware).*
    
- [x] Evaluate the base (HCE) model with Stockfish (game review)
	- **Python HCE** baseline (not Cfish classical UCI): `eval_hce.py` + αβ `search.py`
	- **10-game gate (2026-08-05):** depth **2**, qsearch **off**, max 80 plies; SF judge 100 ms/move
	- Combined: **ACPL 20.9** (σ=16.5) · Elo heuristic **2646** (2571–2721) · 440 moves · all 10 games ½–½ @ 44 plies (deterministic self-play line)
	- **Mean per-game Elo:** **~2646** (range across games **2631–2667**)
	- Top blunders (combined CPL): `…d6` / `Be3` ~65 cp (opening inaccuracy, not tactical collapse)
	- **nps microbench** (startpos, d2, qsearch cap 6, 50 searches): ~**28k nps** (Python; not Cfish-class)
	- Artifacts: `plots/PGN_and_JSON/hce_d2_q6_10game_gate.pgn`, `hce_d2_q6_10game_gate_acpl.json`
	- Prior multi-game d2 no-q: `hce_d2_gate_acpl.json` (16 games, ACPL **24.5** / Elo **~2610**, 2026-07-10) — consistent order of magnitude
	- CLI: `scripts/eval_bot_acpl.py --eval hce --depth 2 --games 10 --sf-movetime-ms 100` · lib: `bot_eval/`
	- *Cfish pure-classical UCI baseline remains optional/distinct if ever needed; this ticks the PC HCE strength gate.*

### First NNUE

- [x] Download a NNUE 
	- `src/cfish/nn-62ef826d1a6d.nnue` (also `make net` in `src/cfish/`)
	- [URL](https://tests.stockfishchess.org/api/nn/nn-62ef826d1a6d.nnue)
	- *This is the **stock Stockfish-family** net shipped with Cfish — not a SARDINE-trained student.*
	
- [x] Replace the value function that Cfish uses with the new NNUE 
	- Default: `DefaultEvalFile` in `src/cfish/evaluate.h` · load/INCBIN: `src/cfish/nnue.c` · UCI option: `src/cfish/ucioption.c` (`EvalFile`, `Use NNUE`)
	- *No extra patch required for the stock net. Wiring a **custom SARDINE** `.nnue` into Cfish is a different (later) step — device path is still “own C port,” not necessarily Cfish-hosted student weights.*
	  
- [x] smoke test Hybrid NNUE log
	- Hybrid NNUE evaluation using the new NNUE (Cfish `Use NNUE` hybrid with EvalFile present)
	- *UCI smoke with net on disk is done; a dedicated hybrid-only log / nps artifact is not archived yet.*
	  
- [x] Evaluation with Stockfish
	- Cfish Hybrid self-play + Stockfish ACPL (2026-08-04): `scripts/cfish_selfplay_pgn.py` + `scripts/eval_bot_acpl.py --pgn …`
	- Artifacts: `plots/PGN_and_JSON/cfish_hybrid_d5_gate.pgn`, `cfish_hybrid_d5_gate_acpl.json` · copy `images/games/Cfish-hybrid-d5_vs_Cfish-hybrid-d5_2026-08-04.pgn`
	- Protocol: 3 games, depth 5 Hybrid, max 80 plies; SF judge movetime 100 ms
	- Combined: **ACPL 42.0** (σ=73) · Elo heuristic **2435** (2327–2542) · 178 moves · play nps mean ~427k
	- *Judge = Stockfish on PATH / `STOCKFISH_PATH` (local download used under `tools/stockfish/` if not on PATH). ACPL gates for HCE / pilot NNUE / Sunfish / Lc0 already under `plots/PGN_and_JSON/`.*

### Dataset

- [x] Download the raw data 
	- mainly board positions: **Lichess primary** (production) + **Lc0** supplement (E1 / ASSETS)
	- Lc0: `data/raw/lc0/` (`tars/`, `chunks/`, `manifest.json`) · `scripts/download_lc0.py`
	- Lichess smoke PGN only so far: `data/raw/lichess_smoke50.pgn` — full monthly dump `data/raw/lichess/` **not** downloaded
	- Kaggle: `data/raw/games.csv` · `scripts/download_data.py` — **smoke/stats only**, not NNUE train
	- ChessBench bags: `data/raw/chessbench/test/` · `scripts/download_chessbench.py` — encoder smoke only
	  
- [ ] remove duplicate positions 
	- but keep track of the multiplicity, so we may use it later
	- *No dedicated dedup + multiplicity pass in the data pipeline yet (would live under `src/tinymlinternship/data/` + a `scripts/` CLI).*
	  
- [x] add teacher evaluations to each board state
	- list of board–eval pairs: **Lc0** WDL → **`expected_reward`** White POV ∈ \([-1,+1]\) (C1 / ASSETS)
	- label CLI: `scripts/label_positions.py`
	- teacher binary: `models/teacher/lc0/lc0.exe` · net: `models/teacher/lc0/791556.pb.gz` · manifest: `models/teacher/manifest.json`
	- install/smoke: `scripts/download_teacher.py`, `scripts/smoke_test_teacher.py`
	- mini labeled blocks: `data/processed/labeled/lichess_labeled.parquet`, `data/processed/labeled/lc0_labeled.parquet`
	- pre-label FENs: `data/processed/lichess/positions.parquet`, `data/processed/lc0/positions.parquet`
	- extract: `scripts/lichess_pgn_to_fen.py`, `scripts/prepare_lc0_dataset.py` · schema: `src/tinymlinternship/data/schema.py`
	- *Stockfish is **not** the train teacher — it is the **ACPL judge** only (D1, I1: PATH / `STOCKFISH_PATH`).*
	  
- [x] Clean the data into a single, uniform dataset
	- list of $(s, v)$ pairs
	- merge CLI: `scripts/merge_training_sets.py`
	- product: `data/processed/labeled/train.parquet`, `data/processed/labeled/val.parquet`, `data/processed/labeled/manifest.json` (~5.3k / 214 rows, seed 42)
	- *Uniform on **`expected_reward` only**. Scale is **mini/smoke**, not production \(10^5\)–\(10^6+\). Policy: [ASSETS.md](ASSETS.md).*
    

### Train the Network

- [x] Train small NNUE (pilot / smoke)
	- **Production shape (F3):** dual-POV L1 `844 → W` + concat `2W` + **single** head `2W → 1` (no multi-expert routing until §D)
	- train CLI: `scripts/train_nnue.py` (`--architecture single_head` **default**) · model: `SingleHeadNNUE` / legacy `BucketedNNUE` in `src/tinymlinternship/nnue/model.py` · loader: `nnue/dataset.py` · eval load: `engine/eval_nnue.py`
	- **F3 path + mini train (2026-08-05):** `models/checkpoints/nnue/single_W128_mini_ep30/` on `data/processed/labeled/{train,val}.parquet` (5306 / 214 rows; best val_mse **0.196** @ ep6; 30 ep run)
	- *Legacy pilots used **8** expert heads (`pilot_W128_844`, `smoke_prod_W128_844`) — experimental only; still the stronger d1 ladder point until data scale improves.*
	- ChessBench pilot data: `data/processed/chessbench/splits/{train,val}.parquet` · `scripts/prepare_chessbench_dataset.py` (smoke wiring only)
	- *Full-volume single-head train, nnue-pytorch adapt, gradual L1 prune, PTQ export still open.*
      
- [ ] Thesis idea: multi-expert / task-vector bucketing (after Elo path — J1)
	- training by bucketing states via task vectors or ablation-chosen partitions
	- *Not on critical path. Router experiments: `src/tinymlinternship/features/bucket.py` (legacy 8-way). See [NOTES/Thesis.md](NOTES/Thesis.md).*
      
- [x] Evaluation with Stockfish (ACPL judge — D1)
	- 10 quick self-play games + worst moves / top CPL
	- bench the nps
	- CLI: `scripts/eval_bot_acpl.py` · lib: `src/tinymlinternship/bot_eval/acpl.py`
	- self-play: `scripts/record_engine_game.py --eval nnue` · NNUE hook: `src/tinymlinternship/engine/eval_nnue.py`
	- artifacts: `plots/PGN_and_JSON/nnue_d1_gate.pgn`, `plots/PGN_and_JSON/nnue_d1_gate_acpl.json` (also `nnue_d2_*`)
	- **F3 single-head mini gate (2026-08-05):** `single_W128_mini_d1_gate*` — ACPL **~583** / Elo floor **400** (worse than random ~276; pilot multi-head d1 ~139 remains reference student)
	- demos: `images/nnue_d1_game.gif`, `images/nnue_d2_game.gif` (and related under `images/games/`)
	- *Playing-strength path **not** closed by mini single-head. **nps** microbench still open. Known issue: NNUE d2 ACPL collapse vs d1 (pilot).*

### On the Hardware

- [ ] Wio Terminal smoke test 
	- *Legacy pre-SARDINE Wio sketches removed (2026-07-22). Device eng: [TODOs.md](TODOs.md) §5 + §8 — not started. No active device tree path yet.*
      
- [ ] Evaluation with Stockfish
	- 30 self-play games
	- bench the nps
	- *Same judge stack as PC once UCI/Serial works: `scripts/eval_bot_acpl.py` + Stockfish on host; device metrics under future `bench/runs/` (see blueprint §Benchmark Infrastructure).*
      
- [ ] Connect the Wio to Lichess and play!

## For the Thesis

- [ ] Compare the new and other techniques 
	- bucketing through task vectors vs clustering directly through embeddings
	- *After a working base NNUE + Elo path; see [NOTES/Thesis.md](NOTES/Thesis.md) and blueprint §Later: optimal feature combinations & task vectors.*

## Stretch Goals

- [x] Piece-count distribution study
	- script: `scripts/plot_piece_count_distribution.py`
	- data: `data/excel/piece_count_distribution.xlsx`, `data/excel/piece_count_distribution_10k.xlsx`
	- plot: `plots/piece_count_distribution.png`, `plots/piece_count_distribution_10k.png`
       
- [x] Feature encoder (716 base + tactical → 844)
	- `src/tinymlinternship/features/index_map.py`, `encoder.py`, `mirror.py`, `tactical.py`, `bucket.py`
	- tests: `tests/test_features.py`, `tests/test_tactical.py`
	- *Detail: [TODOs.md](TODOs.md) §7a — critical path for student, not optional stretch work.*
    
- [x] Tweak the base tree search (alpha-beta + quiescence)
	- `src/tinymlinternship/engine/search.py` (PC v0.3) · `tests/test_engine.py`, `tests/test_perft.py`
	- *Remaining search stack: [TODOs.md](TODOs.md) §7b–7c (TT, nps, futility/LMR/null-move, killers).*
    
- [x] MVV-LVA move ordering
	- in `src/tinymlinternship/engine/search.py` (main search + qsearch)
	- *Killers (depth > 4): TODOs §7c.*
      
- [x] Dual-perspective encoding, king mirror / castling frame fix
	- `src/tinymlinternship/features/encoder.py` (`encode_dual`), `mirror.py`
	- castling frame fix covered in `tests/test_features.py` · TODOs §7a

---

## Cfish Features

Tree: `src/cfish/` · binary: `src/cfish/cfish.exe` · net: `src/cfish/nn-62ef826d1a6d.nnue` · notes: `NOTES/Cfish.md`

- **Search Core (Alpha-Beta Framework)** — `src/cfish/search.c`, `search.h`  
  - Alpha-Beta pruning (PV and Non-PV nodes)  
  - Iterative Deepening (ID)  
  - Aspiration Windows  
  - Quiescence Search (captures + checks)  
  - Principal Variation (PV) handling  

- **Pruning & Reduction Heuristics** — `src/cfish/search.c`  
  - Null-Move Pruning (with verification search)  
  - Futility Pruning  
  - Late Move Reductions (LMR)  
  - ProbCut  
  - Mate Distance Pruning  

- **Move Ordering & Profiling (MovePicker)** — `src/cfish/movepick.c`, `movepick.h`  
  - Transposition Table (TT) Move (highest priority)  
  - Captures sorted by MVV-LVA  
  - Killer Moves (two per ply)  
  - Countermove  
  - Butterfly History (`mainHistory`)  
  - Continuation History (`counterMoveHistory`)  
  - Low Ply History (`lowPlyHistory`)  
  - Capture History (`captureHistory`)  

- **Core Chess Logic & Move Generation**  
  - Bitboard representation — `src/cfish/bitboard.c`, `bitboard.h`  
  - Magic Bitboards — `src/cfish/magic-*.c/h`, `bmi2-*.c/h`  
  - Move generation — `src/cfish/movegen.c`, `movegen.h`  
  - Perft (move generation verification)  
  - Zobrist Hashing — `src/cfish/position.c` (hash keys)  

- **Evaluation & NNUE**  
  - Classical / hybrid eval — `src/cfish/evaluate.c`, `evaluate.h` (`DefaultEvalFile`)  
  - NNUE — `src/cfish/nnue.c`, `nnue.h`, `nnue-regular.c`, `nnue-sparse.c`  
  - Material / pawns / endgame helpers — `material.c`, `pawns.c`, `endgame.c`, `psqt.c`  

- **Transposition Table (TT)** — `src/cfish/tt.c`, `tt.h`  
  - Cluster-based hash table (4 entries per bucket)  
  - Depth + age replacement strategy  
  - Hashfull calculation  
  - Parallel TT clearing  

- **Data Structures & Incremental Updates**  
  - Position object — `src/cfish/position.c`, `position.h`  
  - State Stack (history, repetition detection)  
  - Incremental PSQT updates  
  - Repetition / Draw detection (50-move, 3-fold)  

- **Engine Infrastructure & I/O**  
  - Full UCI Protocol — `src/cfish/uci.c`, `uci.h`, `ucioption.c`  
  - Polyglot Opening Book — `src/cfish/polybook.c`, `polybook.h`  
  - Syzygy Tablebases — `src/cfish/tbprobe.c`, `tbprobe.h`  
  - Time Management — `src/cfish/timeman.c`, `timeman.h`  

- **Parallelism & Hardware Support**  
  - Multi-threading (Lazy SMP) — `src/cfish/thread.c`, `thread.h`  
  - NUMA support — `src/cfish/numa.c`, `numa.h`  
  - Large Pages support (TT allocation)  
  - Build — `src/cfish/Makefile`, `config.h`  

- **Benchmarking & Debugging**  
  - `bench` command — `src/cfish/benchmark.c`  
  - Compiler SIMD detection (SSE, AVX2, AVX512, NEON)  

---

## Cfish features SARDINE must change

Cross-walk of the list above vs [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md) v1 + Wio budget (192 KB RAM / ~500 KB flash).  
**Change** = rewrite or retarget · **Strip** = remove for device v1 · **Keep** items are omitted (e.g. bitboards, αβ skeleton, MVV-LVA, Zobrist, perft).

PC reference implementations (Python) for the student path:

| Layer | Path |
| ----- | ---- |
| Features 844 + bucket metadata | `src/tinymlinternship/features/` |
| Student NNUE (F3 `SingleHeadNNUE`; legacy `BucketedNNUE` pilots) | `src/tinymlinternship/nnue/model.py` |
| Search / eval hooks | `src/tinymlinternship/engine/search.py`, `eval_nnue.py`, `eval_hce.py` |
| ACPL judge stack | `src/tinymlinternship/bot_eval/`, `scripts/eval_bot_acpl.py` |
| Cfish baseline tree | `src/cfish/` |

### Evaluation (largest delta)

| Cfish today | Required change |
| ----------- | --------------- |
| Stock HalfKP-class NNUE (`src/cfish/nn-62ef826d1a6d.nnue`, hybrid/pure classical in `nnue.c` / `evaluate.c`) | Replace with **SARDINE micro-NNUE**: features **844** (`features/`), **not** HalfKP |
| Dense L1 + large expert layout | Shared **sparse** L1 `844 → W` (\(W \in \{128,256\}\)), int8 non-zeros only in flash; **single** head `2W → 1` until §D (**F3**) — multi-expert only after ablation locks a scheme. PC model: `nnue/model.py` (legacy 8-head pilots = experimental) |
| Centipawn-ish eval scale | Output **expected reward** \([-1,+1]\) via **tanh LUT**; train labels = Lc0 `expected_reward` (`scripts/label_positions.py`), not SF cp (C1) |
| Incremental FT for SF topology | **New** dual-POV accumulators (int16), STM‖Opp concat; lazy add/sub + full refresh on king centre-file cross; castling / EP bit updates — **no** piece-count router until §D (F3) |
| Hybrid HCE + NNUE | Device v1: **NNUE-first** (PC HCE baseline: `engine/eval_hce.py`) |
| AVX2/AVX-512/NEON NNUE kernels | Retarget to **Cortex-M4** int8×int16 → int32 MAC (no desktop SIMD) |

### Search core & heuristics

| Cfish today | Required change |
| ----------- | --------------- |
| Full Stockfish search stack | Keep αβ + **ID**; **simplify** — no stack-surfing; budget ~**1 s**/move fixed / ID |
| Quiescence (captures + checks) | Retarget: v1 PC path is **captures (+ promotions)** first; re-evaluate checks if nps allows |
| Null-move, futility, LMR | **Keep in v1** but re-tune thresholds for shallow / slow eval (SPSA on search params only) |
| ProbCut, mate-distance pruning, aspiration | Optional / **defer** until nps + Elo justify complexity |
| Lazy eval (SF style) | Reimplement paired with **SARDINE lazy accumulators** (skip NNUE when TT cutoff makes eval moot) |

### Move ordering

| Cfish today | Required change |
| ----------- | --------------- |
| TT move + MVV-LVA + 2 killers | **Keep** TT move, MVV-LVA; killers when depth > 4 |
| Countermove, butterfly / continuation / low-ply / capture history | **Strip for v1** (blueprint: no full history suite; countermove history out of v1) |
| Policy net ordering | **None** in v1 (search-only; policy head is v2) |

### Transposition table

| Cfish today | Required change |
| ----------- | --------------- |
| Multi-MB cluster TT, parallel clear, large pages | **Shrink** to **128–160 KB** total; design entry (~10 B tight vs 16 B aligned) for SAMD51; single-thread clear |
| Hashfull / age replacement | Keep idea; retune for tiny table |

### Memory, I/O, infrastructure

| Cfish today | Required change |
| ----------- | --------------- |
| Full UCI | **Minimal UCI** (enough for cutechess / Serial on Wio); not full option surface |
| Polyglot opening book | **Strip v1** (flash → search + NNUE); book only post–Elo gate |
| Syzygy TB (WDL/DTZ/DTM) | **Strip** — no room under 192 KB RAM + flash budget |
| Time management (`optimumTime` / `maximumTime`) | Retarget to **~1 s** move budget + simple clocks; no multi-hour TC machinery |
| Lazy SMP multi-thread + NUMA + large pages | **Strip** — Wio is **single-core**; no NUMA/large pages |
| Desktop `bench` + SIMD feature flags | Keep **nps/bench** idea; add **Wio** metrics (`platform=wio`, `-O3` vs `-Os`); drop x86 SIMD detect for device build |
| PSQT incremental classical eval path | Drop or gate behind fallback ladder; not the production eval |

### Explicit non-goals when forking Cfish (v1)

Do **not** port or re-enable for the first ship gate:

- Opening book · tablebases · multi-thread / NUMA  
- Full history move-ordering suite · policy network  
- HalfKP / stock SF net format as the student  
- Desktop-only SIMD paths as the only NNUE kernel  
- Multi-expert routing before §D ablation (**F3**)

Reference: blueprint §Search, §Memory, §NNUE Architecture, §Move ordering · [ai-feed.md](ai-feed.md) · eng gates [TODOs.md](TODOs.md) §7–§8.

---

#todo ideas for agent eval: tactical test suites, BayesElo, Ordo, OpenBranch fishtest

---

#core
