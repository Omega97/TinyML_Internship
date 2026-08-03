# Project: SARDINE

**SARDINE** — *Small Artificial RAM-restricted Deep Intelligent Neural Engine*

Chess engine for the **Wio Terminal**: neural evaluation + alpha-beta search, maximizing **Elo per byte** under **192 KB RAM** / **~500 KB flash**. No cloud, no GPU. Target: playable bot (ideally on *Lichess*).

| Doc                                                                        | Role                                                                       |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md) | Spec, architecture, pipeline, design decisions                             |
| [TODOs.md](TODOs.md)                                                       | Checkpoint checklist vs blueprint (**progress source of truth**)           |
| [ASSETS.md](ASSETS.md)                                                     | Paths, teachers, label uniformity                                          |
| [Goal.md](Goal.md)                                                         | Short mission statement                                                    |
| [ai-feed.md](ai-feed.md)                                                   | Slim code map for agents                                                   |
| [NOTES/Thesis.md](NOTES/Thesis.md)                                         | Later research: task vectors / optimal bucketing                           |

Inspiration from [Kaggle Challenge](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/linrock-my-solution-cfish-nnue-data-1st), [repo](https://github.com/linrock/minifish)

---

## Progress Overview


_Progress vs repo as of ~2026-08-01. Checklist marks work that is **done enough to build on**; italics flag caveats or disagreements with later blueprint sections. Detailed gates live in [TODOs.md](../TODOs.md) / [PROJECT.md](../PROJECT.md)._

### Run Cfish

- [x] Cfish smoke test 
	- `src/cfish/cfish.exe` · launcher `run-cfish.bat` · notes `NOTES/Cfish.md`
	- UCI `uciok` / `readyok` verified (2026-07-31)
	- *Formal `go depth 5` → `bestmove` recipe + archived **nps** bench still thin; `scripts/cfish.py` still has a stale `./cfish` path.*

### First NNUE

- [x] Download a NNUE 
	- `src/cfish/nn-62ef826d1a6d.nnue` (also `make net` in `src/cfish/`)
	- [URL](https://tests.stockfishchess.org/api/nn/nn-62ef826d1a6d.nnue)
	- *This is the **stock Stockfish-family** net shipped with Cfish — not a SARDINE-trained student.*
	
- [x] Replace the value function that Cfish uses with the new NNUE 
	- Default: `DefaultEvalFile` / INCBIN in `src/cfish/` (`evaluate.h`, `nnue.c`)
	- *No extra patch required for the stock net. Wiring a **custom SARDINE** `.nnue` into Cfish is a different (later) step — device path is still “own C port,” not necessarily Cfish-hosted student weights.*
	  
- [x] smoke test Hybrid NNUE log
	- Hybrid NNUE evaluation using the new NNUE (Cfish `Use NNUE` hybrid with EvalFile present)
	- *UCI smoke with net on disk is done; a dedicated hybrid-only log / nps artifact is not archived yet.*
	  
- [ ] Evaluation with Stockfish
	- 10 quick self-play games + run the engine on every move to get the 5 biggest blunders
	- benchmark the nps
	- *Stack ready: `scripts/eval_bot_acpl.py` (top-5 CPL) + self-play recorders. **Cfish** self-play + ACPL still open (carry-over 2026-08-01). ACPL gates already exist for HCE / pilot NNUE / Sunfish / Lc0 under `plots/PGN_and_JSON/`.*

### Dataset

- [x] Download the raw data 
	- engine games + human games, to have good coverage
	- mainly board positions
	- *Lc0 chunks ~1.1 GiB in `data/raw/lc0/` (`scripts/download_lc0.py`). Human side is only **smoke** (`data/raw/lichess_smoke50.pgn`) — full Lichess monthly dump under `data/raw/lichess/` **not** downloaded. ChessBench bags = encoder smoke only.*
	  
- [ ] remove duplicate positions 
	- but keep track of the multiplicity, so we may use it later
	- *No dedicated dedup + multiplicity pass in the data pipeline yet.*
	  
- [x] add Stockfish evaluations to each board state
	- data is a list of board-eval pairs
	- if Stockfish returns centipawns then we will stick with that
	- ***Disagrees with §Training data / ASSETS:** train labels are **not** Stockfish centipawns — they are **Lc0** WDL → **`expected_reward`** White POV ∈ \([-1,+1]\) (`scripts/label_positions.py`, teacher `791556`). Stockfish is the **ACPL / match judge** only. Mini blocks labeled: `lichess_labeled` + `lc0_labeled`.*
	  
- [x] Clean the data into a single, uniform dataset
	- list of $(s, v)$ pairs
	- *Done as `data/processed/labeled/{train,val}.parquet` + `manifest.json` (~5.3k / 214 rows, seed 42). Uniform on **`expected_reward` only**. Scale is **mini/smoke**, not production \(10^5\)–\(10^6+\).*
    

### Train the Network

- [x] Train small NNUE 
	- input layer + 2 hidden layers + output layer, standard input shape
	- ***Disagrees with §NNUE Architecture:** trained net is **bucketed** micro-NNUE — shared L1 `844 → W=128` + **8** expert heads `2W → 1` (not plain 2-hidden). Via `scripts/train_nnue.py`; checkpoints `models/checkpoints/nnue/pilot_W128_844/` (ChessBench pilot) and `smoke_prod_W128_844/` (mini labeled). nnue-pytorch adapt, gradual L1 prune, PTQ export still open.*
      
- [ ] Thesis idea: Replace the single NNUE with a MoE
	- training by bucketing the states based on the task vectors for each data point
	- *Runtime already multi-expert (piece-count + queen-split, interim **8** buckets until §D). Task-vector / dispatcher research is **later** ([Thesis.md](Thesis.md), §Later) — not blocking v1 Elo gate.*
      
- [x] Evaluation with Stockfish
	- 10 quick self-play games + run the engine on every move to get the 5 biggest blunders
	- bench the nps
	- *ACPL self-play for pilot NNUE archived (`plots/PGN_and_JSON/nnue_d1_gate*`; top-5 CPL in `eval_bot_acpl.py`). Not a full closed package for every checkpoint; **nps** microbench still open (TODOs §B). Known issue: NNUE d2 ACPL collapse vs d1.*

### On the Hardware

- [ ] Wio Terminal smoke test 
	- *Legacy pre-SARDINE Wio sketches removed (2026-07-22). Device work restarts with C-port steps E–F — not started.*
      
- [ ] Evaluation with Stockfish
	- 30 self-play games
	- bench the nps
      
- [ ] Connect the Wio to Lichess and play!

## For the Thesis

- [ ] Compare the new and other techniques 
	- bucketing through task vectors vs clustering directly through embeddings
	- *After a working base NNUE + Elo path; see §Later: optimal feature combinations & task vectors.*

---

#core
