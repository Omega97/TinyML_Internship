# Demo — Lichess monthly dump → `{fen, wdl, value, visits}`

Stream a **slice** of a Lichess standard-rated monthly dump (`.pgn.zst`) into Goal §1 JSON: unique `{fen, wdl, value, visits}` plus a parquet twin.

Do **not** decompress the whole dump to disk. The converter streams zstd, skips games by `[Event ` header count (no chess parse on the skip), then parses games `[n, m)` (1-based, **n included, m excluded**).

Run from the **repo root**.

---

## Prerequisites

```powershell
pip install -e ".[data]"
```

| Need | Path / notes |
| ---- | ------------ |
| CLI | `scripts/lichess_dump_to_fen_value_visits.py` |
| Dump | `data/raw/lichess/dumps/lichess_db_standard_rated_2026-07.pgn.zst` (~27 GiB, keep compressed) |
| Teacher | `models/teacher/lc0/lc0.exe` + `791556.pb.gz` (`LC0_NETWORK_DEFAULT`) |
| Python extras | `python-chess`, `pandas`, `pyarrow`, `zstandard`, `tqdm` |

If the dump is missing:

```powershell
py -3.12 scripts/download_lichess_dump.py --month 2026-07
```

---

## Convert games `n`–`m`

```powershell
py -3.12 scripts/lichess_dump_to_fen_value_visits.py n m
```

`n` is **included**, `m` is **excluded** (Python `range` style). Adjacent batches `1 1000` then `1000 2000` do not share a game. Example: `1 11` is the first ten games.

stderr shows three tqdm bars (skip, extract, label), redrawn about once a second, with rate and **ETA**. Pass `--progress-every 0` to hide them.

### Smoke (first 10 games → stem `…_0-10`)

```powershell
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10 --force-label --progress-every 1
```

Expected: ~10 games, ~657 plies, **645** unique EPDs, then Lc0 labels. Wall time on this machine was ~52 s (extract is ~0.1 s; labeling dominates).

### Larger slice

```powershell
# games [20_000_001, 21_000_001) — one million games
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 20000001 21000001 --progress-every 10000
```

Skipping millions of games still streams the compressed file from the start (header count only). Plan for minutes of skip time before parse starts.

---

## Outputs (default dump month `2026-07`)

| Artifact | Path |
| -------- | ---- |
| Extract | `data/raw/lichess/lichess_db_standard_rated_2026-07_<n>-<m>_extract.parquet` |
| Extract stats | same stem + `.stats.json` |
| Labeled parquet | `data/processed/labeled/lichess_db_standard_rated_2026-07_<n>-<m>.parquet` |
| **JSON + parquet twin** | `data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_<n>-<m>/fen_value_visits_lichess_db_standard_rated_2026-07_<n>-<m>.json` |
| **features.npz** | same folder (`encode_slice_features.py`; skip with `--skip-encode`) |

JSON objects:

```json
{"fen": "...", "wdl": [0.501, 0.399, 0.100], "value": 0.401, "visits": 3}
```

- `fen` — unique EPD (SHA-256 of `board.epd()`).
- `wdl` — STM `[W, D, L]` from Lc0 (permille / 1000).
- `value` — White-POV \(\pm(W-L)\).
- `visits` — how many times that EPD appeared in games `n`–`m`.

Join all slices (sum visits, sort descending):

```powershell
py -3.12 scripts/join_fen_value_visits.py
```

Writes `data/processed/board_eval/fen_value_visits.parquet` and `.json`.

---

## Useful flags

| Flag | Meaning |
| ---- | ------- |
| `--input PATH` | Other `.pgn.zst` / `.pgn` (default: 2026-07 dump) |
| `--max-unique N` | Cap unique EPDs (`0` = keep all in the range) |
| `--stop-when-unique-full` | Stop parsing once the unique cap is hit |
| `--no-startpos` | Do not count the initial FEN |
| `--progress-every N` | `0` hides bars; any positive value shows skip / extract / label bars (~1 s, with ETA) |
| `--batch N` | Label checkpoint size (default 20 000) |
| `--skip-extract` / `--skip-label` | Resume after extract or labels already exist |
| `--extract` / `--labeled` / `--output` | Override default paths |

Labeling can be resumed: if the labeled parquet is shorter than the extract, it continues from the last row.

---

## Inspect

```powershell
py -3.12 scripts/inspect_fen_value_visits.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_db_standard_rated_2026-07_1-10/fen_value_visits_lichess_db_standard_rated_2026-07_1-10.json
```

Smoke JSON already on disk: `fen_value_visits_lichess_db_standard_rated_2026-07_1-10.json` (645 rows).
