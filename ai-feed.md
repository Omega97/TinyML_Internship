# JSON vs Parquet for `{fen, value, visits}`

Goal §1 names **JSON** as the dataset shape. On disk we currently write **both** JSON and a parquet twin. Numbers below are from the live join (2026-08-21): **882 730** unique EPDs.

| Artifact | JSON | Parquet | JSON / parquet |
|----------|-----:|--------:|---------------:|
| Joined table `data/processed/board_eval/fen_value_visits.*` | 108.5 MB | 22.2 MB | **4.9×** |
| All per-source slices under `fen_value_visits/` | 110.6 MB | 22.6 MB | **~5×** |
| Slices **+** join (today, dual format) | | | **~264 MB** of the same three columns twice |

JSON is pretty-printed (`indent=2`). Compact JSON would shrink, but not down to parquet: FENs are long repeated strings with no columnar compression.

The row is always:

```json
{"fen": "...", "value": 0.093, "visits": 10055}
```

---

## What each format is good at

### JSON (`.json`)

- Matches [Goal.md](Goal.md) §1 (the spec example is a JSON list).
- Readable in an editor / `Get-Content` / git diff for **small** slices (smoke, 1–10 games).
- No extra library to eyeball a file.
- Language-agnostic: any script can `json.load`.

Costs:

- **5× disk** vs parquet on this table; toward \(10^6\)–\(10^7\) rows that becomes gigabytes.
- Slow to parse (full document into RAM). Pandas `read_json` on the 109 MB join is much slower than `read_parquet`.
- Pretty-print makes diffs noisy and writes slow.
- No column prune: you always load `fen` even if you only need `value`/`visits`.
- Not a training format (PyTorch `DataLoader` wants arrays / parquet / memory-map).

### Parquet (`.parquet`)

- Columnar, compressed (FEN dictionary-codes well). **22 MB** for the same 882 k rows.
- Fast load; can read only `["value", "visits"]`.
- Already the pipeline’s working format: extracts, labeled tables, join, `inspect_fen_value_visits.py` default to parquet.
- `pyarrow` is already in the `data` extra.
- Natural input for Goal §2 training.

Costs:

- Not human-readable; need `pandas` / `inspect_fen_value_visits.py`.
- Goal.md’s illustrative format is JSON, not parquet.
- Slightly less trivial to hand to a non-Python tool.

### Dual (current default)

Every converter (`lichess_dump_to_fen_value_visits.py`, `export_*`, `join_fen_value_visits.py`) writes **JSON + parquet**.

- Convenience: inspect in a text editor *and* train from parquet.
- Waste: two copies of every slice **and** of the join. The join JSON (109 MB) is the expensive one.
- Join already prefers parquet when both exist (`join_fen_value_visits.py`), so the JSON slices are unused at join time.

---

## Options

| Option | Disk | Train / join | Spec / inspect | Verdict |
|--------|------|----------------|----------------|---------|
| **A. JSON only** | Worst (~5×) | Slow | Best for Goal.md | Fine for toys, not for \(10^6\) rows |
| **B. Parquet only** | Best | Best | Need a small inspector | **Best default for production tables** |
| **C. Dual, always** | ~6× parquet (today) | Fine (join ignores JSON) | Easy | Current; does not scale |
| **D. Parquet canonical + JSON only for small slices** | Almost B | Same as B | JSON where it is readable | **Best compromise** |
| **E. JSON Lines (`.jsonl`)** | Between JSON and parquet | Streaming, still text-heavy | Grep-able | Not worth a third format |

Do **not** add CSV, HDF5, or Arrow IPC for this table. Three columns, two numeric: parquet covers it.

---

## Recommendation

1. **Canonical table = parquet**  
   `data/processed/board_eval/fen_value_visits.parquet` is what train / join / inspect should read. Goal §2 should load this (or sharded parquet), not the JSON.

2. **JSON is a view, not a store**  
   Keep JSON for smoke and short dump ranges (thousands of rows), or behind `--json` / `--no-json` (join already has `--no-json`). Stop writing pretty JSON for the full join and for 100 k-row dump slices.

3. **If Goal.md must stay JSON-shaped**  
   Treat the JSON example as the *schema*, not the *file format*. The columns are `fen`, `value`, `visits`; the on-disk encoding can be parquet. Optionally dump a compact JSON **sample** (first \(N\) rows) for the write-up.

4. **Re-join after policy change**  
   After dropping JSON twins, disk for slices+join drops from ~264 MB to ~45 MB at current size, and the gap grows linearly with more Lichess dump batches.

---

## Practical rule

| Rows | Write |
| ----:|------ |
| \(\lesssim 10^4\) (smoke, puzzles sample) | JSON + parquet is cheap |
| \(\gtrsim 10^5\) (dump batches, join, Lc0 269 k) | **parquet only** |

Training should never `json.load` the joined dataset.
