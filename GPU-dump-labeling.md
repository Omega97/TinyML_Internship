# GPU dump labeling

How `scripts/lichess_dump_to_fen_value_visits-gpu.py` uses the NVIDIA GB10 (Spark) for Lc0 teacher labels.

The CPU script `lichess_dump_to_fen_value_visits.py` is unchanged.

## What was wrong

The `-gpu` script was a copy of the CPU pipeline. Teacher labels went through a persistent Lc0 **UCI** process:

- backend default **`blas`** (CPU OpenBLAS), not CUDA
- one FEN per `position fen` + `go nodes 1`
- each eval is a batch-size-1 kernel plus pipe wait

On this machine that is ~40 positions/s even with `--backend cuda-fp16`. The GB10 stayed idle.

PGN extract (zstd + `python-chess`) is CPU-only and still is. The GPU work is **labeling unique FENs** with the Lc0 value head.

## What changed

### 1. `lc0 fenbatch` mode (`/home/omar/lc0`)

Lc0 on this box already had CUDA backends (`cuda-auto`, `cuda`, `cuda-fp16`) and `EvaluateBatch`, but nothing exposed a FEN list to that API.

New mode: stdin FENs → CUDA minibatch → stdout STM WDL.

```
lc0 fenbatch --weights=… --backend=cuda-fp16 --batch-size=256 --nncache=0
```

Protocol:

- process writes `READY` on stdout after the net is loaded and warmed up
- client writes one FEN per line
- a full `--batch-size` (or `EVAL` / EOF) runs `Backend::EvaluateBatch`
- each FEN gets `ok w d l` (permille-rounded, same formula as UCI `info wdl`)
- `QUIT` exits

Files: `src/tools/fenbatch.{h,cc}`, hooked from `src/main.cc` and `meson.build`. Binary: `/home/omar/lc0/build/release/lc0` (symlink `models/teacher/lc0/lc0.exe`).

### 2. Python client

`src/tinymlinternship/engine/eval_lc0_batch.py` — `Lc0FenBatch` holds that process and `evaluate_batch(fens)`.

Terminal positions are still assigned locally (mate / stalemate / insufficient material), same as the UCI teacher.

### 3. GPU dump script

`scripts/lichess_dump_to_fen_value_visits-gpu.py`:

- labels through `Lc0FenBatch` instead of `Lc0Teacher` UCI
- **overlap (default):** each new unique FEN is queued while PGN parse continues; a side thread sends minibatches to the GPU. CUDA init runs on that thread so skip/parse is not blocked on driver startup. Partial batches flush after 20 ms so the GPU does not wait for a full minibatch if extract stalls.
- `--no-gpu-overlap` extracts first, then labels (GPU idle during parse)
- `--skip-extract` still labels an existing parquet in GPU minibatches, with parquet checkpoints every `--batch` rows

New flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--backend` | `cuda-auto` | Lc0 NN backend (`cuda-fp16` to force half precision) |
| `--nn-batch` | `256` | Positions per `EvaluateBatch` (1..1024) |
| `--backend-opts` | empty | Passed to `lc0 --backend-opts` |
| `--no-gpu-overlap` | off | Serial extract then label |

Slice names, dropout, max-draw, max-moves, JSON + `features.npz` are the same as the CPU script.

## Throughput (this Spark)

Teacher net `791556.pb.gz`, backend `cuda-fp16`:

| Path | Positions / s |
|------|----------------|
| UCI `go nodes 1` (CUDA) | ~40 |
| `fenbatch` batch=256 | ~21,000 |

About **500×** on the teacher step. Extract + JSON encode are still CPU.

`nvidia-smi` stays low during a large game skip (no positions yet) and during encode. It should be in a compute kernel once unique FENs start appearing, and for the whole `--skip-extract` label pass.

## Commands

Same game range as the CPU script, GPU path:

```bash
python3.12 -u scripts/lichess_dump_to_fen_value_visits-gpu.py 1020000 1030000 --dropout 0.95
```

Keep the GB10 busier:

```bash
python3.12 -u scripts/lichess_dump_to_fen_value_visits-gpu.py 1020000 1030000 --dropout 0.95 --backend cuda-fp16 --nn-batch 512
```

`--backend cuda` is fp32 (slightly more stable labels, slower). fp16 WDL can differ from UCI permille by ~0.001–0.002.

## Tests

- `tests/test_eval_lc0_batch.py` — WDL conversion + fenbatch line parser (no GPU)
- `tests/test_lichess_dump_gpu.py` — extract callback + batched `label_extract` with a fake teacher
