# SARDINE — console commands

Quick snippets to run **active** SARDINE code (feature encoder, tests). Run from the **project root**.

**Convention:** `py -3.12` on Windows (`python3` on Linux/macOS).

Spec: [SARDINE.md](SARDINE%20Engine%20Blueprint.md) · Legacy export pipeline: `legacy/pre-sardine/`

---

## Environment (once)

```bash
py -3.12 -m pip install -e .
py -3.12 -m pip install -e ".[dev]"
```

`pytest` comes from the `[dev]` extra in `pyproject.toml`.

---

## Tests — feature encoder (`tests/test_features.py`)

33+ tests: index map (844 dims), king mirroring, tactical planes, `encode_perspective`, `encode_dual`, `bucket_id`, golden FEN snapshots.

### Option A — pytest (recommended)

```bash
py -3.12 -m pytest tests/test_features.py -v
```

All project tests:

```bash
py -3.12 -m pytest -v
```

Data smoke test only:

```bash
py -3.12 -m pytest tests/test_data.py -v
```

### Option B — no pytest

```bash
py -3.12 -c "import sys; sys.path.insert(0,'src'); import tests.test_features as t; [getattr(t,n)() or print('ok',n) for n in dir(t) if n.startswith('test_')]"
```

---

## Manual checks — encoder

Start position (expect **36** active features: 32 pieces + 4 castling):

```bash
py -3.12 -c "import sys; sys.path.insert(0,'src'); import chess; from tinymlinternship.features import encode_perspective; print(len(encode_perspective(chess.Board(), chess.WHITE)))"
```

Dual perspective on startpos (symmetric — same length, same indices):

```bash
py -3.12 -c "import sys; sys.path.insert(0,'src'); import chess; from tinymlinternship.features import encode_dual; w,b=encode_dual(chess.Board()); print(len(w), len(b), w==b)"
```

After `1. e4` (asymmetric — white ≠ black lists):

```bash
py -3.12 -c "import sys; sys.path.insert(0,'src'); import chess; from tinymlinternship.features import encode_dual; b=chess.Board(); b.push(chess.Move.from_uci('e2e4')); w,bl=encode_dual(b); print(len(w), len(bl), w!=bl)"
```

Bucket on startpos (expect `7`):

```bash
py -3.12 -c "import sys; sys.path.insert(0,'src'); import chess; from tinymlinternship.features import bucket_id; print(bucket_id(chess.Board()))"
```

Index map sanity (expect `844 716 704`):

```bash
py -3.12 -c "import sys; sys.path.insert(0,'src'); from tinymlinternship.features.index_map import FEATURE_DIM, piece_square_count, meta_base; print(FEATURE_DIM, piece_square_count(), meta_base())"
```

---

## Data & analysis (SARDINE training prep)

```bash
py -3.12 scripts/download_data.py
py -3.12 scripts/plot_piece_count_distribution.py
```

---

## Legacy pre-SARDINE (archive)

Old Wio value-net scripts live under `legacy/pre-sardine/`. Example:

```bash
py -3.12 legacy/pre-sardine/scripts/run_model.py --help
```

Full legacy reference: [Commands.md](Commands.md) (export pipeline, Arduino headers).

---

[← Back to Notes index](_notes.md)