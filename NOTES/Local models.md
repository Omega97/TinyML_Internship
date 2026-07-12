# Modelli locali — inventario

Indice dei modelli **scaricati** e **allenati** nel repo. Path relativi alla root del progetto.

Per il survey dei candidati teacher e le decisioni di design → [Models.md](Models.md).  
Per le schede agente (eval + search + comandi) → [agents/](agents/).

_Aggiornato: 2026-07-10_

---

## Riepilogo rapido

| Categoria          | Path root                                                     | Stato         | Uso principale                     |
| ------------------ | ------------------------------------------------------------- | ------------- | ---------------------------------- |
| Teacher Lc0        | [`models/teacher/`](../models/teacher/)                       | ✅ installato  | Label training, bot baseline, demo |
| Teacher HF         | [`models/teacher/hf/`](../models/teacher/hf/)                 | ✅ installato  | Smoke inferenza veloce             |
| Stockfish 18       | [`models/teacher/stockfish/`](../models/teacher/stockfish/)   | ✅ installato  | ACPL gate, analisi mosse           |
| Sunfish            | [`models/teacher/sunfish/`](../models/teacher/sunfish/)       | ✅ clone       | Calibrazione depth-1 (ACPL)        |
| NNUE SARDINE       | [`models/checkpoints/nnue/`](../models/checkpoints/nnue/)     | ✅ 2 run pilot | Engine `--eval nnue`               |

---

## Modelli scaricati

### Lc0 — teacher principale

**Installazione:** `py -3.12 scripts/download_teacher.py` · **Smoke:** `py -3.12 scripts/smoke_test_teacher.py`

| Artefatto | Path | Size | Note |
|-----------|------|------|------|
| Binario | [`models/teacher/lc0/lc0.exe`](../models/teacher/lc0/lc0.exe) | 2.1 MB | v0.32.1, OpenBLAS |
| Rete **fast** (default) | [`models/teacher/lc0/791556.pb.gz`](../models/teacher/lc0/791556.pb.gz) | 18 MB | Bot, labeling, demo |
| Rete T1-256 | [`models/teacher/networks/t1-256x10-distilled-swa-2432500.pb.gz`](../models/teacher/networks/t1-256x10-distilled-swa-2432500.pb.gz) | 35 MB | Alternativa CPU |
| Rete BT4 | [`models/teacher/networks/BT4-1024x15x32h-swa-6147500.pb.gz`](../models/teacher/networks/BT4-1024x15x32h-swa-6147500.pb.gz) | 320 MB | Qualità ref, troppo lento |
| Manifest | [`models/teacher/manifest.json`](../models/teacher/manifest.json) | — | SHA256 BT4 + lc0 |

**Fonte reti:** [training.lczero.org](https://training.lczero.org/)  
**Codice:** `src/tinymlinternship/engine/eval_lc0.py` (`Lc0Teacher`)  
**Label:** `expected_reward = (W − L) / 1000` via UCI WDL · script [`scripts/label_positions.py`](../scripts/label_positions.py)

**Preset `--network`:** `fast` · `t1-256` · `bt4` in `record_teacher_game.py`

---

### Hugging Face — teacher smoke

**Installazione:** `py -3.12 scripts/download_hf_teacher.py`

| Modello | Path | Size | Fonte HF | Note |
|---------|------|------|----------|------|
| **chess_lite** | [`models/teacher/hf/chess_lite/chess_lite.pth`](../models/teacher/hf/chess_lite/chess_lite.pth) | 39 MB | [satana123/chess_lite](https://huggingface.co/satana123/chess_lite) | ~2 ms/eval, gioco debole d1 |
| **artoria-small** | [`models/teacher/hf/artoria-zero/small/checkpoint.pt`](../models/teacher/hf/artoria-zero/small/checkpoint.pt) | 101 MB | [Shinapri/artoria-zero](https://huggingface.co/Shinapri/artoria-zero) | Config in `small/config.json` |

**Codice:** `eval_chess_lite.py` · **Play:** `scripts/record_hf_game.py --model auto`

---

### Stockfish 18 — analisi / gate

| Artefatto | Path | Size | Note |
|-----------|------|------|------|
| Binario | [`models/teacher/stockfish/stockfish.exe`](../models/teacher/stockfish/stockfish.exe) | 109 MB | SF 18 avx2 |

**Uso:** `scripts/eval_bot_acpl.py` (ACPL su self-play), non teacher per training NNUE.

---

### Sunfish — baseline calibrazione

| Artefatto | Path | Note |
|-----------|------|------|
| Repo | [`models/teacher/sunfish/`](../models/teacher/sunfish/) | Clone GitHub · UCI instabile su Windows |
| Self-play | [`scripts/sunfish_selfplay_pgn.py`](../scripts/sunfish_selfplay_pgn.py) | Gate depth-1: ACPL **818 cp** |

---

## Modelli allenati (SARDINE NNUE)

**Training:** `py -3.12 scripts/train_nnue.py --run-name <name>`  
**Default engine:** `NNUE_CHECKPOINT_DEFAULT` → `pilot_W128_844/best.pt` in `settings.py`

### `pilot_W128_844` — attivo ✅

Encoder **844-dim** (716 base + 128 tattico). Checkpoint di produzione per `--eval nnue`.

| File | Path |
|------|------|
| **best** (usato dall'engine) | [`models/checkpoints/nnue/pilot_W128_844/best.pt`](../models/checkpoints/nnue/pilot_W128_844/best.pt) |
| last | [`models/checkpoints/nnue/pilot_W128_844/last.pt`](../models/checkpoints/nnue/pilot_W128_844/last.pt) |
| config | [`models/checkpoints/nnue/pilot_W128_844/config.json`](../models/checkpoints/nnue/pilot_W128_844/config.json) |
| history | [`models/checkpoints/nnue/pilot_W128_844/history.json`](../models/checkpoints/nnue/pilot_W128_844/history.json) |

| Metrica | Valore |
|---------|--------|
| Parametri | 110 216 |
| Epoche | 5 |
| val_mse | **0.056** |
| Dataset | ChessBench → `data/processed/chessbench/splits/` |
| Gate depth-1 ACPL | **121.1 cp** → Elo ~1644 |

**Schede agente:**
- [agents/nnue-w128-844-d1.md](agents/nnue-w128-844-d1.md) — depth 1, qsearch off
- [agents/nnue-w128-844-d2.md](agents/nnue-w128-844-d2.md) — depth 2

**Demo:** [`images/nnue_d2_game.gif`](../images/nnue_d2_game.gif) · architettura: [`plots/sardine_nnue_architecture.png`](../plots/sardine_nnue_architecture.png)

```bash
py -3.12 scripts/run_engine.py --eval nnue --nnue-checkpoint models/checkpoints/nnue/pilot_W128_844/best.pt --depth 2
```

---

### `pilot_W128_chessbench` — superseded ⚠️

Primo pilot su encoder **716-dim** (pre-tactical). Non usare con l'engine 844-dim.

| File | Path |
|------|------|
| best | [`models/checkpoints/nnue/pilot_W128_chessbench/best.pt`](../models/checkpoints/nnue/pilot_W128_chessbench/best.pt) |
| config | [`models/checkpoints/nnue/pilot_W128_chessbench/config.json`](../models/checkpoints/nnue/pilot_W128_chessbench/config.json) |
| history | [`models/checkpoints/nnue/pilot_W128_chessbench/history.json`](../models/checkpoints/nnue/pilot_W128_chessbench/history.json) |

| Metrica | Valore |
|---------|--------|
| Parametri | 93 832 |
| val_mse | **0.058** (ep 5) |
| Note | Sostituito da `pilot_W128_844` dopo estensione tattica |

---

## Confronto eval / baseline (depth-1 gate)

| Backend | Modello / checkpoint | ACPL (cp) | Artefatti |
|---------|----------------------|-----------|-----------|
| **NNUE** | `pilot_W128_844/best.pt` | **121.1** | `plots/nnue_d1_gate_acpl.json` |
| **HCE** | euristica built-in | **275.0** | `plots/hce_d1_gate_acpl.json` |
| **Sunfish** | repo `models/teacher/sunfish/` | **818.3** | `plots/sunfish_d1_gate_acpl.json` |

**HCE depth-2 demo:** [`images/hce_d2_game.gif`](../images/hce_d2_game.gif) · scheda [agents/hce-d2-qsearch.md](agents/hce-d2-qsearch.md)

---

## Legacy pre-SARDINE (archivio)

Encoder **768-dim**, output centipawn — **non compatibile** con pipeline SARDINE 844.

| Categoria | Path |
|-----------|------|
| Checkpoints PyTorch | [`legacy/pre-sardine/models/checkpoints/`](../legacy/pre-sardine/models/checkpoints/) |
| Export int8 / ONNX | [`legacy/pre-sardine/models/exported/`](../legacy/pre-sardine/models/exported/) |
| Headers Arduino/Wio | [`legacy/pre-sardine/models/arduino/models/`](../legacy/pre-sardine/models/arduino/models/) |

---

## Prossimi (non ancora locali)

| Modello | Path previsto | Script |
|---------|---------------|--------|
| NNUE produzione (nnue-pytorch) | `models/checkpoints/nnue/<prod_run>/` | post Lichess + Lc0 labeling |
| Export int8 + tanh LUT | `models/exported/` | PTQ pipeline |
| Lichess PGN | `data/raw/lichess/` | `download_lichess.py` (futuro) |

---

## Script di installazione

| Script | Cosa scarica |
|--------|--------------|
| [`scripts/download_teacher.py`](../scripts/download_teacher.py) | lc0.exe + BT4 (+ 791556 nello zip lc0) |
| [`scripts/download_hf_teacher.py`](../scripts/download_hf_teacher.py) | chess_lite + artoria-small |
| [`scripts/download_lc0.py`](../scripts/download_lc0.py) | Chunk training data Lc0 (~1–2 GB) |
| [`scripts/download_chessbench.py`](../scripts/download_chessbench.py) | ChessBench test split (dati, non modello) |