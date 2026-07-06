# SARDINE — Dataset survey

_Riferimenti: [SARDINE Engine Blueprint](SARDINE%20Engine%20Blueprint.md) §Training data · [Lc0 preprocessing pipeline](Lc0%20preprocessing%20pipeline.md) · [Models.md](Models.md) §Teacher._

---

## Teacher scelto (v1)

**Lc0 value head** — rete **BT4** (o ultima variante BT4 stabile da [training.lczero.org](https://training.lczero.org/)), via binario `lc0` + UCI.

| Campo                              | Valore                                                                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Output nativo                      | WDL — tre probabilità che sommano a 1                                                                                  |
| **Expected reward** (side to move) | `Q = W − L` (range circa `[-1, +1]`)                                                                                   |
| Uso depth-1 baseline               | `position fen …` → `go nodes 1` → leggere WDL dalla risposta UCI                                                       |
| Uso labeling batch                 | stesso binario/rete su ogni FEN in `positions.parquet` (e futuro Lichess)                                              |
| Fallback                           | Stockfish 16+ con `UCI_ShowWDL value true` + comando `eval`                                                            |
| Alternativa leggera                | [chess_lite](https://huggingface.co/satana123/chess_lite) (PyTorch, tanh in `[-1,1]`) solo se `lc0` non è installabile |

**Validazione teacher:** correlazione `W−L` vs outcome reale su split val + posizioni Syzygy note; se calibrazione scarsa → fallback Stockfish WDL.

---

## Dataset scaricati (locale)

### 1. Lc0 training chunks — **supplemento / volume**

| | |
|---|---|
| **Path raw** | `data/raw/lc0/` |
| **Tars** | `data/raw/lc0/tars/` — 2 shard (~419 MiB + ~762 MiB) |
| **Chunks** | `data/raw/lc0/chunks/` — **54 866** file `.gz` (~1.15 GiB totali) |
| **Manifest** | `data/raw/lc0/manifest.json` |
| **Download** | `scripts/download_lc0.py` |
| **Formato** | Protobuf V6, `input_format=1` (classical) — parser in `lc0_parser.py` |
| **Ruolo SARDINE** | Posizioni da self-play forte; **non** feature space Lc0 (usiamo FEN → encoder 716) |
| **Label attuale** | ❌ nessuna colonna `expected_reward` — solo metadati Lc0 (`best_q`, `root_q`, `result_q`, …) |
| **Nota** | `best_q`/`root_q` nel chunk sono del net che ha generato il training data, **non** del teacher BT4 scelto → labeling uniforme via `lc0` UCI |

### 2. Lc0 processed (pilot) — **pronto per labeling**

| | |
|---|---|
| **Path** | `data/processed/lc0/` |
| **File** | `positions.parquet`, `splits/train.parquet` (1 063 righe), `splits/val.parquet` (60 righe) |
| **Pipeline** | `scripts/prepare_lc0_dataset.py`, `stats_lc0_processed.py` |
| **Campi** | FEN, `bucket_id`, ply, `plies_left`, visits, shard |
| **Pilot stats** | 120 chunk scansionati · 6 038 posizioni valide · 3 149 dopo filtro ply/bucket |
| **Filtri** | `min_ply=32` (16 mosse), bucket 7 relax `min_ply=8` |
| **Prossimo step** | `scripts/label_positions.py` → colonna `expected_reward` via Lc0 BT4 |

### 3. Kaggle Lichess sample — **smoke / statistiche bucket**

| | |
|---|---|
| **Path** | `data/raw/games.csv` (~7.3 MB, ~20k partite) |
| **Download** | `scripts/download_data.py` (Kaggle `datasnaek/chess`) |
| **Ruolo SARDINE** | Distribuzione piece-count / bucket (`excel/piece_count_distribution_10k.xlsx`); **non** per training NNUE |
| **Label** | Solo outcome partita (`white`, `black`, `draw`) — **non** expected reward per posizione |

### 4. Analisi bucket (derivato)

| | |
|---|---|
| **Path** | `excel/piece_count_distribution.xlsx`, `excel/piece_count_distribution_10k.xlsx` |
| **Script** | `scripts/plot_piece_count_distribution.py` |
| **Ruolo** | Progettazione 8 bucket queen-split nel blueprint |

---

## Dataset pre-etichettati — survey

Cercati dump già pronti (FEN + WDL / expected reward) compatibili con encoder 716 SARDINE.

| Candidato | Esito | Note |
|-----------|-------|------|
| Lc0 training chunks (`best_q`, `result_q`) | ⚠️ parziale | Q/WDL nel record, ma legati al net di generazione; non sostituiscono teacher BT4 uniforme |
| Lichess + label community | ❓ da esplorare | Nessun dump compatibile ancora in repo; cercare mirror HF/Kaggle “lichess positions wdl” |
| Stockfish NNUE `.binpack` training sets | ❌ formato diverso | HalfKP / centipawn — non allineato a 716-dim + expected reward |
| nnue-pytorch example data | ❌ riferimento only | Stockfish centipawn, architettura diversa |

**Conclusione:** nessun dataset pre-etichettato riutilizzabile end-to-end. Pipeline custom: campiona FEN → `lc0` BT4 → `expected_reward`.

---

## Dataset futuri (pianificati)

### Priorità alta — diversità posizioni (blueprint: Lichess primary)

| Fonte | URL / accesso | Ruolo | Note |
|-------|---------------|-------|------|
| **Lichess monthly PGN** | [database.lichess.org](https://database.lichess.org/) | Posizioni principali da partite umane | Filtrare rating/time control; campionare FEN ogni N ply; label con Lc0 BT4 |
| **Lichess puzzle DB** | export Lichess | Tattica / posizioni critiche | Utile come boost bucket medi; volume limitato |
| **Lc0 shard aggiuntivi** | [storage.lczero.org](https://storage.lczero.org/files/training_data/) | Volume + copertura fasi | Estendere oltre i 2 tar attuali; stesso preprocess |

### Priorità media — qualità / validazione

| Fonte | Ruolo | Note |
|-------|-------|------|
| **Syzygy TB positions** | Validazione teacher + sanity eval | Outcome noto; pochi GB, non per training massivo |
| **Cutechess / engine self-play** | Test Elo gate | Generato runtime, non archivio training |
| **Pre-labeled dumps** (se trovati) | Risparmio tempo labeling | Criterio: FEN + scalar in `[-1,1]` o WDL; convertibile a `expected_reward` |

### Esclusi v1

| Fonte | Motivo |
|-------|--------|
| Kaggle `games.csv` | Outcome partita, non eval per posizione; partite deboli |
| HF policy nets (Marvin, chess-bot) | Policy-first; non teacher value uniforme |
| Stockfish `.epd` centipawn sets | Target centipawn, non expected reward |

---

## Layout directory (target)

```
data/
├── raw/
│   ├── lc0/           # chunk .gz + manifest          ✅
│   ├── games.csv      # Kaggle smoke                  ✅
│   └── lichess/       # PGN monthly (futuro)          ❌
├── processed/
│   ├── lc0/           # parquet + splits + labels     🔄 pilot senza label
│   └── lichess/       # FEN campionati + labels       ❌
└── labeled/           # merge train/val unificato     ❌ (post label_positions.py)
```

---

## Prossimi script

| Script | Stato | Scopo |
|--------|-------|-------|
| `download_lc0.py` | ✅ | Incrementale chunk Lc0 |
| `prepare_lc0_dataset.py` | ✅ | FEN + bucket da chunk |
| `download_lichess.py` | ❌ | PGN monthly → FEN parquet |
| `label_positions.py` | ❌ | FEN → `expected_reward` via `lc0` BT4 UCI |
| `merge_training_sets.py` | ❌ | Lichess + Lc0, bucket-stratified export nnue-pytorch |