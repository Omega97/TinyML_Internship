
# Cfish: Overview & Guide

## What is Cfish?

**Cfish** is a direct port of the world-renowned **Stockfish** chess engine written entirely in **pure C** (originally created by Ronald de Man). While the official Stockfish project is written in modern C++, Cfish reimplements its core algorithms in standard C to reduce executable overhead, improve compilation flexibility, and explore performance micro-optimizations across different C compilers.

Because C binaries often have a smaller footprint and simpler dependencies than C++ binaries, Cfish is widely favored in resource-constrained environments—such as custom chess engine research, hardware-limited microcontrollers, and micro-competitions (like the _FIDE & Google Efficient Chess AI Challenge_).

## How It Works

Cfish mirrors Stockfish’s architectural pipeline while taking advantage of low-level C features:

### 1. Engine Structure & Search

- **Alpha-Beta Search:** Uses principal variation search (PVS) combined with aggressive pruning techniques (null move pruning, late move reductions, futility pruning).
    
- **Bitboards:** Represents the board state using 64-bit integer bitmasks for lightning-fast piece movement and attack pattern computations.
    
- **Transposition Tables:** Uses a global hash table to cache previously evaluated board states and branch evaluations.
    

### 2. Evaluation Subsystem

- **NNUE Support:** Like Stockfish, Cfish supports **Efficiently Updatable Neural Networks (NNUE)**. It can process compressed neural network weights (`.nnue` files) using SIMD instruction sets (AVX2, AVX-512, NEON) for evaluation.
    
- **Hand-Crafted Evaluation (HCE):** In stripped-down custom builds, NNUE can be replaced with lightweight classical piece-square tables and heuristic positional terms to save memory.
    

### 3. C-Specific Optimizations

- **Direct Memory Allocation:** Avoids C++ standard library abstractions (`std::vector`, `std::string`) in favor of lightweight structs and raw pointers.
    
- **SIMD & Intrinsics:** Features low-level vectorized bitboard operations for move generation and neural net forward passes.
    

## How to Build and Use Cfish

### 1. Building from Source

To compile Cfish, you need a standard C compiler (`gcc` or `clang`) and `make`.

Bash

```
# Clone the repository
git clone https://github.com/syzygy1/Cfish.git
cd Cfish/src

# Build for modern x86 architectures (AVX2 optimization)
make -j ARCH=x86-64-avx2

# For ARM / Embedded devices
make -j ARCH=armv8
```

### 2. Basic UCI Command Line Usage

Cfish communicates using the standard **Universal Chess Interface (UCI)** protocol via standard input/output (`stdin`/`stdout`).

#### Start the engine:

Bash

```
./cfish
```

#### Basic UCI Commands:

Plaintext

```
# 1. Initialize UCI protocol
uci

# 2. Configure Hash Memory (e.g., set to 64MB)
setoption name Hash value 64

# 3. Load NNUE Weights File (optional, if using external weights)
setoption name EvalFile value net.nnue

# 4. Prepare engine for a new game
isready
ucinewgame

# 5. Set up a board position (Starting position or FEN string)
position startpos moves e2e4 e7e5

# 6. Start calculation/search
go depth 15
# Or search with time control (movetime in milliseconds):
go movetime 2000
```

### 3. Interfacing via Python (`python-chess`)

You can automate and interact with Cfish programmatically using the popular `python-chess` library:

Python

```python
import chess
import chess.engine

# Start the Cfish subprocess
engine = chess.engine.SimpleEngine.popen_uci("./cfish")

# Create a standard chess board
board = chess.Board()

# Evaluate position and get the best move
result = engine.play(board, chess.engine.Limit(time=1.0))
print(f"Best Move: {result.move}")

# Inspect evaluation score
info = engine.analyse(board, chess.engine.Limit(depth=12))
print(f"Score: {info['score'].relative}")

# Terminate process
engine.quit()
```

---

## ♟️ Cheat Sheet Comandi UCI per Cfish

### 1. Inizializzazione e Opzioni

All'avvio, è buona norma inizializzare il motore e impostare la memoria o i thread prima di iniziare l'analisi.

Plaintext

```
uci
```

> Inizializza il motore. Cfish risponderà restituendo le sue opzioni e la conferma `uciok`.

Plaintext

```
setoption name Hash value 1024
```

> Imposta la Hash Table (RAM allocata per la Transposition Table) a 1024 MB (default solitamente 16 MB).

Plaintext

```
setoption name Threads value 4
```

> Imposta il numero di thread di ricerca da utilizzare.

Plaintext

```
isready
```

> Verifica che il motore abbia applicato le opzioni e sia pronto. Risponderà con `readyok`.

Plaintext

```
ucinewgame
```

> Informa il motore che sta per iniziare un'analisi da zero (pulisce la Transposition Table). Da lanciare prima di inserire una nuova posizione.

### 2. Inserimento della Posizione (`position`)

#### Posizione Iniziale (Startpos)

Plaintext

```
position startpos
```

> Imposta la scacchiera sulla posizione di partenza standard.

Plaintext

```
position startpos moves e2e4 e7e5 g1f3
```

> Imposta la posizione iniziale e applica la sequenza di mosse specificata (in notazione algebrica coordinata LAN: casa_partenza casa_arrivo).

#### Posizione tramite FEN

Plaintext

```
position fen r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
```

> Imposta una posizione arbitraria passando direttamente la stringa FEN.

Plaintext

```
position fen r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 moves d2d4 e5d4
```

> Imposta la FEN ed esegue mosse successive a partire da quella posizione.

### 3. Esecuzione della Ricerca (`go`)

#### Ricerca a Tempo o Profondità Fissa

Plaintext

```
go depth 20
```

> Analizza la posizione fino alla profondità specificata (es. 20 semi-mosse / ply).

Plaintext

```
go movetime 5000
```

> Analizza per un tempo fisso espresso in millisecondi (es. 5000 ms = 5 secondi).

Plaintext

```
go nodes 10000000
```

> Interrompe la ricerca dopo aver analizzato esattamente un determinato numero di nodi.

#### Ricerca Infinita e Stop Manuale

Plaintext

```
go infinite
```

> Avvia la ricerca a tempo indeterminato. Il motore continuerà finché non invii manualmente il comando di stop.

Plaintext

```
stop
```

> Ferma immediatamente la ricerca in corso. Cfish restituirà l'ultima valutazione e la mossa migliore trovata (`bestmove`).

#### Ricerca per Partita (Time Control)

Plaintext

```
go wtime 300000 btime 300000 winc 2000 binc 2000
```

> Simula il controllo del tempo di una partita reale: tempo rimanente in ms per Bianco (`wtime`) e Nero (`btime`), con relativo incremento per mossa (`winc`/`binc`).

### 4. Gestione Output e Chiusura

- **Lettura delle info di ricerca:**
    
    Durante il comando `go`, Cfish stamperà righe come:
    
    `info depth 18 score cp 45 nodes 1254300 nps 2100000 pv e2e4 e7e5`
    
    - **`depth`**: Profondità raggiunta.
        
    - **`score cp 45`**: Valutazione della posizione in centipiedini ($+0.45$ in favore del giocatore di turno). Se c'è un matto imminente mostrerà `score mate X`.
        
    - **`nps`**: Nodi al secondo (Nodes Per Second).
        
    - **`pv`**: Principal Variation (la sequenza di mosse migliore calcolata finora).
        
- **`bestmove e2e4`**: Viene stampato alla fine della ricerca per indicare la mossa ottimale consigliata.
    

Plaintext

```
quit
```

> Chiude ed esce dal processo di Cfish.

### 💡 Esempio pratico di sessione rapida

Per testare subito Cfish da riga di comando:

Plaintext

```
uci
isready
ucinewgame
position fen r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4
go depth 15
quit
```
