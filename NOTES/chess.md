# Chess Notes ♟

> *Everything Chess-related...*

### NNUE

See the dedicated note: [NNUE.md](NNUE.md) — architecture, incremental updates, quantization, and relevance to the Wio project.

Quick summary: [NNUE](https://www.chessprogramming.org/NNUE) (Efficiently Updatable Neural Network) replaces hand-crafted evaluation in alpha-beta engines with a compact net that updates incrementally via add/sub on sparse 768-feature inputs. Stockfish adopted it in 2020.

---

### FEN string

FEN (Forsyth-Edwards Notation) is a standard string representation of a chess position. 
It has 6 fields, separated by spaces:

$$
\underbrace{\text{rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR}}_{\substack{\textbf{Piece Placement} \\ \text{(Ranks 8 to 1)}}}
\; ... 
$$
$$
...
\;
  \underbrace{\text{b}}_{\substack{\textbf{Active Color} \\ \text{(w or b)}}}
  \;
  \underbrace{\text{KQkq}}_{\substack{\textbf{Castling} \\ \text{Availability}}}
  \;
  \underbrace{\text{e6}}_{\substack{\textbf{En Passant} \\ \text{Target}}}
  \;
  \underbrace{\text{0}}_{\substack{\textbf{Halfmove} \\ \text{Clock}}}
  \;
  \underbrace{\text{2}}_{\substack{\textbf{Fullmove} \\ \text{Number}}}
$$

---

### Pawn Promotion

- Create promotion move: `chess.Move.from_uci("e7e8q")` 
- 5th char = piece (q,r,b,n)
- Check if promotion: `if move.promotion: ...`
- The standard 4096-output head doesn't distinguish promotions -> default to Queen

---

## UCI

Il protocollo **UCI** (Universal Chess Interface) è uno standard di comunicazione aperto che consente ai motori scacchistici (come [Stockfish](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-&-Commands.html)) di dialogare con interfacce grafiche (GUI). [1](https://it.wikipedia.org/wiki/Universal_Chess_Interface)

Le interfacce grafiche fungono da controllore e mantengono lo stato del gioco, mentre il motore si occupa esclusivamente di calcolare le mosse tramite istruzioni testuali . [1](https://www.chessprogramming.org/UCI)

---

## Stack Surfing

Invece di avere una profondità di ricerca fissa, il codice controlla a runtime quanta memoria Stack è rimasta libera nel microcontrollore; se c'è spazio, spinge l'albero di ricerca un "ply" (semi-mossa) più in profondità.

---

[← Back to Notes index](_notes.md)
