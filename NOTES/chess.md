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

## Quiescence Search

#### The problem it solves: the "Horizon Effect"

Imagine your engine searches to depth 4 and evaluates a position. At depth 4, it sees a knight hanging (undefended). The engine thinks "great, I can capture it!" — but it doesn't realize that **at depth 5**, the opponent has a devastating queen sacrifice that leads to checkmate.

The engine's "horizon" is the maximum depth it can see. Anything beyond it is invisible. This is the **horizon effect**: the engine makes catastrophically bad decisions because it stops searching right before something important happens.

#### The solution: extend the search until the position is "quiet"

**Quiescence search** is a secondary search that runs **only at the leaf nodes** of the main alpha-beta tree. It keeps searching until the position has no more "noisy" moves — meaning:

- No more captures
- No more checks
- No more promotions

Once the position is "quiet" (quiescent), it's safe to evaluate.

---

## Perft (Performance Test)

#### What it is

**Perft** (short for "performance test") is a debugging and benchmarking tool for your **move generator**. It counts the total number of leaf nodes at a given depth, **without any evaluation** — just pure move generation.

#### Why it exists

Writing a correct chess move generator is **surprisingly hard**. You have to handle:

- Piece movements (each piece has its own rules)
- Castling (with 5+ edge cases: king/rook moved, in check, through check, to check, path blocked)
- En passant (only available immediately after a double pawn push)
- Pawn promotion (4 choices per promotion square)
- Pin detection (can't move a pinned piece if it exposes the king)
- Check evasion (must get out of check)

A single bug in any of these can cause illegal moves or miss legal ones. **Perft gives you a definitive test**: if your perft numbers match known reference values, your move generator is correct.

---

[← Back to Notes index](_notes.md)
