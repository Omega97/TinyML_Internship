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

[← Back to Notes index](_notes.md)
