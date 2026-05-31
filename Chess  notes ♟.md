
# Chess notes ♟


> *Everything Chess-related...*


### NNUE

[NNUE](https://www.chessprogramming.org/NNUE) (Efficiently Updatable Neural Network) is 
a neural network architecture specifically designed for board game engines running on CPUs. 
Originally invented for Shogi by Yu Nasu in 2018 and later popularized in chess by Stockfish 
in 2020, NNUE revolutionized traditional alpha-beta engines by replacing hand-crafted 
evaluation functions with a compact neural network while maintaining extremely high speed.

[Basic NNUE](https://www.chessprogramming.org/NNUE#Basic_NNUE): The most basic form 
of an NNUE network consists of three layers: an input layer of length 768 
(768 = 6 pieces x 2 colors x 64 squares), one hidden layer of arbitrary size, and 
an output layer consisting of one neuron, representing the evaluation of the position. 
A NNUE network also commonly consists of two perspectives. That is, two hidden layers 
representing both sides are concatenated into a single hidden layer of twice the length, 
before being forwarded to the output layer. 

Read also: [NNUE Stockfish](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)

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
