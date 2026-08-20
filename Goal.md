
# Chess bot project


> *Building a small and efficient chess bot, that can run on extremely limited hardware. The following is a clear description of the steps to complete the project.*

---

## 1 - Building the dataset

- [ ] Download many games from the web ($\gtrsim{10^6}$ board positions from bot games, human games, puzzles, ...).
    
- [ ] Download a strong *value function* (possibly one that returns the expected reward from any given state, like *Lc0*). The stronger the better, but remember that you will have to run it once on every entry of the dataset.
    
- [ ] Run the *value function* on the downloaded positions to create a JSON fen-value-visits, where you have:
	- the **FEN** code of the position, 
	- the **value** $\hat v \in [-1, +1]$ given by the value function (the expected value of the white player, for simplicity), and 
	- the number of **visits** of the position (how many times it was found in the dataset).

```json
[
  {
    "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "value": 0.037,
    "visits": 30
  },
  {
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "value": 0.0,
    "visits": 63
  }
]
```

---

## 2 - Training the model

- [ ] Train an **NNUE** $f_w(s)=v$ with **sparse input**, two hidden layers, *CReLU*, *tanh* output:
	- **Input**: the board state, sparse representation.
	- **L1**: a **shared** FFNN `844 → W` (sparse, approx. 128x2 neurons, int8 weights $w^{(1)}$). Called **twice** per position: once on the **own‑side features**, once on the **opponent‑side features** (board mirrored). The two output vectors (`h_own` and `h_opp`, each of size `W`) form the **dual‑POV accumulator** (the activations are recycled via incremental add/sub on make/unmake). They are **concatenated** to size `2W` **after** L1, just before the expert head. Both sets of activations are used as a single embedding vector $h \equiv a^{(1)}$.
	- **L2**: second hidden layer (approx. 256 neurons, param $w^{(2)}$).
	- **Output**: One single neuron, represents the expected reward (proba white win - proba white lose) of the position (param $w^{(3)}$).

---

## 3 - Fine-tuning

- [ ] **Embeddings**: A forward pass of the NNUE on the database will provide the embedding $h$ for each position.
    
- [ ] **Task vectors**: Compute the *task vector* $\delta = \nabla_{w^{(head)}} \; \mathcal L_{acc}\,(f_w (s), \hat v)$ for each position, where $\hat v$ is the value of the position estimated by the *teacher*, $w=(w^{(1)}, w^{(2)}, w^{(3)})$ are all the parameters of the model, and $w^{(head)} = (w^{(2)}, w^{(3)})$ are the parameters of the expert head. These vectors tell you how the model would like to adapt to learn each position-value pair.
    
- [ ] **Clustering**: Apply a clustering algorithm (like *k-Means*) with $B$ clusters on the normalized task vectors $\hat \delta$ to obtain the labels $b_i\in\{1,\ldots,B\}$. Each cluster groups together similar *task vectors*, making it easier for the expert heads to learn them.
    
- [ ] **Dispatcher**: Train a minimal model $g_\phi​(h)=\text{softmax}(W_\phi\, h​)$ to predict the class $b$ based on the L1 activations $h$ (the bias component is implied). _The dispatcher is not used during training of the experts — it is only for inference routing._
    
- [ ] **Fine-tuning**: The board positions are clustered based on the model's needs rather than the representation of the states (or the states themselves, for that matter). Freeze the $w^{(1)}$ weights, and run a final fine-tuning step on each bucket of states, starting from the initial model $f_w$. This step will yield a model $f^{(i)}_{w'}$ that should outperform the original model.

---

## 4 - Inference


- [ ] The eval (forward) step:
	1. Run L1. The accumulator activations $h \equiv a^{(1)}$ are the representation of the state.
	2. Route with dispatcher logits only: $b = \arg\max(W_\phi\, h)$ (no softmax).
	3. Run expert head $i=b$ (L2 + output) on $h$ to get $v$.
    
- [ ] Recycle L1 in the spirit of NNUE: on make/unmake, add/remove the moved piece's feature contributions on both POVs instead of re-encoding the whole board. Full re-encode only on king-mirror or irreversible resets. Then only L2 + output (and the tiny dispatcher) run at the node.
    
- [ ] **Search = Cfish αβ, student eval.** Keep Cfish's search stack as-is (PVS αβ, quiescence, transposition table, move ordering). Do not reimplement search. Replace *only* the value function that `evaluate(pos)` / `nnue_evaluate(pos)` returns:
	- **Vanilla baseline:** plug the base net $f_w$ into that hook.
	- **MoE:** same hook, but eval is dispatcher $b=\arg\max(W_\phi h)$ then expert $f^{(b)}_{w'}$.
	- Map $v \in [-1,+1]$ onto Cfish's internal `Value` (centipawn-like) so comparisons, mate scores, and qsearch still work.
	- Keep Cfish's incremental NNUE accumulators: expose L1 as the same add/remove-feature update Cfish already uses on make/unmake.
	- Drive the engine with ordinary UCI (`position …`, `go depth …` / `go movetime …`), same as stock Cfish.

---

## 5 - Evaluation

Evaluate performance using three complementary methods, each serving a different purpose. 

- **ACPL analysis** measures move quality via centipawn loss against Stockfish, providing a fast heuristic Elo estimate from self-play games. 
    
- **Strategic test suites** (e.g., STS) run the engine on thousands of positions with known best moves, yielding a reproducible score that correlates with strategic understanding. 
    
- For definitive comparison, we run a **BayesElo match** between the vanilla (pre-fine-tuning) and MOE (*Mixture Of Experts*) models under tournament conditions. 

The MOE model is considered successful if it consistently outperforms the vanilla baseline across all three instruments, with particular emphasis on the match Elo gain. This multi‑instrument approach avoids reliance on any single metric and provides a robust assessment of the MOE improvement.

---
