
## Ideas

- AlphaZero-style Dual head (policy + value net, all in one)? Maybe too heavy...
    
- **NNUE** - connect the NN with small pre-computed patterns to reduce network complexity. Standard: dual-perspective, single hidden layer NNUE with 768 inputs
    
- i transformer possono risultare più **compatti** perché spesso usano i parametri in modo più efficiente rispetto ad alternative dense o molto profonde, specialmente quando il compito ha struttura forte e ripetitiva come gli scacchi
    
- **C** instead of C++ (si può fare senza problemi)
    
- Accumulators, lazy loading: When a piece moves from `e2` to `e4`, the engine simply takes the cached hidden layer state, subtracts the weights for the piece on `e2`, and adds the weights for the piece on `e4`
    
- maintain two separate accumulators with vertically flipped perspectives (see NNUE)
    
- Horizontal King Mirroring: When the friendly king moves across the vertical centerline, the network perspective flips horizontally (This trick safely reduced the number of active king inputs by half, maximizing structural compressibility.)
    
- maybe we come up with a simplified representation that minimizes the degrees of freedom, both for input and output
	- In practice probably doesn't make a big difference
    
- Would it be more efficient to have two NNs for the policy, one to select the piece to move, and one to decide where to move the selected piece? (only 64 * 2 output params)
	- Answer: NO. It seems like splitting a 4,000-output network into a 16-output and a 64-output network saves compute. However, in a neural network, 99% of the FLOPs and memory bandwidth are consumed by the **hidden layers** (the convolutions or transformer blocks that process the board state). The final linear layer (the output head) is computationally trivial. Splitting the output head saves almost zero actual compute, but forces you to run the heavy hidden layers twice (or manage a complex shared backbone). You also break the gradient flow between the "what" and the "where".
    
- Leela Chess Zero data: high-quality games
    
- **Dynamic Output Buckets:** Implement distinct output buckets selected dynamically based on the game state's piece count. This allows the network to specialize its evaluation between dense middlegames and sparse endgames.
  My twist: 8 buckets;
  $$\text{Bucket ID} = \text{int} \left( \frac{34 - \text{piece\_count}}{4} \right)$$
  (and hard-coded draw with 2 pieces)
    
- Cfish: stockfish but in C
    
- Pruning: eliminate weights corresponding to useless inputs (like pawns on ranks 1-8). 704 feature inputs (pruning impossible states) 
    
- SPSA for hyperparameter optimization?
      
- Neural Training with Grapheus?
    
- Quantize the NNUE networks
    
- Search: stripped-down, lightning-fast alpha-beta search with aggressive Late Move Reductions (LMR) and Null Move Pruning (NMP) calibrated specifically to compensate for the smaller network's lower evaluation fidelity.