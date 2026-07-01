
## Ideas

- AlphaZero-style Dual head (policy + value net, all in one)? 
	- Maybe too heavy...
    
- **NNUE** - connect the NN with small pre-computed patterns to reduce network complexity. Standard: dual-perspective, single hidden layer NNUE with 768 inputs
    
- maintain two separate accumulators with vertically flipped perspectives (see NNUE)
    
- no Hand-Crafted Evaluation (HCE) 
    
- **C** instead of C++ (si può fare senza problemi) Cfish: stockfish but in C. Probably a must. Most competitors used Cfish to great success.
    
- i transformer possono risultare più **compatti** perché spesso usano i parametri in modo più efficiente rispetto ad alternative dense o molto profonde, specialmente quando il compito ha struttura forte e ripetitiva come gli scacchi
    
- Accumulators, lazy loading: When a piece moves from `e2` to `e4`, the engine simply takes the cached hidden layer state, subtracts the weights for the piece on `e2`, and adds the weights for the piece on `e4`
    
- Horizontal King Mirroring: When the friendly king moves across the vertical centerline, the network perspective flips horizontally (This trick safely reduced the number of active king inputs by half, maximizing structural compressibility.)
    
- maybe we come up with a simplified representation that minimizes the degrees of freedom, both for input and output
	- In practice probably doesn't make a big difference
    
- Would it be more efficient to have two NNs for the policy, one to select the piece to move, and one to decide where to move the selected piece? (only 64 * 2 output params)
	- Answer: NO. It seems like splitting a 4,000-output network into a 16-output and a 64-output network saves compute. However, in a neural network, 99% of the FLOPs and memory bandwidth are consumed by the **hidden layers** (the convolutions or transformer blocks that process the board state). The final linear layer (the output head) is computationally trivial. Splitting the output head saves almost zero actual compute, but forces you to run the heavy hidden layers twice (or manage a complex shared backbone). You also break the gradient flow between the "what" and the "where".
    
- Leela Chess Zero data: high-quality games
    
- **Dynamic Output Buckets:** Implement distinct output buckets selected dynamically based on the game state's piece count. This allows the network to specialize its evaluation between dense middlegames and sparse endgames.
  
  My twist: Each bucket contains a similar number of board positions from the dataset.

| Bucket ID | piece count |
| --------- | ----------- |
| 0         | 2-12        |
| 1         | 13-17       |
| 2         | 18-21       |
| 3         | 22-24       |
| 4         | 25-27       |
| 5         | 28-29       |
| 6         | 30-31       |
| 7         | 32          |

- Pruning: eliminate weights corresponding to useless inputs (like pawns on ranks 1-8). 704 feature inputs (pruning impossible states) 
    
- SPSA for hyperparameter optimization?
      
- Neural Training with Grapheus?
    
- Quantize the NNUE networks
    
- Search: stripped-down, lightning-fast alpha-beta search with aggressive Late Move Reductions (LMR) and Null Move Pruning (NMP) calibrated specifically to compensate for the smaller network's lower evaluation fidelity.
    
- MOE: Different network based on the nature of the position? (king under attack, imminent high-value piece capture)
	  Note: switching heads may reduce accumulator efficacy, but as long as the number of pieces goes always down you have to switch heads unfrequently
	
- `O3` speed optimizations??

---

## See Also

### "Dog" (NNUE su ESP32)

Questa è probabilmente la scoperta più rilevante per te. Uno sviluppatore (Folkert van Heusden) ha creato **Dog**, un motore scacchistico progettato specificamente per girare su un microcontrollore ESP32.

- **Cosa lo rende speciale:** A differenza dei motori classici, _Dog implementa effettivamente una rete neurale NNUE_ a bordo del microcontrollore, oltre a un book di aperture. L'autore ha dovuto far entrare tutto (inclusa la Transposition Table) nei limitati 320 KB di RAM dell'ESP32.
    
- **Performance:** Su PC raggiunge circa 2400 Elo, mentre limitato dall'hardware dell'ESP32 gioca con una forza stimata di poco inferiore (intorno ai 2000+ Elo).
    
- **Perché ti è utile:** Dimostra che il tuo obiettivo (un TinyML per scacchi su un chip con ~200KB di RAM) è assolutamente fattibile e competitivo. L'autore lo ha anche interfacciato in modalità UCI per farlo giocare online su Lichess direttamente dal chip.
    

### "MicroChess"

Se la sfida di Kaggle imponeva un limite di 5 MB di RAM, questo progetto su GitHub porta il concetto di "vincolo" alla follia.

- **Cosa lo rende speciale:** È un motore scritto in C/C++ in grado di girare con **meno di 2 KB di RAM** e 32 KB di memoria Flash (pensato per gli Arduino Uno/Nano più piccoli).
    
- **Tecniche:** Ovviamente non usa reti neurali, ma implementa regole complete (inclusi en passant, arrocco, promozione) e usa una tecnica geniale chiamata _Stack Surfing_. Invece di avere una profondità di ricerca fissa, il codice controlla a runtime quanta memoria Stack è rimasta libera nel microcontrollore; se c'è spazio, spinge l'albero di ricerca un "ply" (semimossa) più in profondità.
    
- **Perché ti è utile:** È il benchmark definitivo per capire come scrivere codice C _bare-metal_ super efficiente. Se vuoi implementare l'Alpha-Beta sul Wio Terminal senza sprecare byte, il codice sorgente di MicroChess è un'enciclopedia dell'ottimizzazione.
    

###  "ESP32 Chess Engine" (di Sergey Urusov)

Questo progetto è un'evoluzione di un vecchio motore per Arduino Mega, riscritto per sfruttare i microcontrollori moderni.

- **Cosa lo rende speciale:** Non usa Bitboards e non usa reti neurali, ma sfrutta pesantemente le euristiche classiche che discutevamo ieri: _Null Move Pruning_, _Killer Heuristic_, _Futility Pruning_ e _Lazy Evaluation_.
    
- **Performance:** Riesce a valutare circa **20.000 nodi al secondo (20 kNps)** sull'ESP32, risolvendo la quasi totalità dei test tattici "Win At Chess" (WAC) in meno di 1 minuto, raggiungendo i 2023 Elo (stimati su Elometer).
    
- **Perché ti è utile:** Nel tuo stress-test, hai calcolato 2.16 milioni di _forward passes_ al secondo per la sola rete neurale. Confrontando i tuoi numeri con i 20.000 nodi al secondo di Urusov (che includono l'overhead della generazione mosse e della tree search), puoi iniziare a fare delle stime matematiche su quanti nodi effettivi riuscirà a esplorare il tuo Wio Terminal una volta unito il tuo Mixture of Experts all'algoritmo di ricerca.
    

---

