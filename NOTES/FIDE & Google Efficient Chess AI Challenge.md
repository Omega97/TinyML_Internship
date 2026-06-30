
# FIDE & Google Efficient Chess AI Challenge


The absolute cutting edge of "small hardware" chess architectures, the FIDE & Google Efficient Chess AI Challenge hosted on [Kaggle](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge). This competition explicitly forbids brute-force computation and forces participants to build engines under extreme hardware constraints, such as allocating a maximum of just 5MiB of RAM
	- **[FIDE and Google create the Efficient Chess AI Challenge (FIDE.com)](https://www.fide.com/fide-and-google-create-the-efficient-chess-ai-challenge-hosted-on-kaggle/)**: The official announcement challenging developers to create smart, resource-light chess programs
	- **[FIDE & Google Efficient Chess AI Challenge (Kaggle)](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/557921)**: The competition forums and discussion boards are a goldmine for seeing the exact lightweight architectures (like heavily pruned NNUE variants or tiny custom networks) that top competitors used to maximize Elo per byte
	- [Discussion](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion?sort=undefined)
	- [Leaderboard](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/leaderboard)

---

## The Core Architectures: NNUE vs. HCE

Going into the competition, many participants assumed that NNUEs (Efficiently Updatable Neural Networks) would be too heavy to fit into a 64 KiB compressed binary. However, the top three solutions completely upended this assumption by designing highly customized, micro-sized NNUEs.

Engines that relied strictly on Hand-Crafted Evaluation (HCE) still managed top-10 finishes by squeezing every ounce of search efficiency out of traditional chess heuristics, but NNUEs ultimately dominated the absolute top of the leaderboard.

---

## Breakthroughs from the Top Solutions

> *Cfish is a good start. C is more efficient than C++. Leverage symmetry, cut the unnecessary parts.*

### 🥇 1st Place: `Cfish` by linrock

- **Writeup:** [My solution: Cfish, nnue, data (1st)](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/linrock-my-solution-cfish-nnue-data-1st)
    
- **Source Code:** [GitHub - linrock/minifish](https://github.com/linrock/minifish)

Linrock realized that under severe space constraints, the strength of the engine is better stored in highly optimized network weights rather than complex search code.

- **The Foundation:** Built on **Cfish** (a discontinued C port of Stockfish). Cfish was the engine of choice for almost all top competitors because its `glibc` memory consumption overhead is significantly lower than C++ implementations, leaving more of the precious 5 MiB RAM available for the search tree and transposition tables.
    
- **The Symmetrical Network:** Used a standard dual-perspective, single hidden layer NNUE with 768 inputs. To make it fit within 64 KiB, linrock applied **Horizontal King Mirroring**. When the friendly king moves across the vertical centerline, the network perspective flips horizontally. This trick safely reduced the number of active king inputs by half, maximizing structural compressibility.
    
- **Zeroing Unused Weights:** Since pawns can never legally occupy the 1st or 8th ranks, the weights corresponding to those states were hard-zeroed out, allowing compression algorithms to shrink the binary dramatically without adding complex masking code.
    
- **Stochastic Data Filtering:** Squeezed massive Elo out of a tailored training pipeline using Leela Chess Zero data. Linrock built a custom dataloader that flattened the piece-count distribution curve (removing the overrepresented 32-piece starting layouts), skipped the first 28 plies of games, and stochastically retained positions where tactical piece sacrifices were optimal.
    

### 🥈 2nd Place: `Approvers` by Shahin M. Shahin

> *Dynamic Output Buckets are an interesting approach.*

**Writeup:** [Approvers Submission (2nd)](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/approvers-approvers-submission-2nd)

The runner-up solution proved that an aggressive structural network compression pipeline can squeeze a fully multi-bucket evaluation model into a microscopic footprint.

- **Architecture:** Implemented a `(768x1hm -> 64)x2 -> 1x8` pattern featuring a single hidden layer utilizing **SCReLU** (Squared Clipped Rectified Linear Unit) activations: $f(x) = \min(\max(x, 0), 1)^2$.
    
- **Dynamic Output Buckets:** Implemented 8 distinct output buckets selected dynamically based on the game state's piece count:
    $$\text{Bucket ID} = \frac{\text{piece\_count} - 2}{4}$$
    
    This allowed the network to specialize its evaluation between dense middlegames and sparse endgames.
    
- **Extreme Pruning & Precision:** Feature inputs were pruned down to 704 by wiping out impossible pawn placements and mirrored king coordinates. Feature Transformer (FT) and Layer 1 weights were aggressively quantized to **8-bit integers**, while Layer 1 biases were kept at 16-bit. The entire resulting neural net took up **just 45 KB**.
    

### 🥉 3rd Place: `KaggleFish` by "Fix the bugs?" (Andrew Grant & Kim Georg Kåhre)

> *Professional, check their tools.*

- **Writeup:** ["Fix the bugs?" Solution Write-up (3rd)](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/fix-the-bugs-fix-the-bugs-solution-write-up-3rd)
    
- **Source Code:** [GitHub - AndyGrant/KaggleFish](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FAndyGrant%2FKaggleFish)

Led by Andrew Grant (the author of Chess.com's ==Torch== engine), this team heavily leveraged **enterprise-grade optimization frameworks** to maximize execution performance.

- **SPSA & OpenBench:** Rather than manually guessing positional heuristics, the team utilized **SPSA** (Simultaneous Perturbation Stochastic Approximation) distributed across massive clusters. By simultaneously perturbing hundreds of evaluation and search hyperparameters and running millions of fast games, they mathematically ground down the ideal configurations.
    
- **Neural Training with Grapheus:** Employed specialized neural network training configurations explicitly tuned for low-memory chess structures to achieve maximum evaluation stability during deep alpha-beta prunings.


### 4th Place: by nagiss

[Nagiss](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/nagiss-4th-place-solution) took a highly structured, minimalist approach to NNUE design, proving that you do not need a massive code envelope if your neural network architectures utilize precise geometric pruning.

- **The Compact NNUE:** Implemented a highly optimized `768 -> 16 -> 1` network topology. By drastically shrinking the hidden layer size down to just 16 nodes, nagiss was able to fit the network parameters into a microscopic memory footprint while retaining the rich non-linear evaluation advantages of an NNUE over traditional HCE.
    
- **Aggressive Bit-Width Reduction:** Quantized all primary network weights to **8-bit integers (int8)** and biases to **16-bit integers (int16)**. This allowed the full network to sit comfortably within the strict 64 KiB binary limit without relying on heavy external runtime decompression routines.
    
- **Search Modifications:** Focused the remaining code space on a stripped-down, lightning-fast alpha-beta search with aggressive Late Move Reductions (LMR) and Null Move Pruning (NMP) calibrated specifically to compensate for the smaller network's lower evaluation fidelity.


### 5th Place: `A` by Rafbill

[Rafbill](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/a-my-solution-5th-place) maximized raw execution velocity and structural safety by stripping away the traditional features of a monolithic engine and engineering a hyper-optimized, standalone solution from the ground up.

- **Ditching Stockfish/Cfish Heritage:** Unlike most top solutions that modified Cfish or older Stockfish versions, Rafbill engineered an exceptionally streamlined evaluation pipeline engineered purely for single-core CPU execution speed.
    
- **Pure Evaluation Efficiency:** Maximized the efficiency of traditional Hand-Crafted Evaluation (HCE) layers. By avoiding the multi-kilobyte lookup tables required by NNUE feature transformers, the execution pipeline freed up vital CPU instruction caches, allowing the search tree to probe significantly deeper within the exact same time controls.
    
- **Zero-Waste Transposition Table:** Implemented an aggressive memory packing format for the Transposition Table (TT). Each entry was carefully bit-packed to fit inside minimal bytes, maximizing the total node storage slots achievable within the strict 5 MiB RAM constraint and effectively lowering the engine's time-to-depth metrics.


### 7th Place: `Noggenfogger` by Marko Rupnik

[Marko Rupnik](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/566888) successfully bridged the gap between HCE footprint and NNUE performance by conditioning his network parameters on immediate tactical constraints rather than sprawling piece-square combinations.

- **Conditional Activation Architecture:** The solution deployed a unique, specialized quantized network footprint utilizing an `int8` weight configuration scaled explicitly by positional context, such as tracking active check states (`inCheck`) and tactical piece exchanges (`or-capture`).
    
- **Piece-Count Piece-Square Specialization:** Instead of allocating precious binary bytes to a massive fully connected feature transformer, the model utilized low-dimensional, integer-quantized input transformations mapped directly to structural piece counts.
    
- **Highly Targeted Pruning:** By dynamically switching evaluation behaviors depending on whether the friendly king was under direct attack or a high-value piece capture was imminent, the network achieved top-tier defensive and tactical calculation stability while remaining underneath the 64 KiB threshold.


### 9th Place: `Cfish + Simple SPSA` by ymg_aq

The [9th place](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/ymg-aq-9th-place-solution-cfish-simple-spsa) entry proved that traditional Hand-Crafted Evaluation (HCE), when combined with an aggressive automated parameter-tuning suite, can comfortably out-search heavier neural alternatives on constrained hardware.

- **Complete NNUE Eviction:** The author completely excised all NNUE classes, neural layers, and benchmark telemetry (including Bench, TBProbe, PolyBook, and NUMA logic) from a Cfish base, shrinking memory overhead down to a lean **4 MiB base usage**.
    
- **Extreme Table Rescaling:**
    
    - Compressed the Countermove History tables by collapsing contextual branches (like `inCheck` and `CaptureOrPromotion`), cutting branching space by 75%.
        
    - Transformed Countermove History indexing to reference _generic piece types_ instead of specific piece instances across the coordinates, dropping table size by an additional 50%.
        
    - Capped the Material and Pawn Hash tables at exactly 1,024 slots while allocating a full **1 MiB purely to the Transposition Table (TT)**.
        
- **Custom Local SPSA Engine:** Rather than relying on distributed testing clusters like OpenBench, ymg_aq built a lightweight, 400-line custom Python/Perl SPSA script that ran concurrently via `cutechess-cli` on a single local machine. Iteratively testing parameters against a 2,000-opening book with tight 10-second limits netted a massive **+30 Elo** optimization boost.
    
- **Compilation Efficiency:** Compiled using `O3` speed optimizations rather than `Os` space savings by freeing up code space through feature gutting. Aggressive dead-code elimination flags (`-ffunction-sections`, `-fdata-sections`, `--gc-sections`) combined with **UPX --lzma compression** easily brought the highly performant binary under the 64 KiB cap.

### 10th Place: `Niboshi` by c-number, daiwakun, and KawattaTaido

**Writeup:** [Niboshi's solution](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/niboshi-niboshi-s-solution)

This team took a highly experimental approach to the neural network architecture itself, attempting to replace the standard Feature Transformer (FT) with a hybrid CNN/Dense model to save binary space, though they ultimately found the compute trade-off too high.

- **Engine Progression:** They started with Stockfish 4, moved to Stockfish 16, and finally settled on **Cfish** due to strict RAM constraints.
- **Pure NNUE:** They completely disabled Hand-Crafted Evaluation (HCE) and relied 100% on the neural network.
- **Hybrid Network Architecture:** To save size, they replaced the standard FT by concatenating the outputs of two smaller networks:
    1. **A CNN:** 12 input channels (piece types), 13 output channels, a 15x15 kernel, and padding of 7. This allowed the CNN to produce spatial features for every square on the board.
    2. **A Dense Network:** 64 outputs, but it shared weights for neighboring positions. This drastically reduced the input dimensionality from the standard 768 down to just **96 inputs**.
- **The Compute Trap:** They noted that while this saved binary size, the computation cost was incredibly heavy (an equivalent L1 size of 896). The CPU overhead was too large to compensate for the accuracy lost by sharing the CNN weights.
- **Memory Allocation:**
    - Transposition Table (TT): 512 KiB.
    - Continuation History Hash Table: 128 KiB.
- **Compilation & Stripping:** Compiled `nnue.c` with `-O3` (max speed) and the rest of the engine with `-Os` (min size). To fit the 64 KiB limit, they completely gutted unused features, telemetry (bench, trace), and disabled heavy components like the endgame tablebases and opening books.
- **Training Data:** A mixture of Leela Chess Zero and Stockfish-generated data, though they noted changing the dataset ratio didn't significantly impact performance.

---

## High-Yield Optimization Tactics

For developers working on similar resource-constrained environments (like microcontrollers or edge AI devices), the competition solidified a handful of golden rules:

> 💡 **The 5 MiB RAM Gradient:** Top competitors discovered that search efficiency has a tight trade-off curve with memory usage. To maximize the Transposition Table (TT) size within 5 MiB, everything else had to be gutted:
> 
> - **TT Truncation:** Transposition tables were slashed down to 512 KiB – 1 MiB (compared to gigabytes in desktop environments).
>     
> - **History Merging:** Countermove and continuation history table dimensions were flattened. Instead of indexing by individual piece instances across the grid, tables were shrunk by indexing strictly by generic _piece types_.
>     
> - **Hash Table Slicing:** Pawn and Material hash tables were scaled down to just 1024 slots.
>     

|**Optimization Vector**|**Hand-Crafted Evaluation (HCE) Approach**|**NNUE Neural Network Approach**|
|---|---|---|
|**Primary Strength**|Negligible binary size, lightning-fast CPU execution speed.|Richer, deeply nuanced position evaluation.|
|**Memory Allocation**|Frees up RAM to expand the Transposition Table to 1-2 MiB.|Consumes RAM for layer updates; restricts TT to ~512 KiB.|
|**Core Tuning Vector**|Pure SPSA parameter tuning over thousands of traditional heuristic weights.|Advanced offline data filtering, quantization, and king-mirroring symmetries.|

The overwhelming consensus of the challenge was clear: **C-base engines** stripped of all bloated telemetry, combined with **quantized 8-bit networks** using geometric symmetries to mask out impossible game states, represent the current absolute ceiling of performance per byte.

---
