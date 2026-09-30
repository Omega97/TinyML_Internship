
# Goal

> The goal is to provide a comprehensive evaluation suite for all candidate architectures. Every script must implement a model or pipeline, export structured numerical results, and generate corresponding diagnostic plots.

---

## Battery of Tests

This document serves as both the exhaustive specification of project deliverables and a living dashboard to track progress. 

### Emoji Legend
- 🔴 = Not implemented
- 🟠 = Implemented with errors / Under debugging
- 🟢 = Fully implemented and verified
- 📋 = Exports numerical metrics / evaluation dataset
- 📊 = Generates diagnostic plots

---

### 1. Clustering 👥 

`[RESULTS/clustering/]`

Clustering is performed across three representations: **Board State (Raw Features)**, **L1 Activations**, and **Sample-Wise Head Gradients** ($\nabla_W \mathcal{L}_i$). An additional baseline handles rule-based bucketing **by piece count**. Clustering is evaluated on a representative subset of $1\times 10^6$ positions sampled from the $150.8\times 10^6$ master dataset. Only summary metrics and projections are persisted, not the full cluster assignments.

#### Metrics
- **Cluster Balance**: Sizes and relative proportions per cluster
- **Inertia**: Within-Cluster Sum of Squares (WCSS)
- **Silhouette Score**: Average silhouette coefficient
- **Centroid Separation**: Pairwise cosine distance matrix between centroids
- **Adjusted Rand Index (ARI) / NMI**: Overlap analysis across different feature representations

#### Plots
- 2D/3D PCA projections (Board State, L1 Activations, Sample Gradients)
- 2D/3D t-SNE / UMAP projections (Board State, L1 Activations, Sample Gradients)
- Cluster size distribution bar charts

#### Algorithms & Experiments
- [x] 🟢📋📊 Board State (Raw Features) with Mini-Batch $k$-Means ($k \in \{2, 4, 8, 16\}$)
- [x] 🟢📋📊 Board State (Raw Features) with DBSCAN (Tuned $\varepsilon$ and `min_samples` targeting $\approx 2\text{--}16$ clusters)
- [x] 🟢📋📊 L1 Activations with Mini-Batch $k$-Means ($k \in \{2, 4, 8, 16\}$)
- [x] 🟢📋📊 L1 Activations with DBSCAN (Tuned $\varepsilon$ and `min_samples` targeting $\approx 2\text{--}16$ clusters)
- [x] 🟢📋📊 Sample Gradients ($\nabla_{W_\text{head}} \mathcal{L}$) with Mini-Batch $k$-Means ($k \in \{2, 4, 8, 16\}$)
- [x] 🟢📋📊 Sample Gradients ($\nabla_{W_\text{head}} \mathcal{L}$) with DBSCAN (Tuned targeting $\approx 2\text{--}16$ clusters)
- [x] 🟢📋📊 Handcrafted Piece-Count Bucketing (Disjoint intervals):
  $$\text{Buckets} = \{32\}, \; [28, 29], \; [30, 31], \; [24, 27], \; [20, 23], \; [16, 19], \; [10, 15], \; [2, 9]$$

#### Results Summary
- Gradient-space clustering is the cleanest signal: silhouette ≈ 0.25 (k=2) → 0.13 (k=16), well-balanced. Board features cluster weakly (0.11 → 0.02); L1 sits in between but collapses at k=16 (silhouette ≈ 0).
- Cross-representation overlap is low (ARI ≤ 0.19 for every pair), so board, L1, and gradient partitions capture largely disjoint structure — gradients look like the most informative routing signal.
- The piece-count baseline is trivially separable (silhouette ≈ 0.61) yet aligns poorly with any learned partition.

---

### 2. Dispatcher 📤 

`[RESULTS/dispatcher/]`

The dispatcher routes input instances to an expert bucket ID. The piece-count router processes bitboards/raw board inputs directly, whereas activation and gradient dispatchers map $L1$ activations to cluster IDs. Up to $5\times 10^6$ positions may be used to train non-linear dispatchers.

#### Metrics
- **Classification Accuracy**: Top-1 and Top-2 Accuracy
- **Macro-F1 & Weighted-F1 Score**: Per-bucket performance
- **Adjusted Rand Index (ARI) & Normalized Mutual Information (NMI)**: Alignment with pseudo-ground-truth cluster labels

#### Plots
- Normalized Confusion Matrix
- Per-class Precision-Recall Curves

#### Algorithms & Experiments
- [x] 🟢📋📊 Centroids from the Board State clustering (`argmin` of cosine similarity) — 1688-d board-state centroids, $B \in \{2, 4, 8, 16\}$
- [x] 🟢📋📊 Centroids from the Accumulator layer clustering (`argmin` of cosine similarity) — 256-d L1 centroids, $B \in \{2, 4, 8, 16\}$
- [x] 🟢📋📊 Linear $L1$ Dispatcher ($L1 \to \text{Bucket ID}$) — predicts L1 $k$-means buckets, $B \in \{2, 4, 8, 16\}$
- [x] 🟢📋📊 Single-Hidden MLP $L1$ Dispatcher ($L1 \xrightarrow{h=64} \text{Bucket ID}$) — predicts L1 $k$-means buckets, $B \in \{2, 4, 8, 16\}$
- [x] 🟢📋📊 Linear Sample-Gradient Approximation Dispatcher ($L1 \to \text{Bucket ID}$) — linear head predicting gradient $k$-means buckets, $B \in \{2, 4, 8\}$
- [x] 🟢📋📊 Single-Hidden MLP Sample-Gradient Approximation Dispatcher ($L1 \xrightarrow{h=64} \text{Bucket ID}$) — predicts gradient $k$-means buckets, $B \in \{2, 4, 8, 16\}$
- [x] 🟢📋📊 Single-Hidden MLP $L1$ Dispatcher ($L1 \xrightarrow{h \in \{32, 64, 128\}} \text{Bucket ID}$) — predicts L1 $k$-means buckets, $B \in \{2, 4, 8, 16\}$
- [x] 🟢📋📊 Single-Hidden MLP Sample-Gradient Dispatcher ($L1 \xrightarrow{h \in \{32, 64, 128\}} \text{Bucket ID}$) — predicts gradient $k$-means buckets, $B \in \{2, 4, 8, 16\}$
- [x] 🟢📋📊 Decision Tree / XGBoost $L1$ Dispatcher (Baseline non-neural router) — predicts L1 $k$-means buckets, $B \in \{2, 4, 8, 16\}$
- [x] 🟢📋📊 Piece-Count Rule-Based Dispatcher ($\text{Board Input} \to \text{Bucket ID}$) — fixed 8-interval piece-count router, evaluated by ARI/NMI vs. learned clusters

#### Results Summary
- Routing L1 → L1-cluster buckets is near-trivial (Top-1 ≈ 0.97–0.99): MLP and XGBoost are best and essentially tied, linear only slightly behind, hidden width (32–128) makes little difference.
- Routing L1 → gradient-cluster buckets is much harder (Top-1 ≈ 0.49–0.64, ARI ≤ 0.37): **L1 activations are a weak proxy for gradient direction** — the core motivation for gradient-based MoE routing.
- Cosine-centroid routers reproduce k-means at 0.82–0.95 (board) / 0.94–0.98 (L1); the piece-count rule aligns weakly with L1 (ARI ≈ 0.11–0.17) and not at all with gradients (ARI ≤ 0.05).

---

### 3. Base Models ♟

`[RESULTS/base/]`

Dense baseline models trained end-to-end on the full dataset without any clustering or routing mechanism.

#### Metrics
- **Loss Functions**: Cross-Entropy (CE) / MSE Evaluation Loss
- **Regression / Valuation Metrics**: Mean Absolute Error (MAE), $R^2$ Score
- **Inference Speed**: Throughput (NPS — Nodes Per Second / Batches Per Second)
- optional (not now!): Spearman Rank Correlation ($\rho$) & Pearson ($r$) measures how well the model preserves the relative ordering of positions compared to ground truth (Stockfish depth evaluation).

#### Plots
- Train vs. Validation Loss curves over training steps
- Prediction Error / Residual Distribution histograms

#### Algorithms & Experiments
- [x] 🟢📋📊 Linear Model Baseline
- [x] 🟢📋📊 FFNN Single-Hidden Layer ($h \in \{64, 128, 256\}$)
- [x] 🟢📋📊 FFNN Dual-Hidden Layer ($h=H \in \{64, 128, 256\}$)
- [x] 🟢📋📊 Standard NNUE Architecture (Accumulator $h$, Hidden $H \in \{2h=H=64, 2h=H=128, 2h=H=256\}$) — 5 widths staged: $(h,H) \in \{(32,64),(64,128),(128,128),(128,256),(256,512)\}$, plus a CE-vs-train-size scaling sweep
- [ ] 🔴 *Optional*: Deep NNUE Architecture (Accumulator $h$, Dual-Hidden $H_1, H_2$)
- [ ] 🔴 *Optional*: Capacity-Matched Dense Baseline (e.g., $H=512$ single head to compare against $B=4, H=256$ MoE)

#### Results Summary
- Five two-hidden NNUE models staged; test CE reaches 0.629 for the largest (W256_H512). Scaling sweeps show CE keeps improving with training data up to ~90M rows.
- Linear baseline (5,067 params) reaches test CE 0.766 / MAE 0.258 / R² 0.716; the single-hidden FFNN improves with width (H=64: CE 0.669, H=128: CE 0.654, H=256: CE 0.643 / R² 0.880) but a flat FFNN over concatenated features still underperforms the dual-POV NNUE at equal or smaller parameter count (e.g. FFNN_H256 = 433k params vs W128_H256 = 175k params at CE 0.632) — evidence of the NNUE inductive bias.
- The dual-hidden FFNN (H1=H2 ∈ {64,128,256}) adds a second CReLU layer: CE 0.656 / 0.645 / 0.632 at 112k/233k/499k params. At H=256 (499k params) it roughly matches the single-hidden FFNN and approaches the two-hidden NNUE at equal parameter count (W256_H512, 480k params, CE 0.629) — the extra depth recovers most of the flat-vs-accumulator gap, but the NNUE still edges it out at comparable size.
- Capacity-matched dense baseline not yet run.

---

### 4. Mixture of Experts (MoE) 👨‍🔬

`[RESULTS/moe/]`

MoE models combine a trained dispatcher (or soft gating network) with $K$ specialized expert heads. Each expert head must be trained on a minimum of $7\times 10^6$ routed samples for $H=256$ models to prevent underfitting.

#### Metrics
- **Overall Loss**: Cross-Entropy / MSE on test set vs. Base Model
- **Expert Utilization / Load Balance**: Variance of sample distribution across experts
- **Gating Entropy**: Average router prediction entropy

#### Plots
- Train / Validation Loss comparison (MoE vs. Base Model of equivalent parameter count)
- Expert Allocation Heatmap (Position characteristics vs. Expert selection rate)
- Confusion / Routing Matrix across game stages

#### Algorithms & Experiments

- [x] 🟢📋📊 **Piece-Count Routed MoE**:
	- **clustering**: handcrafted piece-count buckets (8 disjoint intervals, §1 `PIECE_COUNT_BUCKETS`), not learned
	- **dispatcher**: fixed piece-count rule (Board Input → Bucket ID), no training
	- **n buckets:** $K = 8$
	- **L1 frozen**: yes
	- **expert heads**: NNUE (L2, head) blocks, one per bucket, cloned from the base
	- **training**: 2M rows (`moe_b4_2m`), 2-epoch expert fine-tune (frozen L1)
	- **base**: dual-POV two-hidden NNUE ($W=128$, $H=256$, 844-d input)

- [x] 🟢📋📊 **$L1$-Clustered Hard MoE**
	- **clustering**: mini-batch $k$-Means on 256-d $L1$ activations
	- **dispatcher**: single-hidden MLP ($L1 \to \text{Bucket ID}$), $h=64$, 8 epochs
	- **n buckets**: $K \in \{2, 4, 8, 16\}$
	- **L1 frozen**: yes
	- **expert heads**: NNUE (L2, head) blocks, one per bucket, 2-epoch fine-tune
	- **training**: 2M rows (`moe_b4_2m`)
	- **base**: dual-POV two-hidden NNUE ($W=128$, $H=256$, 844-d input)

- [x] 🟢📋📊 **Gradient-Clustered Hard MoE**
	- clustering: mini-batch $k$-Means on 48-d sample head gradients ($\nabla_w \mathcal{L}$, Gaussian-projected from 66,563-d)
	- dispatcher: single-hidden MLP on $L1$ ($h=64$, 8 epochs) predicting gradient-cluster buckets
	- n buckets: $K \in \{2, 4, 8, 16\}$
	- L1 frozen: yes
	- expert heads: NNUE (L2, head) blocks, one per bucket, 2-epoch fine-tune
	- training: 2M rows (`moe_b4_2m`); oracle upper bound (nearest gradient centroid) also evaluated
	- **base**: dual-POV two-hidden NNUE ($W=128$, $H=256$, 844-d input)

- [x] 🟢📋📊 **End-to-End Soft-Gated MoE**
	- clustering: none (implicit, learned end-to-end)
	- dispatcher: top-1 / top-2 softmax gating network (linear, differentiable)
	- n experts: $K \in \{2, 4, 8, 16\}$ (joint end-to-end training)
	- L1 frozen: yes
	- expert heads: NNUE (L2, head) blocks, one per expert, cloned from base
	- load balancing: auxiliary loss ($\alpha = 0.01$), 5 epochs joint training on 2M rows (`moe_b4_2m`)
	- **base**: dual-POV two-hidden NNUE ($W=128$, $H=256$, 844-d input)

- [x] 🟢📋📊 **End-to-End Sparse Top-1 MoE (Switch)**
	- clustering: none (self-organizing)
	- dispatcher: Switch-style single-expert (hard top-1) gating, differentiable gate trained only via load-balancing loss
	- load balancing: $\alpha \in \{0.001, 0.01, 0.05\}$
	- n experts: $K \in \{2, 4, 8, 16\}$ (joint end-to-end training)
	- L1 frozen: yes
	- expert heads: NNUE (L2, head) blocks, one per expert, cloned from base
	- training: 5 epochs joint training on 2M rows (`moe_b4_2m`)
	- **base**: dual-POV two-hidden NNUE ($W=128$, $H=256$, 844-d input)

- [x] 🟢📋📊 **Oracle Upper-Bound MoE**
	- clustering: n/a (reuses the gradient-clustered hard MoE's $K$ expert heads)
	- dispatcher: perfect assignment ($\arg\min_k \mathcal{L}_k$) — retroactively routes each position to its minimum-loss expert
	- n buckets: $K \in \{2, 4, 8, 16\}$ (same expert set as the gradient-clustered MoE)
	- L1 frozen: yes
	- purpose: measures max routing headroom (unachievable upper bound — it uses the target to pick the best expert)
	- **base**: dual-POV two-hidden NNUE ($W=128$, $H=256$, 844-d input)

- [x] 🟢📋📊 **Random Dispatcher Control**
	- **purpose**: isolates routing quality from raw capacity
	- **dispatcher**: random (uniform) bucket assignment over the same expert set (seed 0)
	- **n buckets**: $K \in \{2, 4, 8, 16\}$
	- **L1 frozen**: yes
	- **applied to**: L1-clustered and gradient-clustered hard MoE expert heads
	- **base**: dual-POV two-hidden NNUE ($W=128$, $H=256$, 844-d input)

- [ ] 🔴📋📊 **Well-Resourced Gradient-Clustered Hard MoE (K=8)**
	- clustering: mini-batch $k$-Means on 48-d sample head gradients ($\nabla_w \mathcal{L}$, Gaussian-projected from 66,563-d), $K=8$
	- dispatcher: single-hidden MLP ($h=64$) trained **200 epochs** (early-stopped on validation) predicting gradient-cluster buckets; input = $L1 \oplus$ board features to combat the weak $L1 \to$ gradient proxy (cf. Run 7)
	- n buckets: $K = 8$
	- L1 frozen: yes
	- expert heads: 8 NNUE (L2, OUT) blocks, one per bucket, cloned from base, **100-epoch** fine-tune (warm-start + cosine LR decay) on **7M positions per head** (≈56M total, new `moe_b8_56m` pack)
	- evaluation: base vs MoE CE/MAE + argmin-loss oracle upper bound + capacity-matched dense reference
	- **base**: dual-POV two-hidden NNUE ($W=128$, $H=256$, 844-d input)

- [ ] 🔴 **Capacity-Matched Dense**
	- **purpose**: validates sparse execution vs. dense scaling
	- **reference**: dense single head (e.g. $H=512$) vs. $B=4$, $H=256$ MoE (cf. §3 optional baseline)

- [ ] 🔴 **Inference NPS Benchmark**
	- **purpose**: measures real-world execution overhead
	- **metric**: nodes-per-second (NPS), MoE vs. base
    
- [ ] 🔴 todo: L1 sparse weights



#### Results Summary (preliminary)
- Gradient-clustered hard MoE (full battery, K ∈ {2,4,8,16}, 2M rows, MLP h=64 dispatcher): MoE CE 0.6307 / 0.6306 / 0.6300 / 0.6301 vs base 0.6303 for K=2/4/8/16 — only K=8 (0.63002) and K=16 (0.63006) edge out the base, and MAE is flat-to-worse. The L1→gradient dispatcher recovers the gradient partition poorly (Top-1 ≈ 0.64/0.59/0.53/0.48), confirming L1 is a weak proxy for gradient direction.
- The oracle upper bound shows real headroom that the dispatcher leaves on the table: oracle CE 0.6321 / 0.6312 / 0.6291 / 0.6280 for K=2/4/8/16 — i.e. at K=16 perfect gradient routing would reach 0.628 vs the 0.630 the MLP router actually achieves. More gradient clusters (K↑) monotonically improve the oracle even though each expert sees fewer rows, so the bottleneck is the L1→gradient dispatcher, not the experts.
- No routing gain so far: experts underfit (1–2M rows, 2 epochs, below the $7\times10^6$ per-expert target) and the L1 dispatcher recovers gradient buckets poorly.
- Piece-count routed MoE (K=8, 2M rows) does not beat the base either: MoE CE 0.6303 vs base 0.6303 (MAE 0.1331 vs 0.1326) — per-bucket expert CE ≈ base CE on every bucket, confirming piece count is too weak a routing signal for any specialization.
- L1-clustered hard MoE (K ∈ {2,4,8,16}, 2M rows, MLP h=64 dispatcher) shows no meaningful routing gain: MoE CE 0.6309 / 0.6305 / 0.6303 / 0.6300 vs base 0.6303 for K=2/4/8/16. Only K=16 edges out the base on CE (−0.0002) while MAE stays flat-to-worse. The L1 partition is trivially recoverable (dispatcher Top-1 ≈ 0.98–1.00), so the experts specialize on nearly the same signal the base already models.
- End-to-end soft-gated MoE (K ∈ {2,4,8,16}, top-1 & top-2, 2M rows, 5 epochs, α=0.01) also fails to beat the base. Best is top-2 K=2: MoE CE 0.6304 vs base 0.6303 (the only config with better MAE, 0.1324 vs 0.1326). Top-2 beats top-1 throughout; gate entropy ≈ ln K and load variance ≈ 0, so the load-balancing loss pins the router near-uniform and the experts never specialize — larger K only adds underfit parameters (CE worsens to 0.6323 at top-1 K=16).
- Switch (end-to-end sparse top-1, K ∈ {2,4,8,16}, α ∈ {0.001,0.01,0.05}) also fails to beat the base. Best is α=0.001 K=2 (CE 0.6307); CE worsens monotonically with K (→0.6323–0.6326 at K=16). α barely matters (a given K varies by ≤0.0003 across α), and gate entropy ≈ ln K with load variance ≈ 0 — the hard argmax gate receives no gradient from CE (only the load-balancing term), so it stays near-uniform and no expert specializes.
- True oracle (argmin_k L_k) on the gradient-clustered experts reveals large, monotonic headroom that no router recovers: oracle CE 0.6244 / 0.6160 / 0.6067 / 0.6005 (MAE 0.124 / 0.115 / 0.101 / 0.088) vs base 0.6303 for K=2/4/8/16. This far beats the nearest-gradient-centroid proxy (0.6321/0.6312/0.6291/0.6280), so gradient centroids are NOT the argmin-loss partition; and it beats the realized MoE (≈0.630). The experts can specialize, but the routing signal (L1 → gradient cluster, or gradient centroid) is too weak to realize it.
- Random dispatcher control (same experts, uniform random routing) is always worse than the trained dispatcher and worse than the base, and it degrades with K: random CE 0.6311/0.6337/0.6429/0.6398 (gradient) and 0.6317/0.6327/0.6341/0.6346 (L1) vs base 0.6303. The experts genuinely drift from the base (a random expert is a poor generalist), so it is the dispatcher's routing — not raw capacity — that buys the MoE back to ≈ base (0.630).

---

## Runs

> Report here the runs.

**Run 1 — Clustering battery:** 🟢 2026-09-24, `scripts/run_clustering_battery.py`. Sample is the 1,000,000-row training split in `data/processed/board_eval/moe/moe_b3_1m` (corpus count 150,815,697; the 1% base-model test split is excluded). Board-state (1688-d), L1 (256-d), head-gradient (48-d) mini-batch k-means and DBSCAN, plus handcrafted piece-count bucketing. Tables: `stats.csv`, `overlap.csv` (ARI/NMI), `summary.json`. Plots in `plots/` (PCA/t-SNE/UMAP 2D/3D, cluster-size bars, centroid-cosine panels). The 10,000-point coordinates are `projections/embeddings.npz`.

**Run 2 — Gradient clustering + linear dispatcher:** 🟢 2026-09-23, `scripts/run_cluster_dispatcher_report.py`. Mini-batch k-means on the 2M reference gradient cache (`moe_b4_2m`) at $B \in \{2, 4, 8\}$, then one linear $L1$ dispatcher per $B$. Tables: `dispatcher/stats.csv` (train/val accuracy vs. majority dummy and chance), `clustering/b{B}/diagnostics.json`. Plots in `dispatcher/plots/` (accuracy bars, learning curves).

**Run 2b — Centroid dispatchers (argmin cosine similarity):** 🟢 2026-09-25, `scripts/run_centroid_dispatchers.py`. Two parameter-free dispatchers that route each position to the nearest centroid by cosine similarity: board-state (1688-d) and accumulator/L1 (256-d) k-means centroids, $B \in \{2, 4, 8, 16\}$ on the 1M-row `moe_b3_1m` subsample. Scored against the Euclidean k-means labels as pseudo-ground-truth (Top-1 accuracy, ARI, NMI, macro/weighted F1). Tables: `dispatcher/centroid/stats.csv`, `dispatcher/centroid/<rep>/b{B}/diagnostics.json`; centroids + labels in `dispatcher/centroid/<rep>/b{B}/`. Plots in `dispatcher/centroid/plots/` (accuracy vs. $B$, normalized confusion matrices). Board routing agrees with k-means at 0.82–0.95; L1 at 0.94–0.98.

**Run 2c — L1 dispatchers (linear + MLP):** 🟢 2026-09-25, `scripts/run_l1_dispatchers.py`. Buckets are mini-batch k-means on the 256-d L1 activations; two learned routers (linear and single-hidden MLP $h=64$) map L1 → bucket ID on the 1M-row `moe_b3_1m` subsample ($B \in \{2, 4, 8, 16\}$, 90/10 split). Tables: `dispatcher/l1/stats.csv` (Top-1/Top-2, macro/weighted F1, ARI, NMI vs. k-means labels, majority dummy, chance); clusters in `dispatcher/l1/clusters/`, per-run `dispatcher.pt`/`history.json`/`labels_pred.npy`/`diagnostics.json` in `dispatcher/l1/{linear,mlp64}/b{B}/`. Plots in `dispatcher/l1/plots/` (accuracy, normalized confusion, per-class PR curves). Both routers recover the L1 partition near-perfectly (Top-1 ≈ 0.99 at $B=2$ → ≈ 0.97 at $B=16$; MLP edges out linear at larger $B$).

**Run 2d — Single-hidden MLP dispatchers (hidden-width sweep):** 🟢 2026-09-25, `scripts/run_mlp_dispatchers.py`. Single-hidden MLP routers ($h \in \{32, 64, 128\}$) trained on L1 and scored against two bucket sources on the 1M-row `moe_b3_1m` subsample ($B \in \{2, 4, 8, 16\}$, 90/10 split): L1 k-means and 48-d sample-gradient k-means. Tables: `dispatcher/mlp/stats.csv`; clusters in `dispatcher/mlp/clusters/<target>/b{B}/`, per-run artifacts in `dispatcher/mlp/<target>/h{h}/b{B}/`. Plots in `dispatcher/mlp/plots/` (accuracy vs. $B$ per $h$, normalized confusion, per-class PR curves). L1 buckets are recovered near-perfectly (Top-1 ≈ 0.99→0.97, ARI ≈ 0.97→0.94) with hidden width making little difference. Gradient buckets are only weakly recoverable from L1 (Top-1 ≈ 0.64→0.49, ARI ≈ 0.07→0.37), confirming L1 activations are a poor proxy for gradient-direction clusters.

**Run 2e — Non-neural L1 dispatchers (decision tree + XGBoost):** 🟢 2026-09-25, `scripts/run_tree_dispatchers.py`. Decision tree (max_depth 12) and XGBoost (50 trees, depth 6) routers mapping L1 → L1 k-means buckets on the 1M-row `moe_b3_1m` subsample ($B \in \{2, 4, 8, 16\}$, 90/10 split). Tables: `dispatcher/tree/stats.csv`; clusters in `dispatcher/tree/clusters/b{B}/`, models + `labels_pred.npy` + `diagnostics.json` in `dispatcher/tree/{tree,xgb}/b{B}/`. Plots in `dispatcher/tree/plots/`. XGBoost (Top-1 ≈ 0.99→0.89, ARI ≈ 0.94→0.80) approaches the MLP routers; the single decision tree is weaker (Top-1 ≈ 0.96→0.70, ARI ≈ 0.84→0.51).

**Run 2f — Piece-count rule-based dispatcher:** 🟢 2026-09-25, `scripts/run_piececount_dispatcher.py`. Fixed 8-interval piece-count router (Board → Bucket ID) on the 1M-row `moe_b3_1m` subsample; no training. Tables: `dispatcher/piececount/sizes.csv` (bucket sizes/proportions) and `dispatcher/piececount/alignment.csv` (ARI/NMI vs. L1 and gradient k-means). Plots in `dispatcher/piececount/plots/` (size bars, ARI vs. $B$). The piece-count partition aligns weakly with L1 clusters (ARI ≈ 0.11–0.17) and essentially not at all with gradient clusters (ARI ≈ 0.01–0.05).

**Run 3 — Base NNUE models + scaling:** 🟢 2026-09-08 (staged via `scripts/stage_thesis_results.py`), `scripts/train_nnue-gpu.py`. Five dual-POV two-hidden NNUE checkpoints staged into `base/`: $(h,H) \in \{(32,64),(64,128),(128,128),(128,256),(256,512)\}$. CE-vs-train-size scaling sweeps for $H \in \{64, 256\}$ in `base/scaling/`. Manifest: `manifest.json`.

**Run 3b — Dense base baselines (linear + single/double-hidden FFNN):** 🟢 2026-09-28, `scripts/run_base_baselines.py`. GPU compact-path training mirroring `train_nnue-gpu.py` (test_fraction 0.01, 100 epochs, 512×10,000 steps/epoch, Adam 1e-2→1e-3). Seven runs staged into `base/` via `scripts/stage_base_baselines.py`: `Linear` (5,067 params); single-hidden `FFNN_H64/H128/H256` (108k/217k/433k params); dual-hidden `FFNN2_H64/H128/H256` (112k/233k/499k params, `DualHiddenFFNN`). Each run exports `config.json`, `history.json`, `ce.png`, `best.pt`, plus `metrics.json` (CE, MAE, MSE, R², train/inference NPS). Test CE: Linear 0.766 / FFNN 0.669·0.654·0.643 / FFNN2 0.656·0.645·0.632.

**Run 4 — Preliminary gradient-clustered MoE:** 🟠 2026-09-16, `scripts/run_moe_pipeline.py`. Two preliminary hard-MoE runs: `kmeans_b3_1m` ($K=3$, 1M rows) and `kmeans_b4_2m` ($K=4$, 2M rows). Pipeline: 48-d sample gradients → mini-batch k-means → linear $L1$ dispatcher → per-cluster fine-tuned expert heads (frozen L1). Tables: `eval.json` (base/moe/oracle CE+MAE), `expert_metrics.json`, `diagnostics.json`, `summary.json`. Plots in `plots/` (PCA/t-SNE, cluster sizes, centroid cosine, dispatcher accuracy/confusion, expert CE, MoE-vs-base). Not the full battery: linear dispatcher (not MLP), $K \in \{3, 4\}$ only, and 1–2M routed rows (below the $7\times10^6$ per-expert target for $H=256$).

**Run 5 — Piece-count routed MoE (K=8):** 🟢 2026-09-28, `scripts/run_piececount_moe.py`. Fixed 8-interval piece-count router (the §1 `PIECE_COUNT_BUCKETS`) + 8 expert (L2, head) blocks fine-tuned on the 2M-row `moe_b4_2m` training pack (frozen L1, 2 epochs). No gradient computation or dispatcher. Tables: `RESULTS/moe/piececount_k8/{eval.json, expert_metrics.json, summary.json}`. Plots: `expert_ce_by_bucket.png`, `moe_vs_base_ce.png`. Result: MoE CE 0.6303 vs base 0.6303 (MAE 0.1331 vs 0.1326) — no routing gain; per-bucket expert CE ≈ base CE on every bucket.

**Run 6 — L1-clustered hard MoE (K ∈ {2,4,8,16}):** 🟢 2026-09-29, `scripts/run_l1_clustered_moe.py`. Pipeline per $K$: mini-batch k-means on 256-d L1 activations → single-hidden MLP dispatcher ($h=64$, 8 epochs, 90/10 split) → per-bucket NNUE (L2, head) expert fine-tune (frozen L1, 2 epochs) on the 2M-row `moe_b4_2m` pack. No gradient computation / oracle. Tables per $K$: `RESULTS/moe/l1_clustered_k{K}/{labels.npy, centroids.npy, diagnostics.json, dispatcher.pt, dispatcher_history.json, labels_dispatcher.npy, expert_metrics.json, eval.json, summary.json}`. Plots: `expert_ce_by_bucket.png`, `moe_vs_base_ce.png`. Dispatcher Top-1 ≈ 0.996/0.992/0.986/0.980 for K=2/4/8/16. Result: MoE CE 0.6309 / 0.6305 / 0.6303 / 0.6300 vs base 0.6303 — no meaningful routing gain (only K=16 −0.0002 on CE), confirming the L1 partition carries little valuation-specialization signal.

**Run 7 — Gradient-clustered hard MoE (K ∈ {2,4,8,16}):** 🟢 2026-09-29, `scripts/run_gradient_clustered_moe.py`. Full battery replacing the preliminary Run 4: mini-batch k-means on the cached 48-d sample head gradients (`moe_b4_2m/gradients.npy`, no recompute) → single-hidden MLP dispatcher on L1 ($h=64$, 8 epochs) predicting gradient buckets → per-bucket NNUE (L2, head) expert fine-tune (frozen L1, 2 epochs) → eval with oracle upper bound. Tables per $K$: `RESULTS/moe/grad_clustered_k{K}/{labels.npy, centroids.npy, diagnostics.json, dispatcher.pt, dispatcher_history.json, labels_dispatcher.npy, expert_metrics.json, eval.json, summary.json}`. Dispatcher Top-1 ≈ 0.643/0.587/0.534/0.476 for K=2/4/8/16. Result: MoE CE 0.6307 / 0.6306 / 0.6300 / 0.6301 vs base 0.6303; oracle CE 0.6321 / 0.6312 / 0.6291 / 0.6280 — the oracle improves monotonically with $K$ (0.628 at K=16) but the L1→gradient dispatcher recovers only ~0.48–0.64 of the partition, so the realized MoE cannot reach it.

**Run 8 — End-to-End soft-gated MoE (top-1 & top-2, K ∈ {2,4,8,16}):** 🟢 2026-09-29, `scripts/run_soft_gated_moe.py`. Joint end-to-end training of a differentiable top-1/top-2 softmax gate (linear, random init) + $K$ expert (L2, head) blocks (cloned from base, frozen L1) on the 2M-row `moe_b4_2m` pack, CE + $\alpha=0.01$ load-balancing loss, 5 epochs, Adam 1e-3. New `SoftGatedMoE` in `src/tinymlinternship/nnue/moe.py`. Tables per run: `RESULTS/moe/soft_gated_k{K}_top{tk}/{model.pt, history.json, eval.json, summary.json}` + `soft_gated_summary.json`. Result: no config beats base (CE 0.6303); best is top-2 K=2 (CE 0.6304, MAE 0.1324 vs base 0.1326). Top-2 CE 0.6304/0.6306/0.6306/0.6310 and top-1 CE 0.6307/0.6315/0.6317/0.6323 for K=2/4/8/16. Gate entropy ≈ ln K and load variance ≈ 0 — the load-balancing loss keeps routing near-uniform, so experts never specialize.

**Run 9 — End-to-End sparse top-1 MoE (Switch, α ∈ {0.001,0.01,0.05}, K ∈ {2,4,8,16}):** 🟢 2026-09-29, `scripts/run_switch_moe.py`. Switch Transformer-style hard top-1 gating (reuses `SoftGatedMoE(top_k=1)`) trained jointly on the 2M-row `moe_b4_2m` pack, frozen L1, CE + $\alpha$·load-balancing loss, 5 epochs, Adam 1e-3. Tables per run: `RESULTS/moe/switch_k{K}_a{alpha}/{model.pt, history.json, eval.json, summary.json}` + `switch_summary.json`. Result: no config beats base (CE 0.6303); best α=0.001 K=2 (0.6307). CE by K=2/4/8/16: 0.6307/0.6312/0.6316/0.6326 (α=0.001), 0.6307/0.6312/0.6317/0.6323 (α=0.01), 0.6309/0.6313/0.6319/0.6325 (α=0.05) — α has little effect and K only adds underfit parameters; gate entropy ≈ ln K and load variance ≈ 0, so the hard argmax gate (no CE gradient) never specializes.

**Run 10 — Oracle Upper-Bound MoE (K ∈ {2,4,8,16}):** 🟢 2026-09-29, `scripts/run_oracle_moe.py`. Loads the gradient-clustered hard MoE's $K$ expert heads (`grad_clustered_k{K}/moe.pt`, frozen L1) and computes the true routing upper bound: for each test position, evaluate all $K$ experts and route to $\arg\min_k \mathcal{L}_k$. Tables: `RESULTS/moe/oracle_k{K}/{eval.json, summary.json}` + `oracle_summary.json`; plot `moe_vs_base_ce.png`. Result: oracle CE 0.6244 / 0.6160 / 0.6067 / 0.6005 (MAE 0.124 / 0.115 / 0.101 / 0.088) vs base 0.6303 for K=2/4/8/16 — monotonic, large headroom; the true argmin oracle far beats the nearest-gradient-centroid proxy (0.6321/0.6312/0.6291/0.6280) and the realized MoE (≈0.630).

**Run 11 — Random Dispatcher Control (L1 & gradient hard MoE, K ∈ {2,4,8,16}):** 🟢 2026-09-29, `scripts/run_random_dispatcher.py`. Replaces the trained MLP dispatcher with a uniform random router (seed 0) over the same expert heads (`l1_clustered_k{K}/moe.pt` and `grad_clustered_k{K}/moe.pt`, frozen L1). Tables: `RESULTS/moe/random_<variant>_k{K}/{eval.json, summary.json}` + `random_dispatcher_summary.json`. Result: random routing is always worse than the trained dispatcher and worse than base, degrading with K — random CE 0.6311/0.6337/0.6429/0.6398 (gradient) and 0.6317/0.6327/0.6341/0.6346 (L1) vs base 0.6303. Confirms the experts drift from the base and the dispatcher's routing (not raw capacity) is what returns the MoE to ≈ base.

**Run 12 — Well-resourced gradient-clustered hard MoE (K=8, 7M/head):** 🔴 2026-09-30, `scripts/run_wellresourced_gradient_moe.py` (to be implemented). Targets the two bottlenecks isolated in Runs 6–11 — expert underfitting (≤2M rows, 2 epochs) and the weak $L1 \to$ gradient dispatcher (Top-1 ≈ 0.53 at $K=8$): mini-batch k-means on the cached 48-d head gradients ($K=8$) → single-hidden MLP dispatcher ($h=64$, **200 epochs**, early-stopped; input $L1 \oplus$ board features) → 8 expert NNUE (L2, OUT) blocks (frozen L1) fine-tuned **100 epochs** with warm-start + cosine LR decay on **7M routed positions per head** (≈56M, new `moe_b8_56m` pack). Tables per run: `RESULTS/moe/wellresourced_k8/{labels.npy, centroids.npy, diagnostics.json, dispatcher.pt, dispatcher_history.json, labels_dispatcher.npy, expert_metrics.json, eval.json, summary.json}` + `wellresourced_summary.json`; plots `expert_ce_by_bucket.png`, `moe_vs_base_ce.png`. Also evaluates the argmin-loss oracle upper bound (Run 10) and a capacity-matched dense reference. Target: MoE CE meaningfully below base 0.6303.

