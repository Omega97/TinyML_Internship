
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

Clustering is performed across three representations: **Board State (Raw Features)**, **L1 Activations**, and **Sample-Wise Head Gradients** ($\nabla_W \mathcal{L}_i$). An additional baseline handles rule-based bucketing **by piece count**. Clustering is evaluated on a representative subset of $1\times 10^6$ positions sampled from the $120\times 10^6$ master dataset. Only summary metrics and projections are persisted, not the full cluster assignments.

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
- [ ] 🔴 Linear Model Baseline
- [ ] 🔴 FFNN Single-Hidden Layer ($h \in \{64, 128, 256\}$)
- [ ] 🔴 FFNN Dual-Hidden Layer ($h=H \in \{64, 128, 256\}$)
- [x] 🟢📋📊 Standard NNUE Architecture (Accumulator $h$, Hidden $H \in \{2h=H=64, 2h=H=128, 2h=H=256\}$) — 5 widths staged: $(h,H) \in \{(32,64),(64,128),(128,128),(128,256),(256,512)\}$, plus a CE-vs-train-size scaling sweep
- [ ] 🔴 *Optional*: Deep NNUE Architecture (Accumulator $h$, Dual-Hidden $H_1, H_2$)
- [ ] 🔴 Linear Model Baseline 
- [ ] 🔴 FFNN Single-Hidden Layer ($h \in \{64, 128, 256\}$) 
- [ ] 🔴 FFNN Dual-Hidden Layer ($h=H \in \{64, 128, 256\}$) 
- [ ] 🔴 Capacity-Matched Dense Baseline (e.g., $H=512$ single head to compare against $B=4, H=256$ MoE)

#### Results Summary
- Five two-hidden NNUE models staged; test CE reaches 0.629 for the largest (W256_H512). Scaling sweeps show CE keeps improving with training data up to ~90M rows.
- Linear / FFNN / capacity-matched dense baselines not yet run.

---

### 4. Mixture of Experts (MoE) 👨‍🔬

`[RESULTS/MoE/]`

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
- [ ] 🔴 **Piece-Count Routed MoE**: Fixed Piece-Count Dispatcher + NNUE Expert Heads ($K=7$)
- [ ] 🔴 **$L1$-Clustered Hard MoE**: $L1$ MLP Dispatcher + NNUE Expert Heads ($K \in \{2, 4, 8, 16\}$)
- [ ] 🟠📋📊 **Gradient-Clustered Hard MoE**: Gradient MLP Dispatcher + NNUE Expert Heads ($K \in \{2, 4, 8, 16\}$) — *preliminary*: linear (not MLP) dispatcher, $K \in \{3, 4\}$, 1–2M rows, 2-epoch expert fine-tune
- [ ] 🔴 **End-to-End Soft-Gated MoE**: Top-1 / Top-2 Softmax Gating Network + NNUE Experts (Trained joint end-to-end with load balancing loss)
- [ ] 🔴  [[End-to-End Sparse Top-1 MoE]] - the most elegant approach for a Mixture of Experts; it forces the network to discover its own optimal clustering strategy purely based on minimizing the valuation error.
- [ ] 🔴 **Oracle Upper-Bound MoE**: Perfect assignment ($\arg\min_k \mathcal{L}_k$) to measure routing headroom 
- [ ] 🔴 **Piece-Count Routed MoE**: Fixed Piece-Count Dispatcher + NNUE Experts ($K=8$) 
- [ ] 🔴 **$L1$-Clustered Hard MoE (Frozen vs. Unfrozen $L1$)**: $L1$ MLP Router + Experts ($K \in \{2, 4, 8, 16\}$) 
- [ ] 🔴 **End-to-End Sparse Top-1 MoE**: Switch-style joint training with load balancing loss ($\alpha \in \{0.001, 0.01, 0.05\}$) 
- [ ] 🔴 **End-to-End Top-2 Soft-Gated MoE**: Blended top-2 expert baseline (Performance upper bound vs. Top-1)

#### Results Summary (preliminary)
- Hard gradient-clustered MoE (K=3,4) matches but does not yet beat the base (MoE CE 0.630 vs base 0.630); the oracle router (CE 0.628) leaves modest headroom.
- No routing gain so far: experts underfit (1–2M rows, 2 epochs) and the L1 dispatcher recovers gradient buckets poorly.

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

**Run 4 — Preliminary gradient-clustered MoE:** 🟠 2026-09-16, `scripts/run_moe_pipeline.py`. Two preliminary hard-MoE runs: `kmeans_b3_1m` ($K=3$, 1M rows) and `kmeans_b4_2m` ($K=4$, 2M rows). Pipeline: 48-d sample gradients → mini-batch k-means → linear $L1$ dispatcher → per-cluster fine-tuned expert heads (frozen L1). Tables: `eval.json` (base/moe/oracle CE+MAE), `expert_metrics.json`, `diagnostics.json`, `summary.json`. Plots in `plots/` (PCA/t-SNE, cluster sizes, centroid cosine, dispatcher accuracy/confusion, expert CE, MoE-vs-base). Not the full battery: linear dispatcher (not MLP), $K \in \{3, 4\}$ only, and 1–2M routed rows (below the $7\times10^6$ per-expert target for $H=256$).

