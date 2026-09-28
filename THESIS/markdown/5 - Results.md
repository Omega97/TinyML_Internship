# 5 Experimental Results

#todo DeepSeek: "Figures use repo-root-relative paths (e.g. RESULTS/clustering/..., RESULTS/dispatcher/...); confirm the build resolves them the same way as in Section 5.2."

#todo DeepSeek (resolved): tables are now numbered sequentially 5.1–5.8 (base 5.1; clustering 5.2–5.4; dispatcher 5.5–5.8).

## 5.1 Experimental Setup

<span style="color: #808080;">[Dataset, train/validation/test splits, hardware, training environment]</span> This section describes the environment, the data, the models, and the evaluation protocol used throughout the experiments of this chapter. Every result reported in Sections 5.2–5.5 was produced with the pipeline and metrics fixed here, so that the base models, the clustering diagnostics, the dispatchers, and the mixture-of-experts models are directly comparable.

### 5.1.1 Hardware and Software

<span style="color: #808080;">[Hardware and Software]</span> All models were trained and evaluated on a single NVIDIA DGX Spark workstation, equipped with a GB10 Grace Blackwell processor with 128 GB of unified memory and a CUDA-capable GPU (CUDA 13.0). Training used PyTorch 2.12 in Python 3.12. Two numerical settings are worth stating because they affect the numbers reported below: *matmuls* run with bfloat16 automatic mixed precision and TF32 enabled, while the loss and all reported metrics are accumulated in single precision (float32). The training tables are stored as compact `int16` feature indices resident in GPU memory, which is what allows the full dataset to be sampled on-device with `torch.randint` + gather rather than through a CPU data loader.

### 5.1.2 Dataset and Teacher

<span style="color: #808080;">[Dataset and Teacher]</span> The dataset consists of **150,815,697 chess positions** stored as 429 memory-mapped `.npz` slices. Positions are sampled from Lichess human games, a small tactical collection, and high-level bot games, retaining the natural phase distribution of human play (mostly middlegame, fewer openings and endgames). Each position is labeled with a full win/draw/loss distribution from the side-to-move perspective, $\hat{p} = (p_W, p_D, p_L)$, produced by the Leela Chess Zero (Lc0) teacher (network `791556.pb.gz`) with a single depth-1 MCTS search. We train on the full WDL distribution rather than its scalar summary, which is why the primary loss is soft cross-entropy (Section 5.1.5).

Each FEN is encoded into a sparse binary feature vector of dimension $d_{\text{in}} = 844$ per perspective, split into 716 piece-square features and 128 tactical features (pieces under attack / attacking the king). Two such vectors are produced per position — one for the side to move and one for the opponent — giving the dual-POV representation used by the accumulator layer (Chapter 4).

### 5.1.3 Data Splits

<span style="color: #808080;">[Data Splits]</span> The dataset is partitioned with a stratified per-slice random split: exactly 1% of each slice is held out as the test set (seed 0), leaving the remaining 99% for training. This keeps the same phase mix on both sides and prevents any single slice from leaking positions into the test set. The resulting test set contains **1,507,940 positions**; every test metric reported in this chapter is computed on this fixed set, and no test position participates in clustering, dispatcher training, or expert fine-tuning.

The clustering and dispatcher experiments operate on a representative **1,000,000-position subsample** of the training split (the same subsample reused across all representations), while the sample-gradient caches used for gradient-based routing are computed on up to 2,000,000 positions. Only summary statistics and projection coordinates are persisted from the clustering stage, never the full cluster assignments.

### 5.1.4 Models and Baselines

<span style="color: #808080;">[Models and Baselines]</span> The base models define the reference point against which every MoE variant is judged. They are listed in Table 5.1, together with their parameter counts. The five NNUE models are dual-POV two-hidden architectures of increasing width; the linear, single-hidden FFNN, and dual-hidden FFNN models are dense baselines over the concatenated `[STM || opp]` features that isolate the contribution of the NNUE inductive bias. One further baseline — a capacity-matched dense single head with $H{=}512$ (to compare against a $B{=}4,\ H{=}256$ MoE at equal parameter count) — is planned but not yet trained; it will be added to Table 5.1 when available.

| Model | Architecture | $W$ / $H$ | Parameters | Test CE |
|---|---|---|---|---|
| Linear | concat `[STM‖opp]` → 3 | — | 5,067 | 0.766 |
| FFNN H64 | single hidden | 64 | 108,291 | 0.669 |
| FFNN H128 | single hidden | 128 | 216,579 | 0.654 |
| FFNN H256 | single hidden | 256 | 433,155 | 0.643 |
| FFNN2 H64 | dual hidden | 64 | 112,451 | 0.656 |
| FFNN2 H128 | dual hidden | 128 | 233,091 | 0.645 |
| FFNN2 H256 | dual hidden | 256 | 498,947 | 0.632 |
| NNUE W32 H64 | dual-POV two-hidden | 32 / 64 | 31,395 | 0.656 |
| NNUE W64 H128 | dual-POV two-hidden | 64 / 128 | 70,979 | 0.672 |
| NNUE W128 H128 | dual-POV two-hidden | 128 / 128 | 141,443 | 0.639 |
| NNUE W128 H256 | dual-POV two-hidden (reference) | 128 / 256 | 174,723 | 0.632 |
| NNUE W256 H512 | dual-POV two-hidden | 256 / 512 | 480,515 | 0.629 |

*Table 5.1 — Base models and their test cross-entropy on the fixed 1.5M-position test set.* The reference model `W128 H256` is the checkpoint whose head is replaced by expert heads in the MoE experiments, and whose gradients define the routing signal.

The Mixture-of-Experts models evaluated in Section 5.5 share the frozen L1 accumulator of the reference base model and add $K$ expert heads plus a dispatcher. Four routing families are compared: fixed piece-count routing, L1-activation clustering, sample-gradient clustering, and end-to-end soft/sparse gating, each at $K \in \{2,4,8,16\}$ where applicable, alongside an oracle upper bound that assigns each position to the expert with lowest loss.

### 5.1.5 Training Protocol

<span style="color: #808080;">[Training Protocol]</span> All models are trained with the Adam optimizer at a learning rate of $10^{-2}$ linearly decayed to $10^{-3}$ over 100 epochs. Each epoch consists of 512 batches of 10,000 positions drawn uniformly at random from the training set (5.12M positions per epoch), with the loss accumulated in float32 under bfloat16 AMP. The base models and the dense baselines share this protocol, which makes the parameter-count sweep in Table 5.1 a fair comparison of architecture rather than of training budget. Expert heads in the MoE experiments are fine-tuned from the reference base model, with a minimum of $7\times10^6$ routed samples per expert for the $H{=}256$ heads so that no expert underfits; the preliminary runs in Section 5.5 explicitly report when this budget was not met.

### 5.1.6 Evaluation Metrics

<span style="color: #808080;">[Evaluation Metrics]</span> Accuracy is measured on the held-out test set with four complementary metrics:

- **Cross-entropy (CE)** — the soft cross-entropy between the model's WDL distribution and the Lc0 teacher distribution, the primary loss and accuracy metric.
- **Mean Absolute Error (MAE)** — $|v(s) - \hat{v}(s)|$ where $v(s) = p_W - p_L$ is the scalar expected-reward summary of the predicted
  distribution; MAE is the quantity the search actually consumes.
- **Mean Squared Error (MSE)** and **coefficient of determination ($R^2$)** — the squared-error analogue and the fraction of target variance explained, reported to characterise the residual distribution of the scalar value.
- **Throughput (NPS)** — positions evaluated per second, measured both during training and at inference, as the efficiency proxy relevant to deployment.

	#todo MAE not really necessary?

Routing quality (for the dispatchers and the MoE models) is reported with Top-1/Top-2 classification accuracy, macro/weighted F1, and the Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) against the clustering pseudo-labels. End-to-end playing strength (Elo / ACPL) and on-device search throughput are out of scope for this chapter and are deferred to the deployment evaluation, as they require the full engine integration described in Chapter 4.

### 5.1.7 Experiment Pipeline Overview

<span style="color: #808080;">[Experiment Pipeline Overview]</span> The full battery follows a fixed four-stage pipeline, each stage exporting structured numerical tables and diagnostic plots: (i) **clustering** of three position representations — raw board features (1688-d), L1 activations (256-d), and sample-wise head gradients (48-d) — with mini-batch $k$-means at $k \in \{2,4,8,16\}$, DBSCAN, and a handcrafted piece-count baseline; (ii) **dispatcher
training** to map L1 activations (or the board) to bucket IDs, spanning parameter-free cosine-centroid routers, linear/MLP heads, and decision-tree/XGBoost baselines; (iii) **base-model training** as in Sections 5.1.4–5.1.5; and (iv) **MoE assembly and fine-tuning** of expert heads per partition. Each stage's results and plots are reported in Sections 5.2 through 5.5.

---

## 5.2 Clustering

### 5.2.1 Clustering Algorithm

<span style="color: #808080;">[Sampling]</span> The clustering is run on subsamples of the training split rather than on the full corpus of roughly 150 million positions. Exploratory experiments use one million positions drawn from the seed-0 training split (the subsample cached for the base model, with the held-out test positions excluded), and the final partitions are fitted on two million positions. In all cases the sample gradients are computed with respect to the head parameters of the trained base model ($W=128$, $H=256$), L2-normalised so that each vector lies on the unit hypersphere, and the clustering therefore operates on directions rather than magnitudes.

<span style="color: #808080;">[Representations]</span> Three candidate representations are clustered in parallel, each evaluated for a range of cluster counts $B$. The first is the raw board encoding, the side-to-move-ordered concatenation of the two 844-dimensional binary views, giving a 1688-dimensional vector. The second is the L1 activation space, the concatenation of the two 128-dimensional accumulator vectors after the side-to-move reorder, giving a 256-dimensional vector. The third is the proposed signal, the sample gradients. Because the flattened head gradient is very high-dimensional, it is projected to a 48-dimensional space through a fixed random (Gaussian) projection before clustering. The board and L1 representations serve as the activation-based references, while the gradients constitute the signal under test.

<span style="color: #808080;">[Algorithms]</span> The primary algorithm is Mini-Batch K-Means with $k$-means++ initialisation, a mini-batch size of $10\,000$, and a fixed random seed, run for $B \in \{2, 4, 8, 16\}$. As a density-based alternative that does not require $B$, DBSCAN is applied with `min_samples` fixed at $80$ and $\varepsilon$ selected on a quantile grid to recover a small number of clusters (at most eight). Because the nearest-neighbour density estimates in the native widths are not usable, the board and L1 representations are clustered in a 48-dimensional PCA of the data, whereas the gradient representation is clustered in its native 48-dimensional space. A handcrafted reference partition of eight bins is obtained by bucketing on the scalar piece count, mirroring the conventional NNUE bucketing. The final gradient-based partitions used for the mixture-of-experts are fitted on two million positions for $B \in \{2, 4, 8\}$.

### 5.2.2 Clustering Metrics

<span style="color: #808080;">[Metrics]</span> The quality of each partition is assessed with the diagnostics introduced in Section 3.5.5: the silhouette score, which measures how well each point fits its own cluster relative to the nearest other cluster; the within-cluster inertia, which measures compactness; and the pairwise cosine distance between cluster centroids, reported both as a mean and as a minimum, which serves as a proxy for the diversity of the learning signals captured by the buckets. Cluster balance is monitored through the size distribution of the buckets, since a bucket that is too small cannot support stable expert fine-tuning.

<span style="color: #808080;">[Agreement]</span> In addition, the agreement between partitions is quantified with the adjusted Rand index (ARI) and the normalised mutual information (NMI), both of which measure how well two assignments coincide while correcting for chance. These are used to compare the activation-based and gradient-based partitions with one another and with the piece-count reference. Finally, the stability of the gradient partition is assessed by refitting the clustering on a larger sample and checking that the silhouette and the centroid geometry are preserved as the number of samples increases.

### 5.2.3 Clustering Results

<span style="color: #808080;">[Comparison of representations]</span> Table 5.2 reports the mini-batch K-Means diagnostics across the three representations and the piece-count reference. Two findings stand out. First, the gradient-based partitions are markedly better separated than the activation-based ones: for $B=2$ the two gradient centroids are nearly antipodal, with a cosine distance of $1.94$ (a cosine similarity of $-0.94$), and even at $B=16$ the mean cosine distance remains above $1.0$. In contrast, the board and L1 centroids remain close to one another at every value of $B$, with cosine distances well below $0.3$. Second, the gradient partitions achieve the highest silhouette scores of any learned representation at every value of $B$, whereas the L1 silhouette collapses to near zero, and eventually negative, as $B$ grows.

| Representation | $B$ | Silhouette | Cosine dist. (mean) | Cosine dist. (min) | Min. share | Max. share |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| board | 2 | 0.110 | 0.235 | 0.235 | 0.313 | 0.687 |
| board | 4 | 0.073 | 0.221 | 0.120 | 0.074 | 0.431 |
| board | 8 | 0.040 | 0.257 | 0.043 | 0.062 | 0.285 |
| board | 16 | 0.016 | 0.288 | 0.061 | 0.017 | 0.159 |
| L1 | 2 | 0.153 | 0.174 | 0.174 | 0.275 | 0.725 |
| L1 | 4 | 0.109 | 0.208 | 0.115 | 0.104 | 0.541 |
| L1 | 8 | 0.003 | 0.203 | 0.069 | 0.044 | 0.249 |
| L1 | 16 | −0.007 | 0.244 | 0.057 | 0.015 | 0.173 |
| gradients | 2 | 0.250 | 1.938 | 1.938 | 0.441 | 0.559 |
| gradients | 4 | 0.196 | 1.229 | 0.607 | 0.151 | 0.361 |
| gradients | 8 | 0.191 | 1.126 | 0.367 | 0.086 | 0.182 |
| gradients | 16 | 0.134 | 1.049 | 0.123 | 0.034 | 0.131 |
| piece count | 8 | 0.606 | 0.000 | 0.000 | 0.069 | 0.187 |

*Table 5.2 — Mini-Batch K-Means clustering diagnostics for the three representations and the piece-count reference, on one million training positions. Silhouette is estimated on a $10\,000$-point subsample; the share columns give the smallest and largest bucket as a fraction of the sample.*

<span style="color: #808080;">[Piece-count reference]</span> The piece-count reference attains the highest silhouette score ($0.61$), but this is an artefact of its one-dimensional nature: contiguous, well-separated bins along a single scalar are trivial to recover. Its centroid cosine distance is identically zero, since all bucket means are positive one-dimensional numbers that point in the same direction. The piece-count partition therefore looks compact by silhouette but conveys nothing about the learning signal, and it should not be read as a well-formed candidate for expert specialisation.

<span style="color: #808080;">[DBSCAN]</span> The density-based alternative is summarised in Table 5.3. Since the three representations are clustered in different spaces — the board and L1 encodings in a 48-dimensional PCA of the data and the gradients in the native 48 dimensions — the three rows are reported per-representation and should not be compared directly. DBSCAN recovers very few clusters from each representation — three from the board encoding, two from the L1 activations, and five from the gradients — and it flags a very large fraction of the fit sample as noise: $39.6\%$ for the board encoding, $57.2\%$ for the L1 activations, and $75.2\%$ for the gradients. The recovered clusters are heavily imbalanced, each dominated by a single large core with one or more small satellites, and their silhouette scores, while superficially higher than those of K-Means on the board and L1 data, reflect that dominant core rather than a meaningful multimodal structure. This is consistent with the expectation set out in Section 3.5.3 that density-based clustering is unreliable in a high-dimensional gradient space, and it motivates the use of a fixed $B$ for the mixture-of-experts.

| Representation | Clusters | Bucket sizes | Noise (fit subset) | Silhouette |
| :--- | :-: | :--- | :-: | :-: |
| board | 3 | 78.2% / 10.5% / 11.3% | 39.6% | 0.172 |
| L1 | 2 | 85.6% / 14.4% | 57.2% | 0.217 |
| gradients | 5 | 21.1% / 23.1% / 7.8% / 40.2% / 7.7% | 75.2% | 0.152 |

*Table 5.3 — DBSCAN results, fitted on a $100\,000$-point subsample with `min_samples`$=80$ and $\varepsilon$ selected on a quantile grid. The board and L1 runs are fit in a 48-dimensional PCA of the data, the gradient run in the native 48 dimensions, so the rows are not directly comparable across representations. Bucket sizes are the cluster shares of the full one-million-position assignment; the noise column gives the fraction of the $100\,000$-point fit flagged as noise.*

<span style="color: #808080;">[Partition agreement]</span> Table 5.4 reports the agreement between the partitions at $B=8$. The board and L1 partitions agree only moderately with each other (NMI $0.24$) and agree weakly with the piece-count reference, suggesting that even activation-based partitions capture structure beyond the game phase. The gradient partition, by contrast, is nearly orthogonal to every other partition: its agreement with the board and L1 assignments is at the level of chance (ARI $0.06$–$0.08$), and its agreement with the piece-count bucketing is essentially zero (ARI $0.04$, NMI $0.09$). This is precisely the behaviour the method is designed to induce: clustering the learning signal recovers a partition that does not simply reproduce the board geometry, the internal representation, or the conventional game-phase bucketing.

| Pair | ARI | NMI |
| :--- | :-: | :-: |
| board vs. L1 | 0.124 | 0.236 |
| board vs. gradients | 0.057 | 0.063 |
| L1 vs. gradients | 0.078 | 0.150 |
| board vs. piece count | 0.189 | 0.303 |
| L1 vs. piece count | 0.147 | 0.290 |
| gradients vs. piece count | 0.042 | 0.088 |

*Table 5.4 — Agreement (adjusted Rand index and normalised mutual information) between partitions at $B=8$, computed on one million positions. Higher values indicate closer agreement; the chance baseline is $0$.*

<span style="color: #808080;">[Stability]</span> The final gradient-based clustering, fitted on two million positions for $B \in \{2, 4, 8\}$, reproduces the structure observed on one million positions. The silhouette scores are $0.250$, $0.215$, and $0.179$ for $B=2$, $4$, and $8$ respectively, within $0.02$ of the one-million-position estimates ($0.250$, $0.196$, and $0.191$), and the centroid geometry is preserved, though less tightly at $B=4$: the mean off-diagonal centroid cosine is $-0.938$, $-0.307$, and $-0.128$, against $-0.938$, $-0.229$, and $-0.126$ on the smaller sample — the $B=2$ and $B=8$ centroids are essentially unchanged, while the $B=4$ centroids drift from $-0.229$ to $-0.307$. The buckets remain well balanced at each value of $B$, ranging from $55.8\%/44.2\%$ at $B=2$ to between $10.8\%$ and $16.0\%$ at $B=8$. The partition is therefore stable with respect to the sample size. As $B$ grows the centroids spread out and their mean pairwise similarity approaches zero: at $B=2$ they are nearly antipodal, while at $B=8$ the pairwise centroid cosines range from strongly negative (near $-0.95$) to moderately aligned (near $+0.72$), consistent with a partition that tiles a single dense gradient manifold rather than isolating well-separated modes.

<div align="center">
    <img src="RESULTS/clustering/plots/previous_gradient_2m/silhouette.png" width="600">
</div>

<span style="color: #808080;">[Visualisation]</span> The qualitative evidence supports the quantitative picture. Projections of the gradient space onto the first principal components, and the corresponding t-SNE and UMAP embeddings, show a single connected cloud along which the clusters form contiguous regions, consistent with the low silhouette values and the gradual loss of centroid separation as $B$ increases. In contrast, the L1 and board projections show clusters that are poorly separated and strongly overlapping, mirroring their near-zero cosine distances. The PCA projection of the gradient-based $B=2$ partition in particular shows the two clusters lying on opposite sides of the origin, in line with the antipodal centroids reported above.

<div align="center">
    <img src="RESULTS/clustering/plots/pca_2d_gradients.png" width="600">
</div>

<div align="center">
    <img src="RESULTS/clustering/plots/tsne_2d_gradients.png" width="600">
</div>

<span style="color: #808080;">[Interpretation]</span> Taken together, the results indicate that the gradient-based partition is the most promising candidate for expert specialisation. The antipodal centroids at $B=2$ confirm that the two buckets correspond to opposite directions in head-parameter space — positions whose gradients pull the head in contrary ways — which is exactly the kind of diversity the method seeks. The near-zero agreement with the piece-count and activation-based partitions further suggests that these buckets do not correspond to coarse game-phase or geometric categories, but to a genuinely different structure grounded in the learning dynamics. The implication for the mixture-of-experts is that the gradient partition should induce expert task vectors that are more distinct from one another than those produced by any activation-based bucketing, a hypothesis examined in Section 5.5.

---

## 5.3 Dispatcher

### 5.3.1 Dispatcher Model

<span style="color: #808080;">[Model]</span> The dispatcher is a lightweight classifier that maps a position to a bucket index. Its input is the concatenated L1 activation vector, the side-to-move-ordered pair of two $128$-dimensional accumulator views ($256$ dimensions), computed once by the frozen base model and cached for the entire dataset. The primary dispatcher is a single linear layer that produces $B$ logits from the $2W$-dimensional input, with $2W \times B + B$ parameters ($2056$ parameters for $B = 8$) and the softmax used during training is discarded at inference, where the predicted bucket is simply the argmax of the logits. Three alternatives are evaluated to test whether additional capacity is warranted: a single-hidden MLP with $h \in \{32, 64, 128\}$ hidden units, a decision tree (max-depth $12$), and *XGBoost* ($50$ trees of depth $6$). A parameter-free centroid router is evaluated as an additional reference: each position is assigned to the bucket whose $k$-means centroid is nearest in cosine similarity, using either the $1688$-dimensional board-state centroids or the $256$-dimensional L1 centroids.

<span style="color: #808080;">[Why a learned dispatcher]</span> The distinction between the offline clustering and the dispatcher is central to the method. The partition is defined in sample-gradient space, which is unavailable at inference time — computing a gradient would require the teacher evaluation of the position — so the partition cannot be applied directly to an unseen position. The dispatcher instead predicts the bucket from L1 activations, a feature that is already produced by the incremental accumulator update during the normal forward pass. The centroid routers serve as the geometric upper bound of what the partition allows to be recovered from the representation alone: they reproduce the partition exactly as far as the representation's geometry permits, without any learned non-linearity. The learned dispatcher is trained to approximate the partition with cross-entropy against the $k$-means labels, using Adam with a learning rate of $10^{-2}$ decayed to $10^{-3}$, a batch size of $1024$, eight epochs, and a $90/10$ train/validation split on one million positions. Note that the $k$-means labels are computed on the full one-million-position subsample before the split, so the validation accuracy reported below is transductive rather than a fully independent estimate. Because the L1 activations are cached, each router trains in a few seconds to a minute on a GPU.

### 5.3.2 Dispatcher Metrics

<span style="color: #808080;">[Agreement metrics]</span> The dispatcher is scored against the pseudo-ground-truth cluster labels produced by Mini-Batch $k$-Means, using the agreement metrics introduced in Section 3.5.5. Top-1 and top-2 accuracy report how often the correct bucket is the single highest (or among the two highest) predicted logits; the macro-F1 and weighted-F1 scores summarise per-bucket precision and recall, with the weighted variant reflecting the imbalanced bucket sizes; and the adjusted Rand index (ARI) and normalised mutual information (NMI) quantify the agreement between the predicted and reference partitions while correcting for chance. A normalised confusion matrix and per-class precision-recall curves are produced for each router.

<span style="color: #808080;">[Baselines]</span> Two trivial baselines are reported alongside every learned router. The *chance* baseline predicts uniformly at random, achieving $1/B$; the *dummy* baseline always predicts the majority cluster. Because the buckets are imbalanced, the dummy baseline is the more demanding reference, ranging from $0.727$ at $B=2$ to $0.172$ at $B=16$ for the L1 partitions, and from $0.560$ to $0.134$ for the gradient partitions. The piece-count rule-based router is evaluated separately by its ARI/NMI alignment with the learned partitions, since it defines its own eight buckets rather than predicting an existing partition.

### 5.3.3 Dispatcher Results

<span style="color: #808080;">[Centroid routers]</span> Table 5.5 reports the two parameter-free centroid routers. Both reproduce the Euclidean $k$-means partition well above chance, but the L1 centroids do so markedly better than the board-state centroids: the L1 router attains a top-1 accuracy of $0.979$ at $B=2$, falling only to $0.940$ at $B=16$, while the board router falls from $0.946$ to $0.817$ over the same range. The corresponding ARI values track the same ordering ($0.916 \to 0.892$ for L1 versus $0.791 \to 0.620$ for the board), and the NMI values show the L1 agreement actually *increasing* with $B$ ($0.846 \to 0.884$) while the board agreement peaks near $B=8$ and declines. The L1 partition is therefore recoverable almost entirely through a cosine-nearest-centroid rule, confirming that the activation-based buckets are geometrically well separated, whereas the board-state partition is only partially expressible from its centroid geometry. Because the routers assign points by cosine proximity but are scored against the Euclidean $k$-means labels, these figures are a lower bound on recoverability: scoring against a spherical (cosine) $k$-means, which matches the routing rule, would raise the agreement further.

| Representation | $B$ | Top-1 | ARI | NMI |
| :--- | :-: | :-: | :-: | :-: |
| board | 2 | 0.946 | 0.791 | 0.670 |
| board | 4 | 0.898 | 0.731 | 0.700 |
| board | 8 | 0.875 | 0.701 | 0.763 |
| board | 16 | 0.817 | 0.620 | 0.723 |
| L1 | 2 | 0.979 | 0.916 | 0.846 |
| L1 | 4 | 0.963 | 0.909 | 0.856 |
| L1 | 8 | 0.948 | 0.893 | 0.872 |
| L1 | 16 | 0.940 | 0.892 | 0.884 |

*Table 5.5 — Parameter-free centroid routers (`argmin` cosine similarity to the $k$-means centroids), scored against the Euclidean Mini-Batch $k$-Means labels on one million positions. Chance is $1/B$; the majority dummy ranges from $0.727$ ($B=2$) to $0.172$ ($B=16$) for the L1 labels.*

<div align="center">
    <img src="RESULTS/dispatcher/centroid/plots/accuracy.png" width="600">
</div>

<span style="color: #808080;">[Learned L1 dispatchers]</span> Table 5.6 reports the learned routers trained to predict the L1 $k$-means buckets from L1 activations. Recovery of the L1 partition is essentially trivial for the neural routers: the linear dispatcher reaches $0.993$ at $B=2$ and $0.967$ at $B=16$, and the single-hidden MLP is statistically indistinguishable from it ($0.994 \to 0.971$). XGBoost approaches the neural routers ($0.986 \to 0.891$), while the single decision tree is the weakest of the four ($0.959 \to 0.704$), although even it clears the dummy baseline by a wide margin. The ARI values confirm the same ranking, with the linear and MLP routers preserving more than $0.94$ of the partition structure up to $B=16$, XGBoost $0.80$, and the decision tree $0.51$. Since the buckets are imbalanced, the macro-F1 (not shown) sits slightly below the weighted-F1 at every $B$, but the ordering across routers is unchanged.

| Router | $B$ | Top-1 | ARI |
| :--- | :-: | :-: | :-: |
| linear | 2 | 0.993 | 0.972 |
| linear | 4 | 0.988 | 0.969 |
| linear | 8 | 0.978 | 0.952 |
| linear | 16 | 0.967 | 0.937 |
| MLP ($h{=}64$) | 2 | 0.994 | 0.973 |
| MLP ($h{=}64$) | 4 | 0.990 | 0.975 |
| MLP ($h{=}64$) | 8 | 0.980 | 0.957 |
| MLP ($h{=}64$) | 16 | 0.971 | 0.945 |
| decision tree | 2 | 0.959 | 0.835 |
| decision tree | 4 | 0.906 | 0.788 |
| decision tree | 8 | 0.794 | 0.595 |
| decision tree | 16 | 0.704 | 0.512 |
| XGBoost | 2 | 0.986 | 0.942 |
| XGBoost | 4 | 0.967 | 0.921 |
| XGBoost | 8 | 0.930 | 0.850 |
| XGBoost | 16 | 0.891 | 0.799 |

*Table 5.6 — Learned L1-to-L1 dispatchers on one million positions ($90/10$ split), scored against the L1 Mini-Batch $k$-Means labels. Chance is $1/B$; the majority dummy is $0.727$, $0.542$, $0.248$, and $0.172$ for $B=2,4,8,16$ respectively.*

<div align="center">
    <img src="RESULTS/dispatcher/l1/plots/accuracy.png" width="600">
</div>

#todo legend covers bars - regenerate plot

<span style="color: #808080;">[Hidden-width sweep]</span> Varying the hidden width of the MLP router between $32$, $64$, and $128$ units makes essentially no difference on the L1 target: at every value of $B$ the three widths agree within $0.003$ in top-1 accuracy (for instance $0.993$–$0.994$ at $B=2$ and $0.970$–$0.973$ at $B=16$), and the ARI values are likewise indistinguishable. The L1 partition is linearly recoverable to begin with, so additional non-linear capacity has nothing to exploit.

<span style="color: #808080;">[Gradient target]</span> The picture reverses when the routers are trained to predict the *gradient* $k$-means buckets from L1 activations, as shown in Table 5.7. Here recovery is much harder: top-1 accuracy falls from $0.63$–$0.64$ at $B=2$ to $0.47$–$0.49$ at $B=16$, and the ARI remains at or below $0.37$ throughout. The routers do exceed the dummy baseline — substantially so at the larger values of $B$ (for instance $0.47$–$0.49$ against a dummy of $0.134$ at $B=16$) — but at $B=2$ the MLP routers barely clear the $0.560$ majority label, and their ARI of $0.07$–$0.08$ indicates agreement only marginally above chance. Hidden width again matters little: the three widths are within $0.01$ of one another at every $B$. A separate linear gradient dispatcher, trained earlier on the two-million-position gradient cache at $B \in \{2,4,8\}$, reaches validation accuracy $0.60$, $0.56$, and $0.48$; it is omitted from Table 5.7 because it is trained on a different subsample than the MLP sweep reported here. This is the central negative result of the dispatcher study: **L1 activations are a weak proxy for gradient direction**, and no amount of learned non-linearity of the size tested here recovers the gradient partition. The gradient clustering therefore carries routing information that is not present in the activation representation, which is precisely the motivation for gradient-based routing in the first place — but it also means that the deployable L1 dispatcher cannot faithfully reproduce that partition, a limitation quantified further in Section 5.5.

#todo DeepSeek (resolved): Run 2's linear gradient dispatcher (2M rows) is now noted in the prose as a separate setup from the MLP sweep.

| Target | $h$ | $B$ | Top-1 | ARI |
| :--- | :-: | :-: | :-: | :-: |
| gradient | 32 | 2 | 0.642 | 0.081 |
| gradient | 64 | 2 | 0.632 | 0.069 |
| gradient | 128 | 2 | 0.642 | 0.081 |
| gradient | 32 | 4 | 0.580 | 0.226 |
| gradient | 64 | 4 | 0.587 | 0.225 |
| gradient | 128 | 4 | 0.588 | 0.228 |
| gradient | 32 | 8 | 0.522 | 0.362 |
| gradient | 64 | 8 | 0.525 | 0.366 |
| gradient | 128 | 8 | 0.530 | 0.369 |
| gradient | 32 | 16 | 0.475 | 0.367 |
| gradient | 64 | 16 | 0.483 | 0.368 |
| gradient | 128 | 16 | 0.490 | 0.369 |

*Table 5.7 — MLP dispatchers (hidden width $h$) trained to predict the gradient $k$-means buckets from L1 activations, on one million positions ($90/10$ split). Chance is $1/B$; the majority dummy is $0.560$, $0.361$, $0.184$, and $0.134$ for $B=2,4,8,16$ respectively.*

<div align="center">
    <img src="RESULTS/dispatcher/mlp/plots/accuracy_gradient.png" width="600">
</div>

<div align="center">
    <img src="RESULTS/dispatcher/mlp/plots/confusion_gradient.png" width="600">
</div>

#todo colorbar covers confusion matrix - regenerate plot

<span style="color: #808080;">[Error structure]</span> The macro-F1 consistently lags the top-1 accuracy by a growing margin as $B$ increases (for instance $0.40$–$0.42$ against $0.47$–$0.49$ at $B=16$), indicating that the smaller buckets are disproportionately misclassified: the majority buckets dominate the top-1 accuracy, while the per-bucket precision and recall are dragged down by the rare buckets. The learned routers are therefore a poor *point* approximation of the gradient partition, although — as the ARI of $0.37$ at $B=16$ shows — they retain a coarse, non-trivial signal that exceeds the majority and chance baselines at every value of $B$.

<span style="color: #808080;">[Piece-count router]</span> The fixed eight-interval piece-count router is evaluated purely by its alignment with the learned partitions, since it predicts its own bucket definitions. Table 5.8 shows that it aligns weakly with the L1 partitions (ARI $0.11$–$0.17$) and essentially not at all with the gradient partitions (ARI $0.01$–$0.05$). The piece-count buckets themselves are reasonably balanced, with the eight intervals covering between $6.9\%$ and $18.7\%$ of the sample. This mirrors the clustering findings of Section 5.2.3: piece count is a one-dimensional game-phase signal that shares little structure with either activation- or gradient-based buckets, and it cannot serve as a substitute for the learned routing.

| Target | $B$ | ARI | NMI |
| :--- | :-: | :-: | :-: |
| L1 | 2 | 0.108 | 0.254 |
| L1 | 4 | 0.168 | 0.358 |
| L1 | 8 | 0.147 | 0.290 |
| L1 | 16 | 0.126 | 0.314 |
| gradient | 2 | 0.008 | 0.014 |
| gradient | 4 | 0.014 | 0.037 |
| gradient | 8 | 0.042 | 0.088 |
| gradient | 16 | 0.045 | 0.120 |

*Table 5.8 — Alignment (ARI/NMI) between the fixed eight-bucket piece-count router and the L1 and gradient $k$-means partitions at each $B$, computed on one million positions.*

<div align="center">
    <img src="RESULTS/dispatcher/piececount/plots/alignment.png" width="600">
</div>

<span style="color: #808080;">[Overhead]</span> The computational and memory overhead of the dispatcher is negligible relative to the evaluation function. The linear router holds $2W \times B + B$ parameters — $2\,056$ for $B=8$ and $4,112$ for $B=16$ — and the largest MLP variant tested ($h=128$) still holds fewer than $40,000$ parameters, orders of magnitude below the shared L1 accumulator. At inference the dispatcher adds a single matrix-vector product over $B$ outputs followed by an argmax, both inexpensive integer operations that precede the (unchanged) head forward pass; on the target hardware this is bounded by the cost of a $2W \times B$ multiply-accumulate. Training is similarly cheap, completing in seconds per router because the L1 activations are precomputed.

---

## 5.4 Base Model

### 5.4.1 Base Model Architecture and Training

\[Describe the base NNUE architecture, training procedure, WDL targets, loss function, and relevant hyperparameters.\]

\[Describe the simpler neural baselines used to establish the capacity and computational requirements of the evaluation function.\]

### 5.4.2 Base Model Metrics

\[Report test soft cross-entropy on WDL predictions and MAE on expected value.\]

\[Report model size, parameter count, inference cost, and, where applicable, playing-strength metrics such as Elo, ACPL, depth, or nodes per second.\]

\[For the base NNUE, evaluate performance as a function of training-set size to identify the point at which additional data no longer provides substantial benefit or where overfitting becomes relevant.\]

### 5.4.3 Base Model Results

\[Present the results for the linear model, shallow FFNN, deeper FFNN, and base NNUE.\]

\[Present the dataset-size experiment and identify the training-data regime used for the subsequent MoE experiments.\]

\[Establish the base NNUE as the reference model against which the specialized models are evaluated.\]

---

## 5.5 Mixture of Experts

### 5.5.1 MoE Architecture and Training

\[Describe the shared L1 representation, expert heads, bucket assignment, expert-training procedure, and inference-time routing.\]

\[Describe the activation-based and gradient-based bucketing approaches.\]

\[Separate experiments with fixed $B$ from experiments in which $B$ is varied.\]

\[Explain how the available training data is partitioned among experts and how the dataset-size constraints identified for the base model affect expert training.\]

### 5.5.2 MoE Metrics

\[Evaluate WDL cross-entropy and expected-value MAE for the complete MoE.\]

\[Report per-bucket cross-entropy to determine whether individual experts specialize relative to the base head.\]

\[Compare Oracle routing with learned dispatcher routing where appropriate.\]

\[Report model size, memory footprint, inference latency, nodes per second, and other relevant computational costs.\]

\[For the complete engine, report the playing-strength metrics defined by the evaluation protocol.\]

### 5.5.3 MoE Results

\[Present the results for activation-based bucketing with fixed $B$.\]

\[Present the results for activation-based bucketing with variable $B$.\]

\[Present the results for gradient-based bucketing with fixed $B$.\]

\[Present the results for gradient-based bucketing with variable $B$.\]

\[Compare the resulting expert specialization, predictive quality, and computational cost across the different bucketing strategies.\]

\[Distinguish the effect of the partition itself from the effect of the learned dispatcher by comparing Oracle and learned routing where applicable.\]

---

## 5.6 Summary of Experimental Findings

\[Summarize the main empirical findings without introducing new analysis.\]

\[State which experimental observations will be examined in greater depth in the Discussion, including clustering stability, dispatcher approximation, expert specialization, predictive performance, and computational trade-offs.\]