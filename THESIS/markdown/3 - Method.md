# Method: Unsupervised Bucketing via Sample Gradients

#idea The dispatcher needs only to be called on half of the activations of the dual *accumulator layer*, making it [W]x3 parameters in size.

#idea it's not necessary to perform clustering on the whole training set

#idea to avoid splitting the data too much, train the expert also with data just outside the cluster.

#todo maybe move the definitions elsewhere, and the sample gradient up?

#todo quantization and pruning 

#todo remember that when we partition the dataset, each expert has less data, but it still has to be "enough" (performance vs number of data)

#todo emphasize the fact that the clustering with sample gradients is not as easy to transpose to inference time as L1-based clustering could be, but with the dispatcher it becomes similarly fast.

#todo cosine similarity and spectral clustering

---

## 3.1 Method Overview

<span style="color: #808080;">[Proposed Method]</span> This chapter presents the proposed method for unsupervised state-space bucketing via sample gradients. The central idea is to use the learning signal itself—the per-sample gradients of the loss with respect to the head parameters—as the basis for partitioning the state space, rather than relying on handcrafted features or hidden-layer activations. 

<span style="color: #808080;">[Steps 1-2]</span> The method unfolds through a carefully orchestrated sequence designed to balance computational tractability with the stringent efficiency demands of on-device inference. It begins by training a base NNUE model on the complete dataset, which establishes a shared L1 representation and provides a stable reference point for subsequent gradient computations. From this foundation, normalized per-sample gradients are extracted with respect to the trainable head parameters, effectively capturing the direction in which each position would push the head during optimization.

<span style="color: #808080;">[Steps 3-4]</span> These gradient vectors then become the substrate for clustering, yielding a partition of the state space into $B$ distinct buckets that group together positions exhibiting similar learning dynamics. To enable fast routing at inference time without the prohibitive cost of recomputing gradients, a lightweight linear dispatcher is trained to predict bucket assignments directly from the frozen L1 activations. 

<span style="color: #808080;">[Steps 5]</span> The pipeline culminates in fine-tuning a dedicated expert head on each bucket ( #todo decided by the dispatcher), initialized from the base model while keeping the L1 weights frozen. The end result is a mixture-of-experts NNUE architecture in which each expert develops specialised competence over a coherent region of the state space, guided by a dispatcher that introduces negligible overhead to the evaluation function.

## 3.2 Notation and Definitions

<span style="color: #808080;">[Notation]</span> We introduce here the formal notation used throughout this chapter and the remainder of the thesis. The notation is organized into sets, scalars, vectors and matrices, functions, and key operators.

### 3.2.1 Sets

| Symbol                                                  | Description                                                                                                            |
| :------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------- |
| $\mathcal{S}$                                           | The space of all possible chess positions (states).                                                                    |
| $\mathcal{D}$                                           | The training dataset of positions with labels: $\mathcal{D} = \{(s_i, v_i)\}_{i=1}^{N}$.                               |
| $\mathcal{P} = \{\mathcal{D}_1, \dots, \mathcal{D}_B\}$ | A partition of the state space into $B$ buckets, where each $\mathcal{D}_i \subset \mathcal{S}$ is a non-empty subset. |
| $\Theta$                                                | The space of all model parameters (weights).                                                                           |

### 3.2.2 Scalars

| Symbol               | Description                                                     |
| :------------------- | :-------------------------------------------------------------- |
| $B$                  | The number of experts (buckets).                                |
| $N$                  | The total number of positions in the training dataset.          |
| $N_i$                | The number of positions in bucket $\mathcal{D}_i$.              |
| $h$                  | The hidden dimension of the NNUE accumulator.                   |
| $d_{\text{in}}$      | The input dimension of the NNUE accumulator ([d_in] in this work). |
| $v_i \in \mathbb{R}$ | The scalar label (expected reward) for position $s_i$.          |
| $\eta$               | The learning rate used for gradient updates.                    |

### 3.2.3 Vectors and Matrices

| Symbol                                           | Type   | Description                                                                                                                                                           |
| :----------------------------------------------- | :----- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $W_{L1} \in \mathbb{R}^{d_{\text{in}} \times h}$ | Matrix | Weights of the first layer (accumulator), frozen after base training.                                                                                                 |
| $W_{L2} \in \mathbb{R}^{h \times h}$             | Matrix | Weights of the second layer.                                                                                                                                          |
| $W_{out} \in \mathbb{R}^{h \times 3}$            | Matrix | Weights of the output layer (3 logits for WDL).                                                                                                                       |
| $w_{\text{base}} \in \mathbb{R}^{P}$             | Vector | All parameters of the base model: $w_{\text{base}} = \text{vec}(W_{L1}, W_{L2}^{base}, W_{out}^{base})$.                                                              |
| $\theta_i \in \mathbb{R}^{P}$                    | Vector | All parameters of the expert model $i$: $\theta_i = \text{vec}(W_{L1}, W_{L2}^{(i)}, W_{out}^{(i)})$.                                                                 |
| $\delta_i \in \mathbb{R}^{P_{\text{head}}}$      | Vector | The task vector for expert $i$: $\delta_i = \theta_i^{\text{head}} - \theta_{\text{base}}^{\text{head}}$, where $\theta^{\text{head}} = \text{vec}(W_{L2}, W_{out})$. |
| $\Delta_i \in \mathbb{R}^{P_{\text{head}}}$      | Vector | The **per-sample gradient** for position $s_i$: $\Delta_i = \nabla_{w_{\text{head}}} \mathcal{L}(\hat{f}_{w_{\text{base}}}(s_i), v_i)$.                               |
| $x_i \in \{0,1\}^{d_{\text{in}}}$                | Vector | The sparse binary feature vector for position $s_i$.                                                                                                                  |
| $h_i = W_{L1} \cdot x_i \in \mathbb{R}^{h}$      | Vector | The accumulator output (L1 activation) for position $s_i$.                                                                                                            |

### 3.2.4 Functions and Models

| Symbol | Description                                                                                           |
| :--- | :--- |
| $f_w: \mathcal{S} \rightarrow \mathbb{R}^3$ | The NNUE evaluation function parameterized by weights $w$, producing WDL logits.                      |
| $\hat{f}_w: \mathcal{S} \rightarrow \mathbb{R}$ | The NNUE scalar value function: $\hat{f}_w(s) = \text{softmax}(f_w(s))_W - \text{softmax}(f_w(s))_L$. |
| $g_\phi: \mathcal{S} \rightarrow \{1, \dots, B\}$ | The dispatcher function, parameterized by $\phi$, mapping a position to a bucket index.               |
| $\mathcal{L}(y, v)$ | The loss function, typically soft cross-entropy on WDL probabilities.                                 |
| $\mathcal{L}_{\text{acc}}(y, v)$ | The accumulated loss over a batch or epoch.                                                           |

### 3.2.5 Key Operators

| Symbol                                                     | Description                                                           |
| :--------------------------------------------------------- | :-------------------------------------------------------------------- |
| $\text{vec}(\cdot)$                                        | The vectorization operator, flattening a matrix into a column vector. |
| $\nabla_w\,  \mathcal{L}$                                  | The gradient of the loss with respect to the parameters $w$.          |
| $\| \cdot \|$                                              | The Euclidean (L2) norm.                                              |
| $\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$     | The softmax function over logits $z$.                                 |
| $d_{\text{cos}}(u, v) = 1 - \frac{u \cdot v}{\|u\| \|v\|}$ | The cosine distance between two vectors $u$ and $v$.                  |

### 3.2.6 Relationship Between $\Delta_i$ and $\delta_i$

<span style="color: #808080;">[Two Concepts]</span> A crucial distinction in this work is between **per-sample gradients** $\Delta_i$ and **task vectors** $\delta_i$. These two concepts are similar to each other, but it is worth emphasizing that the task vector $\delta_i = \theta_i - \theta_{\text{base}}$ is the difference in model parameters between before and after the fine-tuning, while the sample gradient $\Delta_i = \nabla_{w_{\text{head}}} \mathcal{L}(\hat{f}_{w_{\text{base}}}(s_i), v_i)$ is the gradient of the loss with respect to the head parameters, evaluated at a single position $s_i$ using the base model. 

<span style="color: #808080;">[Central Hypothesis]</span> The central hypothesis of this work is that clustering positions by their $\Delta_i$ yields a partitioning for which the resulting learning dynamics are maximally diverse - each expert specializes in a distinct region of the state space - and that the dispatcher is able to approximate this partitioning with enough accuracy to preserve its general structure.

---

## 3.3 Step 1: Train the Base Model

<span style="color: #808080;">[Base Model]</span> The first step of our method is to train a **base evaluation model** that will serve as the foundation for all subsequent steps. This model provides two essential functions: it supplies the reference point from which the sample gradients are computed, and it provides the frozen L1 representation used by the dispatcher at inference time.

### 3.3.1 Model Architecture

<span style="color: #808080;">[Architecture]</span> The base model follows the NNUE architecture described in Section 2.1.3: a sparse accumulator layer $W_{L1}$ that maps a binary feature representation to a hidden state $h \in \mathbb{R}^h$, followed by a small fully-connected head $(W_{L2}, W_{out})$ that produces WDL logits. The architecture is kept deliberately small to reflect the resource constraints of the target hardware—specifically, a hidden dimension of $h = [W]$ for the accumulator and $H = [H]$ for the L2 layer, resulting in approximately [n_params] trainable parameters.

### 3.3.2 Training Objective

<span style="color: #808080;">[Loss Choice]</span> The model is trained to minimize the **soft cross-entropy loss** between its predicted WDL distribution and the teacher labels provided by Lc0 (see Section 4.1.4). For a batch of $N$ positions with teacher probabilities $\hat{p}_i = (\hat{P}_i(W), \hat{P}_i(D), \hat{P}_i(L))$ and model outputs $p_i = (P_i(W), P_i(D), P_i(L))$, the loss is:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ \hat{P}_i(W) \log P_i(W) + \hat{P}_i(D) \log P_i(D) + \hat{P}_i(L) \log P_i(L) \right]$$

#note for me: Soft cross-entropy (or soft-target cross-entropy) is a generalized loss function in machine learning that uses full probability distributions instead of strict "hard" one-hot vectors for target labels

<span style="color: #808080;">[Why Soft-CE]</span> This choice of loss is deliberate. Unlike mean squared error on a scalar value (centipawns or expected reward $v = P(W) - P(L)$), the cross-entropy loss encourages the model to match the **full outcome distribution** rather than just its mean. This provides a richer training signal and naturally handles the non-linear relationship between WDL probabilities and the scalar evaluation used during search.

### 3.3.3 Training Protocol

<span style="color: #808080;">[Training Protocol]</span> The base model is trained on the full dataset of approximately [dataset_size] positions using the Adam optimiser with a learning rate of [lr_start], linearly decayed to [lr_end] over the course of training. We use a batch size of [batch_size] and train until convergence on the held-out test set ([test_fraction] of the total dataset). All training is performed in PyTorch on a DGX Nvidia Spark GPU.

### 3.3.4 The Role of the Base Model

<span style="color: #808080;">[Gradient Reference]</span> Once trained, the base model $w_{\text{base}} = (W_{L1}, W_{L2}^{\text{base}}, W_{out}^{\text{base}})$ serves several critical functions in our pipeline, one of which is as reference point for gradients: For each position $s_i$, we compute the sample gradient $\Delta_i$ evaluated at the base model. This gradient represents the direction in which the head would need to move to better fit that specific position—the *learning signal* that drives our bucketing.

<span style="color: #808080;">[Frozen Routing]</span> An other important role of the base model is as frozen representation for routing. The L1 weights $W_{L1}$ are frozen after base training and are never updated during expert fine-tuning. This ensures that all experts operate on the same shared representation, and that the dispatcher can rely on stable L1 activations $h = W_{L1} \cdot x$ as input features.

<span style="color: #808080;">[Expert Init]</span> Finally, the base model is of course also the starting point for expert fine-tuning. Each expert head $(W_{L2}^{(i)}, W_{out}^{(i)})$ is initialised from the base head $(W_{L2}^{\text{base}}, W_{out}^{\text{base}})$ before being fine-tuned on its assigned bucket. 

#note here is where the main talk about the "frozen weights" should happen

### 3.3.5 Why Freeze L1?

<span style="color: #808080;">[Fixed representation space]</span> Freezing the L1 layer is a deliberate design choice. The accumulator is the most expensive component of the NNUE architecture in terms of parameter count, and updating it during fine-tuning would be computationally prohibitive. More importantly, freezing L1 ensures that the representation space remains the same for all experts: the dispatcher, trained on L1 activations, can reliably route positions without needing to account for different representations. This stability is essential for the lightweight inference pipeline, where the dispatcher must operate with negligible overhead.

---

## 3.4 Step 2: Compute Sample Gradients

<span style="color: #808080;">[Sample Gradients]</span> With the base model trained, the second step is to compute, for each position in the dataset, the sample gradient of the loss with respect to the head parameters. These gradients encode the direction in which the head parameters would need to move to improve the prediction for each individual position. They represent the *learning signal* that we will later use for bucketing.

### 3.4.1 Definition of Sample Gradient

<span style="color: #808080;">[Definition]</span> For each position $s_i$ in the dataset $\mathcal{D} = \{(s_i, v_i)\}_{i=1}^N$, the sample gradient is defined as:

$$\Delta_i = \nabla_{w_{\text{head}}} \mathcal{L}(\hat{f}_{w^{\text{base}}}(s_i), v_i)$$

where:

- $w_{\text{head}} = \text{vec}(W_{L2}, W_{out})$ is the vector of all head parameters (the L2 weights and biases, and the output weights and biases)
- $\hat{f}_{w_{\text{base}}}$ is the base model evaluated at the current weights
- $\mathcal{L}$ is the soft cross-entropy loss on WDL probabilities
- $v_i = (p_W, p_D, p_L)$ is the teacher label for position $s_i$

<span style="color: #808080;">[At Base Model]</span> The gradient is computed at the base model $w_{\text{base}}$, before any fine-tuning occurs. It represents the instantaneous direction in weight space that would reduce the loss on that specific position, independent of all other positions.

### 3.4.2 Implementation

<span style="color: #808080;">[Implementation]</span> In practice, the gradient computation leverages PyTorch's `torch.autograd.grad` function, which efficiently computes sample gradients for an entire batch of inputs in a single operation. Crucially, these gradients are computed only with respect to the head parameters—namely $(W_{L2}, W_{out})$​—while the L1 accumulator weights remain excluded from this differentiation.

<span style="color: #808080;">[Forward then Grad]</span> The workflow begins with a forward pass through the model for a batch of positions, producing WDL predictions that are then compared against teacher labels via cross-entropy loss. Calling `torch.autograd.grad(loss, head_parameters, retain_graph=False)` yields the desired gradients, which are subsequently detached from the computation graph and flattened into a single vector representation for each position. This entire computation runs fully parallelised on the GPU and completes in a single pass over the dataset, with the resulting gradient tensors persisted to disk for use in the downstream clustering stage.

### 3.4.3 Normalisation

<span style="color: #808080;">[Normalisation]</span> The raw gradients can have highly variable magnitudes depending on the position, the current state of the model, and on weather the multiplicity of the state is considered or not. The *L2 normalization* of each gradient vector has been shown to work well in gradient-clustering literature (ELREA, GradientSpace) and preserves the relative angular structure of the gradients.

$$\Delta_i^{\text{norm}} = \frac{\Delta_i}{\|\Delta_i\| + \epsilon}$$

<span style="color: #808080;">[Direction, not Magnitude]</span> This projects each gradient onto the unit hypersphere, preserving direction while removing magnitude information. This is appropriate because the *direction* of the gradient encodes the type of specialisation needed, while the magnitude is more sensitive to the current loss value and position difficulty.

#todo distinction between L2 norm and L2 layer?

### 3.4.4 Storage and Compute Considerations

<span style="color: #808080;">[Storage Cost]</span> Computing and storing sample gradients for [dataset_size] positions presents practical challenges. Each gradient vector has dimension $P_{\text{head}} \approx [P_head]$ (flattened L2 and output weights). Storing this as 32-bit floats would require approximately:

$$[dataset_{size}] \times [P_{head}] \times 4 \text{ bytes}$$

#todo considerations on whether to introduce rounding, and whether to use just part of the dataset

### 3.4.5 Handling Duplicate Positions

<span style="color: #808080;">[The problem of frequent positions]</span> In a dataset of chess positions extracted from human games, the same position can occur many times. Opening positions, in particular, are heavily overrepresented: the starting position appears in every game, and the first few moves are shared across a large fraction of the dataset. This multiplicity has two distinct effects on the pipeline, one on training and one on clustering, and they call for different treatments.

<span style="color: #808080;">[Effects on training]</span> During training, the frequency of a position determines its contribution to the loss. If every occurrence is treated equally, the natural distribution of the data is preserved: common positions contribute more, rare positions contribute less. This is statistically appropriate if the goal is to match the distribution of positions encountered during play. However, extreme overrepresentation is wasteful, both computationally and statistically: the starting position carries no more information on its millionth occurrence than on its first, and it can dominate the gradient signal to the detriment of more informative positions. A common remedy is to cap the multiplicity of any position, so that no single state contributes more than a fixed number of times per epoch.

#todo capping reduces the represented density in that spot

<span style="color: #808080;">[Effects on clustering]</span> For clustering, the issue is different. The clustering operates on sample gradients, and duplicated positions produce identical gradient vectors. If duplicates are retained, they occupy a disproportionate volume in the gradient space, and cluster centroids are pulled toward common positions. This biases the partition toward regions of the state space that are frequent in the data, rather than regions that require distinct learning signals. Deduplicating before clustering avoids this bias, but it discards the information that some positions are more important than others.

#todo decide what to do

<span style="color: #808080;">[How we deal with it]</span> In this work we adopt a compromise. The dataset is constructed one slice at the time. Within each slice, the positions are counted, and the visit count is stored alongside each position...

#todo finish
 
---

## 3.5 Step 3: Cluster Sample Gradients

<span style="color: #808080;">[Clustering]</span> With the normalised sample gradients $\{\Delta_i^{\text{norm}}\}_{i=1}^N$ computed for every position in the dataset, the third step is to partition the data into $B$ clusters (buckets) such that positions with similar learning signals are grouped together. This partition will define the assignment of positions to expert heads during fine-tuning.

### 3.5.1 The Objective of Clustering

<span style="color: #808080;">[Objective]</span> Recall the central hypothesis of this work: clustering positions by their sample gradients $\Delta_i$ yields a partition $\mathcal{P} = \{\mathcal{D}_1, \dots, \mathcal{D}_B\}$ for which the resulting task vectors $\delta_i = \theta_i - \theta_{\text{base}}$ are maximally diverse. The role of the clustering algorithm is to discover a partition that approximates this objective. From a practical standpoint, we seek an efficient clustering algorithm that produces clusters that are cohesive, balanced, stable, and, hopefully, as separated as possible. Different clustering algorithms make different trade-offs with respect to these criteria. We consider two broad families: fixed-$B$ algorithms and density-based algorithms.

### 3.5.2 Fixed-$B$ Algorithms: K-Means and Variants

<span style="color: #808080;">[K-Means]</span> The most widely used clustering algorithm is **K-Means**, which partitions the data into $B$ clusters by minimising the within-cluster sum of squares. For a set of clusters $\{\mathcal{C}_1, \dots, \mathcal{C}_B\}$, K-Means minimises:

$$\sum_{k=1}^B \sum_{i \in \mathcal{C}_k} \|\Delta_i - \mu_k\|^2$$

where $\mu_k = \frac{1}{|\mathcal{C}_k|} \sum_{i \in \mathcal{C}_k} \Delta_i$ is the centroid of cluster $k$. This algorithm is fast and well-understood. Specifically, *Mini-Batch K-Means* is particularly attractive for our scale, as it efficiently handles large datasets by processing data in mini-batches, making it suitable for our dataset, that comprises of millions of chess positions. It reduces memory requirements and converges faster than standard K-Means, while producing nearly identical results.

#todo Disadvantages:
- **Sensitivity to initialisation**: While k-means++ mitigates this, K-Means can still converge to local optima.
- **Assumption of spherical clusters**: K-Means performs best when clusters are roughly spherical and of similar size. Our gradient space may not satisfy this assumption.
- **Fixed $B$ is a hyperparameter**: The choice of $B$ is critical and must be determined separately (see Section 3.5.4).
- **Outlier sensitivity**: K-Means can be influenced by outliers, which may distort centroids.

### 3.5.3 Density-Based Algorithms

<span style="color: #808080;">[Density-Based]</span> An alternative family of algorithms, **density-based clustering**, does not require a fixed number of clusters. Instead, these methods identify clusters as regions of high density separated by regions of low density.

<span style="color: #808080;">[Density Peak]</span> **Density Peak Clustering** (Rodriguez & Laio, 2014) is a particularly relevant algorithm for our setting. It works by identifying cluster centres as points that have high local density (many neighbours within a cutoff distance) and are far from points with higher density (suggesting they are local maxima).

<span style="color: #808080;">[Automatic B]</span> The number of clusters emerges naturally from the data: each point with density higher than all its neighbours and with large distance to the nearest higher-density point is a cluster centre. The remaining points are assigned to the same cluster as their nearest higher-density neighbour.

<span style="color: #808080;">[Why This]</span> We chose to use this algorithm because it automatically determines $B$, so it will be interesting to find out what the clusters of board positions actually represent. This algorithm is also notoriously robustness to outliers; points with low density and large distance to higher-density points are naturally identified as outliers.

#todo maybe we choose a different one (must test)

#todo Disadvantages:
- **Parameter sensitivity**: The algorithm requires choosing a distance cutoff (for density estimation) and a threshold for the distance to higher-density points. These parameters can significantly affect the number of clusters.
- **Computational cost**: Computing pairwise distances for [dataset_size] points is prohibitive (O(N²)). We would need to use approximations (e.g., approximate nearest neighbours) or subsample the data.
- **Cluster size imbalance**: Density-based methods may produce clusters of very different sizes, which can be problematic for expert fine-tuning (some experts would have too little data).
- **Unstable number of clusters**: The number of clusters can vary with the parameters or with slight perturbations in the data, complicating the design of a fixed-architecture MoE.

<span style="color: #808080;">[DBSCAN]</span> DBSCAN (Density-Based Spatial Clustering of Applications with Noise) defines clusters as regions of high density separated by regions of low density, using two parameters: $\varepsilon$ (neighbourhood radius) and `min_samples` (minimum points to form a dense region). It does not require the number of clusters $B$ as an input and can label outliers as noise. It avoids the need to pre-specify $B$, which could be useful when the natural structure of the gradient space is unknown. It can detect clusters of arbitrary shapes and is robust to outliers, potentially identifying positions that yield uninformative gradients.

#todo **Why no.** DBSCAN performs poorly on high-dimensional data due to the curse of dimensionality, where density estimates become unreliable. Its time complexity scales poorly with dataset size, making it infeasible for our 5 million positions in 17,000 dimensions. It is also sensitive to its hyperparameters and struggles with clusters of varying densities. We therefore adopt Mini-Batch K-Means as our default clustering algorithm.

### 3.5.4 Choosing $B$ and the Algorithm

<span style="color: #808080;">[Choosing B]</span> The choice between fixed-$B$ and density-based clustering depends on the relative importance of architectural efficiency and data-driven discovery. A density-based approach that automatically determines $B$ could reveal the natural structure in the gradient space, potentially identifying a number of clusters that better reflects the underlying distribution of learning signals. On the other hand, hardware constraints might impose a hard limit on how many expert heads our device is allowed to store based on our architecture of choice.

#todo which approach we end on preferring based on some results

### 3.5.5 Validation and Diagnostics

<span style="color: #808080;">[Purpose of Validation]</span> Once the clustering is complete, the quality of the resulting partition is assessed using several complementary metrics. The purpose of this step is not to select a single best partition, but to understand the structure of the gradient space and to guide the choice of the number of experts $B$ used in the subsequent fine-tuning stage.

<span style="color: #808080;">[Cluster Size Distribution]</span> The **cluster size distribution** provides a first diagnostic. We examine the number of positions assigned to each bucket to ensure that no cluster is too small to support stable fine-tuning. A bucket with very few positions may lead to an expert head that is poorly conditioned or that overfits its small training set. Conversely, a highly imbalanced distribution may indicate that the clustering algorithm has collapsed most of the data into a single region, which would defeat the purpose of partitioning. A perfectly balanced distribution is not required, but the sizes should be large enough to train each expert reliably.

<span style="color: #808080;">[Inertia]</span> **Inertia**, the within-cluster sum of squared distances to the centroid, measures the compactness of the clusters. For a fixed value of $B$, lower inertia indicates tighter clusters. We compute inertia across a range of $B$ values and look for an elbow point, the value beyond which additional clusters yield diminishing reductions in inertia. This heuristic is not definitive, but it provides a useful indication of the natural structure of the gradient space and helps constrain the range of $B$ values worth exploring.

#todo maybe not necessary?

<span style="color: #808080;">[Silhouette Score]</span> The **silhouette score** (*Rousseeuw*, 1987) measures how similar each point is to its own cluster compared to the nearest neighbouring cluster. For a point $i$, the coefficient is defined as

$$s_i=b_i−a_i \max⁡(a_i,b_i)$$​

where $a_i$ is the mean distance to the other points in the same cluster and $b_i$​ is the mean distance to the points in the nearest other cluster. Values range from $−1$ to $+1$, with higher values indicating better-defined clusters. A score near zero suggests overlapping clusters, while negative values indicate that some points may be assigned to the wrong cluster. We report the average silhouette score over all positions, but interpret it with caution: the gradient space is high-dimensional, and silhouette scores tend to degrade as dimensionality increases even when the data exhibits meaningful structure.

<span style="color: #808080;">[Centroid Separation]</span> Finally, the **cosine distance between cluster centroids** provides a direct measure of the diversity that is central to the method. The objective of the partition is to obtain task vectors $δ_i=θ_i−θ_{base}$​ that are maximally diverse, and the pairwise cosine distance between centroids serves as a proxy for this diversity. We compute the cosine distance between the centroid of clusters in the normalised gradient space. High average distances indicate that the clusters are well separated and likely to induce distinct task vectors. Low distances suggest that the clusters are redundant and may not yield meaningful specialisation. We report both the average and the minimum pairwise distance, since the presence of even one nearly collinear pair of centroids may indicate that two buckets could be merged without loss of diversity.

<span style="color: #808080;">[Selecting B]</span> Together, these diagnostics provide a comprehensive view of the partition and inform the choice of $B$. In practice, the clustering pipeline is run for a range of values, such as $B∈\{2,4,8,16,32\}$, and the partition that offers the best balance between compactness, separation, and cluster size is selected. The chosen partition is then used for dispatcher training and expert fine-tuning in the subsequent steps.

---

## 3.6 Step 4: Train the Dispatcher

<span style="color: #808080;">[Why the dispatcher]</span> The clustering step yields a partition of the training dataset into $B$ buckets, but this partition is defined in terms of sample gradients. At inference time, gradients are unavailable; computing them would require the teacher evaluation of the position, which we of corse don't have. To route a new position to the appropriate expert, we therefore need a mechanism that predicts the bucket assignment from features that are already computed during the normal forward pass. The dispatcher $g_\phi$ serves this purpose.

### 3.6.1 Architecture

<span style="color: #808080;">[Architecture]</span> The dispatcher is a linear classifier that takes as input the concatenated L1 activations $h = [h_{\text{own}} \| h_{\text{opp}}] \in \mathbb{R}^{2W}$ and produces a vector of logits over the $B$ buckets:

$$z = W_{\text{disp}} \, h + b_{\text{disp}}$$

where $W_{\text{disp}} \in \mathbb{R}^{2W \times B}$ and $b_{\text{disp}} \in \mathbb{R}^B$. During training, a softmax function converts these logits into a probability distribution over buckets, and the model is trained to predict the cluster assignments produced by the clustering step. At inference, the softmax is discarded and the predicted bucket is simply the argmax of the logits:

$$g_\phi(s) = \arg\max_i z_i$$

This design ensures that routing adds only a matrix-vector multiplication and a comparison, both of which are inexpensive integer operations.

#todo maybe $h_{\text{own}}$ is enough for the dispatcher

### 3.6.2 Why a Linear Model

<span style="color: #808080;">[Efficiency of the linear model]</span> The choice of a linear dispatcher is deliberate and follows directly from the inference-time constraints discussed in Section 2.1.5. A linear layer with $2W$ inputs and $B$ outputs requires $2W \times B + B$ parameters, which for $W=128$ and $B=8$ amounts to approximately 2,056 parameters. This is small enough to fit comfortably in the flash memory of a microcontroller alongside the expert heads. The computation itself is a single matrix-vector product, which can be implemented with integer arithmetic and adds negligible latency to the evaluation function.

<span style="color: #808080;">[The linear model is enough]</span> A more expressive dispatcher, such as a multi-layer perceptron with hidden layers, could potentially achieve higher classification accuracy. However, the additional parameters and non-linearities would increase both memory footprint and inference cost. Given the extreme efficiency requirements of the target hardware, we prioritize simplicity and speed over marginal gains in routing accuracy. As we discuss in Section 3.6.4, the linear dispatcher is sufficient to capture the coarse structure of the partition.

#todo can a complex dispatcher compromise the expert selection process at inference time?
#todo check reference to 3.6.4

### 3.6.3 Training

<span style="color: #808080;">[Training the dispatcher]</span> The dispatcher is trained on the same dataset used for clustering, with the cluster labels serving as targets. The input features are the L1 activations $h_i = W_{L1} \cdot x_i$, which are already computed during the base model's forward pass and can be cached for the entire dataset. The loss is the standard cross-entropy between the predicted distribution and the one-hot cluster assignment:

$$\mathcal{L}_{\text{disp}} = -\frac{1}{N} \sum_{i=1}^N \log (g_\phi(s_i)_{c_i})$$

where $c_i$ is the cluster index assigned to position $s_i$. We optimize this loss using Adam with a learning rate of $10^{-2}$, decayed to $10^{-3}$ over the course of training. Training typically converges within a few epochs, and we use early stopping on a held-out validation set to prevent overfitting. Because the dispatcher is a small linear model, the entire training procedure takes only a few minutes on a GPU.

#todo update hard numbers
#todo remove GPU remark?

### 3.6.4 Can a Linear Dispatcher Capture the Clustering?

<span style="color: #808080;">[Expressiveness of the dispatcher]</span> The sample-gradient space is highly dimensional ($P_{\text{head}} \approx 17,000$), while the dispatcher operates on the L1 activations, which have dimension $2W = 256$. A natural question is whether a linear function of these activations can accurately predict the cluster assignments derived from gradients. The answer depends on how much information about the learning signal is already encoded in the L1 representation.

#todo nuance: we don't really need to capture the clustering perfectly
#todo In practice, we observe that a linear dispatcher achieves classification accuracy in the range of ...

### 3.6.5 Fixed vs. Re-Assigned Cluster Assignments

<span style="color: #808080;">[Why the decision matters]</span> A subtle but important design decision concerns the relationship between the dispatcher and the expert fine-tuning. During the fine-tuning step (Section 3.7), each position is assigned to a bucket. We can either use the original cluster assignments produced by the clustering algorithm, or we can re-assign positions using the dispatcher's predictions. These two options have different implications.

<span style="color: #808080;">[Reassigning positions]</span> If we use the dispatcher to re-assign positions, the experts are trained on the buckets that the dispatcher believes are correct, rather than the buckets discovered by clustering. This could potentially align the experts more closely with the dispatcher's behavior at inference time. The downside is that some of the structure and complexity of the original clustering is lost, in favor of a simpler, linear partitioning.

#todo did not yet decide witch one is better, maybe try both?

---

## 3.7 Step 5: Fine-Tune Expert Heads

<span style="color: #808080;">[Context for MoE]</span> The final step of the pipeline produces the specialised experts. Once the clustering has defined the partition and the dispatcher has been trained to approximate it, we fine-tune a separate head on each bucket, starting from the base model and keeping the L1 representation frozen. The result is a set of $B$ expert heads, each adapted to a distinct region of the state space, sharing a common L1 accumulator.

### 3.7.1 What Is Trained and What Is Frozen

<span style="color: #808080;">[What is frozen]</span> The L1 accumulator remains frozen throughout this step, as it has been since the base model was trained. This is not merely a convenience but a structural requirement: the dispatcher relies on L1 activations to route positions, and those activations must be computed by the same weight matrix regardless of which expert is selected. If each expert were to fine-tune its own L1 weights, the dispatcher would need to account for $B$ different representation spaces, and the inference pipeline would become unworkable. Freezing L1 also keeps the memory footprint constant: all experts share the same accumulator, and only the head parameters differ between them.

<span style="color: #808080;">[What is trained]</span> Each expert head consists of the L2 layer and the output layer, parameterised by $(W_{L2}^{(i)},b_{L2}^{(i)},W_{\text{out}}^{(i)},b_{\text{out}}^{(i)})$. These are initialised from the corresponding parameters of the base model, so that every expert starts from the same well-trained foundation. The only thing that distinguishes one expert from another is the data on which it is fine-tuned—the subset of positions assigned to its bucket by the clustering step.

#todo omit bias?
#todo will we try also double-hidden sparse head?

### 3.7.2 Training Procedure

<span style="color: #808080;">[What is trained]</span> For each bucket $\mathcal{D}_i$, we fine-tune a copy of the base head on the positions in that bucket, using the same soft cross-entropy loss and optimiser as in the base training. The L1 activations are pre-computed once for the entire dataset and cached, so the fine-tuning step only requires forward and backward passes through the small head, not the full model. This makes the procedure extremely fast: fine-tuning $B$ heads on subsets of a dataset of millions of positions takes only a fraction of the time required to train the base model.

<span style="color: #808080;">[Training specs]</span> We use a short training schedule, typically one or two sweeps over the bucket, with early stopping based on the cross-entropy on a held-out portion of the bucket. Depending on the dataset size, one of two approaches is best. If the buckets are too small ( #todo reference overfitting plot), training for too long risks overfitting. In our specific case data is abundant, and the goal is to reach the best possible performance on each bucket in isolation, to produce a set of experts whose combined behaviour significantly improves upon the single base head. The learning rate is set lower than in base training, to avoid large deviations from the base head that could destabilise the shared L1 representation.

#todo rephrase?
#todo decide numbers like `lr`

### 3.7.3 Data Availability per Expert

<span style="color: #808080;">[The problem of partitioning]</span> Partitioning the dataset into $B$ buckets means that each expert sees only a fraction of the total data. If the partition is balanced, each expert is fine-tuned on approximately $N/B$ positions. For $B=8$ and $N=5$ million, this amounts to roughly $625,000$ positions per expert, which is still a substantial training set. However, if the clustering produces imbalanced buckets, some experts may be trained on far fewer positions, which can lead to underfitting or unstable training.

#todo replace hard numbers

<span style="color: #808080;">[How to deal with small buckets]</span> This is one of the reasons why we monitor the cluster size distribution as part of the validation diagnostics in Section 3.5.5. If a bucket is too small to support stable fine-tuning, several remedies are possible: ...

#todo single sweep -> gradient -> optimize for magnitude
#todo train also on nearby clusters?

<span style="color: #808080;">[What we do]</span> We do not adopt this relaxation in the present work, but we note it as a natural extension that could improve the robustness of the method when the partition is uneven.

#todo add reference to the plot that shows that we have plenty data

### 3.7.4 The Resulting MoE Model

<span style="color: #808080;">[The resulting model]</span> At the end of this step, we have a complete mixture-of-experts evaluation function. The model consists of three components: the frozen L1 accumulator, which is shared across all experts; the dispatcher, which maps L1 activations to a bucket index; and the $B$ expert heads, each containing its own L2 and output parameters. During inference, a position is encoded into its sparse feature representation, passed through L1 to obtain the accumulator vector, routed by the dispatcher to a single bucket, and finally evaluated by the corresponding expert head. The output is a WDL distribution from the side-to-move perspective, from which the scalar evaluation is derived as usual.

<span style="color: #808080;">[Comparing to single model]</span> The inference cost is therefore one L1 forward pass (which is incremental during search), one linear dispatcher operation, and one head forward pass. Compared to the single-head base model, the only additional cost is the dispatcher, which as we have seen adds a matrix-vector multiplication of negligible size. The experts themselves are not more expensive than the base head: they have the same architecture, and only one is evaluated per position. The memory cost is $B$ times the size of the head parameters, which remains small relative to the L1 accumulator. This is the essential trade-off of the method: we gain specialisation at the cost of additional head parameters, while keeping the inference path as lean as the base model.

---

## 3.8 Generalisation Beyond Chess

<span style="color: #808080;">[The method is not game-specific]</span> The method presented in this chapter is not specific to chess. Its core ingredients are a state space, a parametric model that maps states to predictions, a differentiable loss, and a dataset of *state-target* pairs. Given these, the pipeline can be applied to a different domain without modification: train a base model, compute per-sample gradients with respect to the parameters of a head, cluster them, train a dispatcher on a frozen intermediate representation, and fine-tune specialised heads on the resulting buckets. Chess is an attractive testbed because it offers a large dataset of labelled positions, a well-established efficient architecture in NNUE, and clear resource constraints that make the efficiency of the dispatcher meaningful. But the underlying principle, that a partition of the state space can be discovered by clustering the learning signal rather than the input or the representation, is domain-agnostic.

<span style="color: #808080;">[Shogi, Go, videogames, and robotics]</span> Several other settings share the structural properties that make the method applicable. In similar board games such as shogi, the same formulation carries over directly: shogi engines already use NNUE-style accumulators, and the position encoding and WDL labels can be adapted with minimal changes. In the ancient game of Go, the NNUE architecture should be swapped for a model of convolutional nature. In video game AI and simulated environments, where agents must evaluate states or select actions under tight latency budgets, a similar decomposition into a shared representation and a small routed head could reduce inference cost while preserving specialisation. In robotics, where control policies often operate on high-frequency sensor streams and must run on embedded hardware, the same pattern of a frozen feature extractor followed by a lightweight, routed head is a natural fit: the feature extractor runs continuously, while the head is selected by a dispatcher that observes the same features at negligible cost.

#note we mention videogames and robotics

<span style="color: #808080;">[World Models]</span> The method also connects to recent work on world models, where a learned representation of the environment is used to predict future states or plan actions. World models typically maintain a latent state that is updated incrementally as the environment evolves, much like the NNUE accumulator is updated as pieces move. This parallel suggests that the gradient-based bucketing idea could be extended to world models, where different regions of the latent state space may require different dynamics or reward predictors. In such a setting, the "head" would be the component that predicts the next latent state or the reward, and the dispatcher would route to a specialised predictor based on the current latent representation. The efficiency argument carries over: a dispatcher that operates on the shared latent state adds minimal cost compared to evaluating multiple full predictors.

#note could be an important connection, maybe elaborate in an other chapter

<span style="color: #808080;">[What is domain-specific]</span> What must change across domains is not the algorithmic structure but the concrete choices within it. The feature encoding, the architecture of the base model, the loss function, and the definition of the target all depend on the domain. In chess, we use a sparse binary encoding and a WDL target from a strong teacher; in another domain, the encoding might be dense and continuous, the loss might be a regression or a contrastive objective, and the target might come from human demonstrations, simulation, or self-play. The dispatcher's input features would likewise be domain-specific: in chess they are the L1 activations, but in another model they could be any intermediate representation that is cheap to compute and informative about the learning signal. These choices affect performance but do not alter the underlying method. The central claim, that per-sample gradients provide a useful signal for partitioning a state space in a way that supports efficient mixture-of-experts inference, is independent of the domain in which it is tested.

#todo rephrase "domain-specific"

---

> **Note for AI**: *The parts marked with a #todo are yet to be completed. The tagged comments are NOT to be exported to the LaTeX document, and are not meant to be implemented while exporting this document to LaTeX. The gray labels are for clarity only and must not be transferred to the Latex. Placeholders in square brackets (e.g. `[dataset_size]`, `[W]`) must be replaced with the current values from `_ai-info_.md` when converting this document to LaTeX.*

[[4 - Implementation]]
