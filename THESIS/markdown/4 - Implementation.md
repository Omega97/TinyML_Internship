# Implementation: NNUE and MoE for Chess


#idea can/should we focus the dataset on positions hard for the NNUE (eval flipped by 180°)? Is there a trick to let the NNUE figure out the steps ahead, like the solution to a tactic? **More layers but sparse connections**?

#idea **certi algoritmi di clustering decidono $B$** 

#idea U Map in 3D, DBScan, Density Peak Clustering
  
#idea PCA per aiutare il clustering?

#todo improve the flavour of the thesis by making each section more *intentional* and clear - align the chapter to its title (in gray) 

#todo we could emphasize the hardest to read positions, but with quiescent search we hardly ever have to evaluate those correctly
  
#todo visualizzare i cluster; plot per la tesi

#todo to specify why we use cross-entropy loss

#todo estimate number of value function calls

---

## 4.1 Dataset and Teacher

<span style="color: #808080;">[Data Overview]</span> The quality of an NNUE evaluation function depends critically on the dataset used for training. For this work, we constructed a dataset of approximately [dataset_size] chess positions extracted from human games and labeled with high-quality value estimates from a strong teacher network, **Leela Chess Zero** (Lc0), the *spiritual* successor of *AlphaZero*.

#todo update hyperparameters

### 4.1.1 Dataset Construction

<span style="color: #808080;">[Sources]</span> The raw positions are sourced from **Lichess** monthly PGN archives, containing standard-rated games played by humans across all time controls. We filter games to include only those with at least 16 moves, excluding very short games that often end in early blunders or resignations and would introduce noisy or uninformative positions into the training set ( #todo omit?). From each remaining game, we sample positions randomly with probability [keep_prob], ensuring a diverse and representative collection of states across all phases of play, and mostly avoiding highly correlated positions.

<span style="color: #808080;">[Phase Mix]</span> The dataset retains the natural distribution of game phases found in human play: a majority of middlegame positions, with fewer openings and endgames. We intentionally avoid resampling to balance phases, as the natural distribution better reflects the positions the engine will encounter during actual play. Each position is stored as a FEN string along with the multiplicity of the position—a visit count that may be used for optional weighting during training.

<span style="color: #808080;">[Tactical Extra]</span> A small fraction of the dataset is also a collection of interesting tactical positions, and high level bot games. They capture a portion of the state space that may be outside of regular human play.

#todo cosa fare con posizioni duplicate (3.4 e 4.1.1) *La loss è dominata dalle posizioni comuni. Il gradiente totale di una posizione comune è proporzionale alla sua frequenza. È statisticamente corretto se vogliamo che il modello sia calibrato sulla distribuzione reale del gioco, ma è inefficiente e rischia di sovrarappresentare posizioni banali come quella iniziale.*

### 4.1.2 Feature Encoding

<span style="color: #808080;">[Encoding]</span> For the NNUE model, each FEN string is encoded as a sparse binary feature vector of length $d_{\text{in}}=$[d_{in}]. This encoding is designed to capture both the positional and tactical structure of the board in a form suitable for the accumulator layer.

<span style="color: #808080;">[Feature Split]</span> The $d_{\text{in}}$ features are divided into two categories:

- **716 base features**: These encode piece-square pairs, representing the presence of each piece type on each square. The feature set is pruned to remove impossible pawn ranks and compressed to reduce redundancy (e.g., the king plane is stored in a compact form). #todo explain better? In the [d_in]‑dim SARDINE encoder, the king plane is compressed from 64 squares to 32, saving features.

- **128 tactical features**: These encode dynamic aspects of the position, specifically which pieces are under attack and which pieces are attacking the king. These features provide the network with explicit information about immediate tactical threats. #todo is it worth trying without these? I don't think so...

#todo list to prose?

<span style="color: #808080;">[Dual POV]</span> A key design choice is the **dual‑POV** encoding: for each position, the encoder produces two sets of sparse indices—one from the perspective of the **side‑to‑move** (STM) and one from the opponent's perspective (obtained by flipping the board rank-wise and swapping colors). This dual representation allows the network to learn symmetric evaluations and is consistent with the NNUE architecture's ability to evaluate positions from either player's viewpoint.

<span style="color: #808080;">[Storage]</span> The encoded features are pre‑computed and stored in `.npz` slices for efficient loading during training.

### 4.1.3 Data Splits

<span style="color: #808080;">[Splits]</span> The dataset is partitioned into training and test splits. The training set consists of approximately [dataset_size] positions, distributed across 165 slices for balanced I/O and stochastic sampling. The test set comprises a random portion of [test_fraction] of the position that are held out of the training set.
#todo remember to update numbers when they change...

### 4.1.4 Teacher Model: Lc0

<span style="color: #808080;">[Teacher]</span> To provide accurate target labels, we use **Lc0** (Leela Chess Zero) as the teacher model. Lc0 is a *convolutional neural network* trained via self‑play reinforcement learning, following the AlphaZero paradigm. Its value head outputs a probability distribution over the three possible game outcomes—Win, Draw, Loss—from the perspective of the side‑to‑move:

$$p_{\text{WDL}}(s) = \text{softmax}(\text{logits}(s)) = (p_W, p_D, p_L)$$

From this distribution, we compute the scalar expected reward:

$$v(s) = p_W - p_L \in [-1, +1]$$

which represents the expected outcome of the game from the current position. This scalar is the training target for the NNUE value head.

<span style="color: #808080;">[Why Lc0]</span> Lc0 is chosen as the teacher for several reasons. First, it natively outputs WDL probabilities, which align directly with the NNUE's output head. Second, its strength—rated well above 3500 Elo—makes it a highly reliable source of positional evaluations.  Finally, Lc0 is open‑source and provides pre‑trained networks, making the labelling pipeline reproducible.

<span style="color: #808080;">[Labelling Setup]</span> For this work, we label positions using Lc0's latest best network (e.g., `791556.pb.gz` from the Lc0 training server). We run Lc0 in UCI mode with `--show-wdl` enabled and evaluate each position with a single MCTS search. While depth‑1 evaluations may occasionally miss short‑term tactics, the resulting label noise is acceptable given the target Elo range of the engine (approximately 1700). For a cleaner but more expensive relabelling, one could increase the search depth.
#todo part of the dataset is already at depth 2...

### 4.1.5 Labelling Pipeline

<span style="color: #808080;">[Pipeline]</span> The complete labelling pipeline is straightforward. We parse Lichess PGNs and sample positions uniformly at random from each game, saving FEN strings and visit counts. This reduces the correlation between the positions in the final dataset. For each unique FEN, we invoke Lc0 in *UCI mode* at depth 1 to obtain WDL probabilities from the STM perspective. We then procede to save the WDL probabilities alongside the FEN and visit counts in JSON format. In the encoding step we pre‑compute the [d_in]‑dimensional sparse feature vectors (both STM and opponent POVs) and store them in `.npz` slices for efficient training. The final result is a dataset of pairs of sparse input board positions and their relative WDL probabilities.

#todo UCI mode? STM perspective?
#todo specify WDL and depth 1?
#todo ensure the prose around it is not just a list of bullet points in disguise.

---

## 4.2 Base NNUE Architecture

<span style="color: #808080;">[Architecture]</span> The base NNUE (Efficiently Updatable Neural Network) model serves as the foundation upon which the Mixture of Experts extension is built. Its architecture is designed to balance representational capacity with the stringent memory and computational constraints of the target hardware. The model follows the dual‑perspective paradigm introduced by the original NNUE design, but incorporates modifications tailored to the specific feature set and bucketing objectives of this work.

### 4.2.1 Model Overview

<span style="color: #808080;">[Overview]</span> The base model $f_\theta$ is a feed‑forward neural network with three parameterised layers: a shared sparse first layer (L1), a dense second layer (L2), and a linear output head, with *softmax* activations. The input to the model is the sparse binary feature representation described in Section 4.1.2, consisting of [d_in] active features per perspective. The model processes both the side‑to‑move (STM) and opponent perspectives through the same L1 layer, producing two accumulator vectors that are later concatenated and passed through the remaining layers.

<span style="color: #808080;">[Forward Pass]</span> Formally, the model computes:

$$
h_{\text{own}} = W_{\text{L1}} \, x_{\text{own}}, \qquad
h_{\text{opp}} = W_{\text{L1}} \, x_{\text{opp}},
$$

where $x_{\text{own}}, x_{\text{opp}} \in \{0,1\}^{[d_in]}$ are the sparse feature vectors for the two perspectives, and $W_{\text{L1}} \in \mathbb{R}^{d_{\text{in}} \times W}$ is the shared weight matrix of the accumulator layer. The output of the L1 layer is a pair of vectors $h_{\text{own}}, h_{\text{opp}} \in \mathbb{R}^W$, where $W$ is the hidden dimension of the accumulator, set to $W = [W]$ in this work.

<div align="center">
    <img src="THESIS/thesis-plots/sardine_nnue_architecture.png" width="600">
</div>

### 4.2.2 Shared L1 Accumulator

<span style="color: #808080;">[Accumulator]</span> The L1 layer, often referred to as the accumulator, is the defining component of the NNUE architecture. Its weight matrix $W_{\text{L1}}$ is shared between the two perspectives, enabling the network to learn a common representation of board structure while retaining perspective‑specific information through the different input features. The sparsity of the input features allows the accumulator to be updated efficiently: rather than recomputing the entire matrix‑vector product for each new position, the network maintains the accumulator vector incrementally, adding or subtracting the contributions of features that change as pieces move.

<span style="color: #808080;">[CReLU]</span> The L1 activations are passed through a **CReLU** (Clipped Rectified Linear Unit) activation function, which maps each activation to the range $[0, 127]$:

$$
a_{\text{own}} = \text{clamp}(h_{\text{own}}, 0, 127), \qquad
a_{\text{opp}} = \text{clamp}(h_{\text{opp}}, 0, 127).
$$

This clipping is essential for integer quantization, as it bounds the dynamic range of the accumulator values and allows the use of low‑precision integer arithmetic during inference.

### 4.2.3 Side‑to‑Move Reorder

<span style="color: #808080;">[STM Reorder]</span> Before concatenating the two accumulator vectors, we apply a 
*side‑to‑move (STM) reorder* to ensure that the expert head always receives the perspective of the current player first. The reordering is governed by the binary flag $\text{stm\_white} \in \{0,1\}$, which indicates whether White is to move:

$$
h_{\text{first}} =
\begin{cases}
a_{\text{own}}, & \text{if } \text{stm\_white} = 1, \\
a_{\text{opp}}, & \text{otherwise},
\end{cases}
\qquad
h_{\text{second}} =
\begin{cases}
a_{\text{opp}}, & \text{if } \text{stm\_white} = 1, \\
a_{\text{own}}, & \text{otherwise}.
\end{cases}
$$

<span style="color: #808080;">[Concatenation]</span> The two vectors are then concatenated to form the input to the L2 layer:

$$
h = [h_{\text{first}} \, \| \, h_{\text{second}}] \in \mathbb{R}^{2W}.
$$

<span style="color: #808080;">[Perspective Invariance]</span> This reordering step is critical for making the evaluation perspective‑invariant: the network always receives the board from the point of view of the side to move, and the output $v \in [-1, +1]$ is always interpreted as the expected reward for the current player, regardless of colour.

### 4.2.4 L2 Layer and Output Head

<span style="color: #808080;">[L2 Layer]</span> The concatenated vector $h$ is passed through a dense L2 layer with hidden dimension $H = [H]$:

$$
z = \text{ReLU}(W_{\text{L2}} \, h + b_{\text{L2}}),
$$

where $W_{\text{L2}} \in \mathbb{R}^{2W \times H}$ and $b_{\text{L2}} \in \mathbb{R}^H$ are the weight matrix and bias of the L2 layer. The ReLU activation introduces non‑linearity and has been shown to work well with the sparse accumulator features.

<span style="color: #808080;">[Output Head]</span> Finally, the L2 activations are projected to a three‑dimensional output representing the logits for Win, Draw, and Loss probabilities:

$$
\text{logits} = W_{\text{out}} \, z + b_{\text{out}},
$$

where $W_{\text{out}} \in \mathbb{R}^{H \times 3}$ and $b_{\text{out}} \in \mathbb{R}^3$. During training, these logits are converted to probabilities via the softmax function:

$$
p_{\text{WDL}}(s) = \text{softmax}(\text{logits}(s)) = (p_W, p_D, p_L).
$$

<span style="color: #808080;">[Scalar Eval]</span> The scalar evaluation used for search is obtained as $v = p_W - p_L$, the expected reward from the side‑to‑move perspective.

### 4.2.5 Training Objective

<span style="color: #808080;">[Soft-CE]</span> The model is trained to minimise the **soft cross-entropy** between its predicted WDL distribution and the teacher labels provided by the Lc0 value function. Let $\mathcal{O} = \{W, D, L\}$ denote the set of possible game outcomes. For a batch of $N$ positions with teacher probabilities $\hat{p}_i = (\hat{P}_i(o))_{o \in \mathcal{O}}$ and model outputs $p_i = (P_i(o))_{o \in \mathcal{O}}$, the loss is:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{o \in \mathcal{O}} \hat{P}_i(o) \log P_i(o)
$$

The outer sum averages the loss over the batch, while the inner sum accumulates the contribution of each outcome for a given position.

<span style="color: #808080;">[Why this loss]</span> This loss is well-suited to the task because the teacher labels are probability distributions rather than point estimates. Soft cross-entropy encourages the model to match the full outcome distribution, not merely its scalar expected reward, and therefore provides a richer training signal. The loss also handles the non-linearity introduced by the softmax in a principled way, and its gradient does not saturate at extreme values, unlike the squared error on the scalar evaluation $v = P(W) - P(L)$.

### 4.2.6 Parameter Count and Model Size

<span style="color: #808080;">[Model Size]</span> With $W = [W]$ and $H = [H]$, the total number of trainable parameters is approximately [n_params]. This compact size is deliberately chosen to fit within the memory constraints of the target hardware: the L1 weights ($[d_in] \times [W]$ `int8` values) dominate the parameter count, while the L2 and output layers contribute only a small fraction. The model is therefore both computationally efficient and storage‑friendly, with a footprint that can be further reduced through pruning and quantisation.
#todo naming convention for the number of neurons per layer...

### 4.2.7 Training Protocol

<span style="color: #808080;">[Training Protocol]</span> The base model is trained on the full training set (approximately [dataset_size] positions) using the Adam optimiser with a learning rate of [lr_start], linearly decayed to [lr_end] over the course of training. We use a batch size of [batch_size] and train for up to 1000 epochs. The model's performance is evaluated on a held‑out test set of random positions, [test_fraction] of the total dataset, ensuring that generalisation is measured on unseen data. The training is conducted on a *DGX Nvidia Spark GPU*.

#todo update numbers: number of positions, batch size

---

## 4.3 Sample Gradient Computation


### 4.3.1 Implementation with PyTorch

<span style="color: #808080;">[Sample Gradients with PyTorch]</span> The sample gradients are computed using PyTorch's automatic differentiation engine. For each batch of positions, a forward pass through the base model is performed to obtain WDL logits. Then, the soft cross-entropy loss is computed against the teacher labels. Finally, `torch.autograd.grad` is called to obtain the gradients of the loss with respect to the head parameters.

<span style="color: #808080;">[The implementation in practice]</span> The computation proceeds as follows. For a batch of $M$ positions, the base model $f_{w_{\text{base}}}$ produces logits $\ell_i \in \mathbb{R}^3$ for each position. The loss is computed as the average soft cross-entropy over the batch:

$$\mathcal{L}_{\text{batch}} = \frac{1}{M} \sum_{i=1}^M \mathcal{L}_{\text{CE}}(\text{softmax}(\ell_i), \hat{p}_i)$$

where $\hat{p}_i$ are the teacher's WDL probabilities. We then call:

```python
grads = torch.autograd.grad(
    loss,
    head_parameters,
    retain_graph=False,
    create_graph=False
)
```

The `head_parameters` list contains the trainable parameters of the L2 layer and the output head: $(W_{L2}, b_{L2}, W_{out}, b_{out})$. The L1 weights are excluded, as they remain frozen throughout the entire pipeline.

#todo implicit or explicit bias?

The resulting gradient tensors are detached from the computation graph to free memory, then flattened and concatenated into a single vector per position:

$$\Delta_i = \text{concat}\left[\text{vec}(\nabla_{W_{L2}} \mathcal{L}_i),\; \nabla_{b_{L2}} \mathcal{L}_i,\; \text{vec}(\nabla_{W_{out}} \mathcal{L}_i),\; \nabla_{b_{out}} \mathcal{L}_i\right]$$

The computation is parallelised across the GPU and performed in a single pass over the dataset. Gradients are computed in batches of 1024 positions, and the resulting vectors are accumulated on disk rather than held in memory, preventing memory exhaustion. The `retain_graph=False` option ensures that the computational graph is freed after each batch, further reducing memory usage.

#todo check batch size

### 4.3.2 Parameter Selection

<span style="color: #808080;">[Why we exclude L1]</span> The gradients are computed with respect to the trainable parameters of the head only, namely the weights and biases of the L2 layer and of the output layer: $(W_{L2}, b_{L2}, W_{out}, b_{out})$. The L1 accumulator is deliberately excluded. This choice follows from the design of the method: the L1 representation is shared across all experts and frozen throughout the pipeline, so its parameters are never updated during expert fine-tuning and their gradients are irrelevant to the partition. Including them would also inflate the dimensionality of the gradient vectors by an order of magnitude, since the L1 weights alone account for the majority of the model's parameters, and would make both storage and clustering substantially more expensive without contributing information that the partition can use.

<span style="color: #808080;">[Signature of the sample]</span> The selected parameters are flattened and concatenated into a single vector per position. If the L2 layer has weight matrix $W_{L2} \in \mathbb{R}^{2W \times H}$ and bias $b_{L2} \in \mathbb{R}^{H}$, and the output layer has $W_{out} \in \mathbb{R}^{H \times 3}$ and bias $b_{out} \in \mathbb{R}^{3}$, the resulting gradient vector has dimension $P_{\text{head}} = 2W \cdot H + 4H + 3$, which for the architecture used in this work amounts to approximately 100,000 values per position. This is the representation that is normalised, stored, and clustered in the subsequent steps.

#todo check 100k

### 4.3.3 Normalisation

<span style="color: #808080;">[L2 Normalization - what we do]</span> Each gradient vector is L2-normalised before storage, so that the clustering operates on directions rather than magnitudes. The normalisation is applied element-wise to every vector in the batch:

$$\Delta_i^{\text{norm}} = \frac{\Delta_i}{\|\Delta_i\| + \epsilon}$$

where $\|\Delta_i\|$ is the Euclidean norm of the flattened gradient and $\epsilon = 10^{-8}$ prevents division by zero for degenerate vectors. The operation is vectorised across the batch: the norms are computed in a single reduction over the flattened buffer, ~~and the division is performed before the batch is written to disk~~. In practice, no gradient in the dataset is exactly zero, but the epsilon guard is retained for safety.

<span style="color: #808080;">[What we don't do]</span> The same normalisation is applied uniformly to all gradients. No per-dimension standardisation is performed, and no additional scaling is introduced. This ensures that the stored vectors all lie on the unit hypersphere, so that the Euclidean distance between two normalised vectors is a monotone function of their cosine similarity, and the clustering algorithm effectively operates on angles rather than magnitudes. ~~Normalising before storage also avoids the need to recompute norms during clustering, which matters given the size of the dataset and the cost of reading it from disk.~~

### 4.3.4 Storage and Memory Management

<span style="color: #808080;">[Sample Gradients take a lot of memory]</span> Computing and storing sample gradients for [dataset_size] positions presents a significant practical challenge. Each gradient vector has dimension $P_{\text{head}} \approx 17,000$, corresponding to the flattened parameters of the L2 layer and output head. Storing the full set in 32-bit floating-point would require approximately 340 GB. We therefore adopt half-precision storage, reducing this to roughly 170 GB while preserving sufficient numerical precision for clustering, as the gradients are normalised and clustered based on their directions rather than their exact magnitudes.

<span style="color: #808080;">[Memory Mapping]</span> The gradients are stored in memory-mapped `.npy` files, which provide efficient random access without loading the entire dataset into memory. Computation proceeds in batches of 1024 positions, with each batch written to disk immediately after computation and the memory-mapped array pre-allocated to avoid accumulating gradients in GPU or CPU memory. For exploratory analysis, gradient computation can be performed on a subset of the dataset, but for the final MoE architecture we use the full dataset. 

#todo considerations on dimensionality reduction?
#todo replace hard numbers with variables 
#note clustering on a random subset mentioned 

## 4.3.5 Computational Cost and Timing

<span style="color: #808080;">[Where do we spend our compute]</span> Computing sample gradients is the most expensive step of the pipeline after base training itself. Unlike inference, which requires only a forward pass through the model, gradient computation requires both a forward and a backward pass for every batch. The backward pass is the dominant cost, since it involves propagating gradients through the L2 layer and the output layer. The L1 accumulator is frozen and excluded from differentiation, so no gradient is propagated through it; this keeps the backward pass confined to the small head and avoids the cost of differentiating through the largest layer of the network.

<span style="color: #808080;">[Parallelization, GPU, CPU]</span> The computation is *embarrassingly parallel* across batches. Each batch is independent of the others, and the only shared state is the model weights, which are read-only during this step. On a GPU, this parallelism is exploited by processing batches of 1024 positions at a time and running them through the model in sequence, with the backward pass overlapping the next forward pass through the use of asynchronous kernel launches. On a CPU, the same batching strategy applies, but the throughput is substantially lower, and the per-position cost becomes the bottleneck. For this reason, the gradient computation is performed on a GPU, where the throughput is roughly two orders of magnitude higher than on a modern multi-core CPU. The exact timing depends on the batch size, the size of the head, and the speed of the storage device used to write the gradients, since the I/O can become the limiting factor when the model itself is small.

<span style="color: #808080;">[Considerations on effective batching and GPU]</span> The batching strategy is designed to balance memory usage and throughput. A batch size of 1024 positions is large enough to saturate the GPU and small enough to fit comfortably in device memory, including the activations needed for the backward pass. Each batch is processed independently, and the resulting gradient vectors are written directly to the memory-mapped output file before the next batch begins. This has two effects: it bounds the memory footprint of the step to a single batch, regardless of the total dataset size, and it makes the computation resilient to interruption, since the gradients already written to disk remain valid. The main cost of this approach is that the GPU is idle during the write phase. For large batches, the write time is negligible compared to the backward pass; for small batches, the overhead becomes more significant, which is one of the reasons for choosing a batch size at the upper end of what the memory allows.

<span style="color: #808080;">[Stats, scaling, smoke test]</span> Under the configuration used in this work, computing gradients for the full dataset of [dataset_size] positions takes approximately [gradient_time] on a DGX Nvidia Spark GPU, with the storage writes contributing a minor fraction of the total time. The throughput scales roughly linearly with the number of positions, so the computation can be halted at any point and resumed later without loss, which is useful when a subset is sufficient for an exploratory analysis. A random subsample of the dataset is often used for tuning the clustering parameters before committing to a full pass, since the clustering cost also grows with the size of the stored gradients and the number of iterations.

#note we do not recommend to run the clustering on a subset of the data points - Mini-batch already looks at the sub-sample automatically

---

## 4.4 Clustering

<span style="color: #808080;">[The choice of clustering algorithm]</span> The clustering step operates on the full set of normalised sample gradients produced in the previous stage. The gradients are loaded from the memory-mapped file ~~in chunks~~, and the clustering is performed with **Mini-Batch K-Means**, as described in Section 3.5.2. The choice of Mini-Batch K-Means over the standard K-Means follows from the size of the dataset: with millions of gradient vectors in dimension $P_{\text{head}} \approx 100,000$, the batch updates provided by Mini-Batch K-Means reduce both memory consumption and convergence time, while producing partitions that are nearly indistinguishable from those of the full algorithm on this scale.

#todo remember to try better clustering algorithms - which ones scale better with the n-dim?

<span style="color: #808080;">[Choosing B]</span> The clustering is run for a range of values of $B$, specifically $B \in \{2, 4, 8, 16, 32\}$, so that the effect of the number of experts on the final evaluation quality can be assessed. The Mini-Batch K-Means implementation uses a batch size of 10,000 gradients and is initialised with the $k$-means++ scheme, which improves the stability of the final partition by spreading the initial centroids across the data. The algorithm is run for a fixed number of iterations with early stopping on the within-cluster sum of squares, and the random seed is fixed so that the partitions are reproducible across runs. The trained centroids and the per-position cluster assignments are stored for the subsequent dispatcher training and expert fine-tuning steps.

<span style="color: #808080;">[DPC]</span> As a secondary reference, Density Peak Clustering is applied to a random subsample of the gradients, typically a few hundred thousand positions, to avoid the quadratic cost of computing pairwise distances on the full dataset. The number of clusters identified by this method is not used directly to set $B$, but serves as a qualitative check on whether the values explored with Mini-Batch K-Means are consistent with the natural structure of the gradient space. If the density-based algorithm consistently selects a number of clusters within the explored range, this provides some evidence that the partition is not being forced by an arbitrary choice of $B$.

<span style="color: #808080;">[Dim-red]</span> The structure of the partition is visualised using two complementary tools. *Principal Component Analysis* is applied to a subsample of the gradients to project them onto the first few principal components, and the projection is coloured by cluster assignment to reveal the gross geometry of the partition. *t-SNE* is used on a smaller subsample to produce a non-linear embedding that preserves local neighbourhoods, which often reveals structure that the linear projection misses. Both visualisations are produced for several values of $B$, so that the progressive refinement of the partition can be inspected. These plots are not used to select the final value of $B$, which is decided on the basis of the downstream evaluation metrics, but they provide a useful qualitative check on the clustering and help identify whether any bucket is dominated by a narrow region of the state space.

#todo did we touch on how we account for multiplicity?

---

## 4.5 Dispatcher Training

<span style="color: #808080;">[Goal of the dispatcher]</span> The dispatcher is trained to predict the cluster assignment of a position from the L1 activations produced by the frozen base model. The training data consists of the same set of positions used for clustering, with the cluster labels obtained from Mini-Batch K-Means serving as targets. Since the L1 weights are frozen and the activations depend only on the input features, the L1 activations can be computed once for the entire dataset and cached. This reduces the dispatcher training to a simple classification problem on a fixed set of features, and the entire procedure completes in a fraction of the time required for gradient computation or expert fine-tuning.

<span style="color: #808080;">[The input]</span> The input to the dispatcher is the concatenated L1 activations after the side-to-move reorder, $h = [h_{\text{first}} \| h_{\text{second}}] \in \mathbb{R}^{2W}$, matching the input received by the L2 layer of the expert heads. Using the same representation as the heads ensures that the dispatcher sees the position in the same perspective-aligned form, so that the routing decision is consistent with the expert that will eventually process the position. An alternative would be to use only the side-to-move activations $h_{\text{own}}$, halving the input dimension and the number of dispatcher parameters. This was considered but not adopted, because the opponent's perspective provides additional information about the position at negligible cost, and because the full concatenation matches the input of the expert heads more closely.

#todo alternatively use only h_first (more stable)?

<span style="color: #808080;">[Architecture]</span> The dispatcher is a single linear layer mapping the $2W$-dimensional input to $B$ logits, with no hidden layers and no non-linear activation beyond the softmax used during training. The softmax is discarded at inference, where the predicted bucket is simply the argmax of the logits. This architecture was chosen for its minimal inference cost and because its limited expressiveness is less likely to overfit, as discussed in Section 3.6.2, and it is trained separately for each value of $B$ explored in the clustering stage.

<span style="color: #808080;">[Training]</span> Training uses the standard cross-entropy loss between the predicted distribution over buckets and the one-hot cluster assignment. The optimisation is performed with Adam, using a learning rate of $10^{-2}$ decayed to $10^{-3}$, a batch size of 1024, and early stopping on a held-out validation set comprising 10% of the training positions. Convergence typically occurs within a few epochs, and the validation accuracy is monitored to detect overfitting. No class weighting is applied, even when the cluster sizes are imbalanced, because the dispatcher's objective is to approximate the partition as faithfully as possible, and artificially balancing the classes would distort the routing behaviour relative to the clustering. The trained dispatcher parameters are stored alongside the expert heads and loaded at inference time as part of the MoE model.

#question: is weighting unnecessary or indifferent?

---

## 4.6 Expert Fine-Tuning

<span style="color: #808080;">[Fine-tuning as the final step]</span> With the partition fixed and the dispatcher trained, the final training step produces the expert heads. Each expert is initialised from the base head and fine-tuned on the positions assigned to its bucket, with the L1 accumulator kept frozen throughout. The implementation follows the procedure described in Section 3.7, and the practical details of the training schedule are described here.

<span style="color: #808080;">[Running the fine-tuning]</span> The fine-tuning runs over the buckets in sequence, each expert trained independently of the others. Because the L1 activations are pre-computed and cached, the fine-tuning only requires forward and backward passes through the small head, not the full model. This makes the procedure fast relative to base training, and it allows the experts to be trained with the same optimiser and loss used for the base model. The initialisation from the base head ensures that every expert starts from a well-trained configuration and that the differences between experts reflect the data they see, not differences in initialisation.

#todo check how much faster

<span style="color: #808080;">[The training schedule]</span> The choice of training schedule is guided by the size of the buckets and the risk of overfitting. In the present work the dataset is large enough that no bucket is small, and the training proceeds for a fixed number of sweeps over the bucket, with the cross-entropy on a held-out portion of the bucket serving as the stopping signal. The held-out portion is drawn from the same bucket, so the stopping criterion reflects performance on the type of positions the expert is intended to handle. When a bucket is small, early stopping is essential, because an expert trained for too long on a limited set of positions will memorise them rather than generalise. When a bucket is large, the risk is lower, and the training can proceed for more sweeps without overfitting, but the gains from additional sweeps diminish quickly and the additional cost is not justified. The schedule used in the experiments is one to two sweeps for each bucket, with a learning rate lower than the one used for base training to avoid destabilising the shared representation.

<span style="color: #808080;">[Results]</span> The results of the fine-tuning are evaluated per bucket, using the cross-entropy of the expert on its own held-out data. This is the same metric used to monitor the base model, and it allows a direct comparison between the expert and the base head on the subset of positions that the expert was trained for. The per-bucket cross-entropy is reported in Chapter 5, along with the corresponding metrics for the base model and the dispatcher accuracy, so that the effect of specialisation can be assessed separately from the effect of routing.

#review

---

## 4.7 Final MoE Model

<span style="color: #808080;">[Structure]</span> The final MoE model assembles the three components produced by the pipeline into a single evaluation function: the frozen L1 accumulator, the trained dispatcher, and the $B$ expert heads. In implementation terms, these correspond to three sets of parameters loaded from disk at startup and held in memory for the duration of the search. The L1 weights are shared and read-only, the dispatcher is a single linear layer with $2W \times B$ weights and $B$ biases, and each expert head contains its own L2 and output parameters initialised from the base head and fine-tuned on its bucket.

<span style="color: #808080;">[Inference]</span> The inference path is linear and adds only one stage to the base model. The engine encodes the position into its sparse feature representation and updates the L1 accumulator incrementally, as it does for the single-head model. The resulting activation vector, after the side-to-move reorder, is passed to the dispatcher, which computes $B$ logits and selects the index of the maximum. The corresponding expert head is then evaluated on the same activation vector, producing WDL logits that are converted to probabilities via the softmax, and the scalar evaluation is derived as $v = p_W - p_L$ from the side-to-move perspective. No other expert is evaluated, and no ensembling is performed. The routing decision is therefore a single matrix-vector multiplication followed by an argmax, and the cost of the head forward pass is identical to that of the base model.

<span style="color: #808080;">[Memory cost of MoE]</span> At the implementation level, the expert heads are stored as separate parameter blocks but share the same architecture, which allows the forward pass to be dispatched to the appropriate block using a single index. The dispatcher parameters and the expert heads are stored separately from the L1 weights, so that the shared representation can be loaded once and reused regardless of the number of experts. The total memory footprint of the model is the sum of the L1 weights, the dispatcher weights, and the $B$ expert heads, which for small $B$ remains modest relative to the L1 layer. The additional cost introduced by the MoE architecture over the base model is therefore bounded by the size of the dispatcher and the $B-1$ additional heads, and the inference path adds only the dispatcher operation to the cost of a single-head evaluation.

<span style="color: #808080;">[Wrapping up the section]</span> The assembled model is integrated into Cfish in the following sections. Its behaviour depends on the choices made throughout the pipeline, and in particular on the value of $B$ and on the quality of the partition. The evaluation of the resulting engine, both in terms of the accuracy of the value estimates and in terms of the cost of the additional dispatcher, is reported in Chapter 5.

#todo remove the last paragraph?

---

## 4.8 Cfish as the Host Engine

<span style="color: #808080;">[Deploying our NNUE with Cfish]</span> The Mixture-of-Experts NNUE developed in this chapter is not intended to run as a standalone evaluator. It is designed to replace the evaluation component of an existing chess engine, so that its behaviour can be assessed within the full search pipeline, and so that its cost can be measured against the constraints of the target hardware. The host engine selected for this purpose is Cfish, a port of Stockfish written in plain C by Ronald de Man and first published on GitHub in July 2016. Cfish shares the same search and evaluation logic as its C++ counterpart, but compiles to a smaller and more portable binary, which makes it a natural fit for embedded targets. Since August 2020, Cfish has included the NNUE evaluation ported from Stockfish, together with SIMD-optimised code paths for AVX2 and AVX-512.

<span style="color: #808080;">[The evaluation function and search in Cfish]</span> The evaluation function in Cfish follows the *HalfKP* feature set, in which the active inputs are the positions of the non-king pieces relative to the square of the friendly king. The accumulator is updated incrementally as moves are made, and the network output is a scalar value in centipawns from the side-to-move perspective. The search is an alpha-beta variant with iterative deepening, move ordering, and a transposition table, and it calls the evaluation function at every leaf node. The integration described in the following section replaces the single NNUE head with the MoE architecture, while leaving the search tree, the move generation, and the transposition table unchanged.

<span style="color: #808080;">[Target hardware]</span> The target hardware for the deployment is the Wio Terminal, an ARM board built around the ATSAMD51P19 Cortex-M4 processor running at 120 MHz. The board provides 512 KB of flash memory, 192 KB of RAM, and 4 MB of external flash. These figures impose hard limits on the size of the model and on the amount of state that can be kept in memory during search. The L1 accumulator must fit within the available RAM alongside the transposition table and the search stack, and the expert heads and dispatcher must fit within the flash budget together with the engine code. The evaluation function must also be fast enough to sustain a reasonable number of nodes per second, which in turn determines the search depth the engine can reach within a given time control. The precise throughput depends on the quantisation strategy and on the efficiency of the integer arithmetic, and is measured as part of the experimental evaluation in Chapter 5.

#todo remove explicit reference to Wio 

---

## 4.9 Integration with Cfish

<span style="color: #808080;">[How to integrate MoE NNUE in Cfish]</span> The integration of the NNUE MoE into Cfish follows the same structure that Cfish uses for the standard NNUE, with the addition of the dispatcher and the selection of the appropriate expert head. The entry point is the `evaluate()` function, which is called at every leaf node of the search. In the original engine, this function computes the accumulator from the current position, passes it through the single head, and returns a centipawn score. In the MoE version, the same accumulator is used, but the head is selected by the dispatcher before the output is computed.

<span style="color: #808080;">[The accumulator layer - why only one?]</span> The accumulator update is unchanged. Cfish maintains the accumulator incrementally as moves are made and unmade during the search, adding and subtracting the contributions of the pieces that change. Because the L1 weights are shared across all experts and frozen, the incremental update logic requires no modification: the accumulator is a single vector, computed once per position and reused by whichever expert the dispatcher selects. This is the practical consequence of freezing L1, and it is what makes the integration tractable. If each expert had its own accumulator, the engine would have to maintain $B$ separate accumulator states and update them all on every move, which would multiply the cost of the most frequent operation in the search.

<span style="color: #808080;">[The dispatcher - a small additional cost]</span> The dispatcher is invoked after the accumulator is computed and before the head forward pass. It reads the concatenated activations in the perspective-aligned order, computes $B$ logits, and returns the index of the maximum. The forward pass of the selected expert is then performed on the same activation vector, using the same code path as the base head. The additional cost per evaluation is therefore one linear layer of size $2W \times B$ and one argmax over $B$ values, both of which are inexpensive relative to the head forward pass. For the values of $B$ explored in this work, the dispatcher adds a small but measurable increment to the per-node cost, and the effect on nodes per second is reported in the evaluation.

<span style="color: #808080;">[Quantization]</span> All weights are quantised to 8-bit integers for storage and inference. The L1 weights, the dispatcher weights, and the expert head weights are all stored as int8, and the biases are stored as int32 to preserve the dynamic range of the accumulation. The accumulator itself is maintained in int16, which provides sufficient precision for the CReLU-clipped activations while keeping the vector within a compact representation. The multiply-accumulate operations at the dispatcher and the head use int32 accumulators, which is the standard integer path in NNUE implementations and is supported efficiently by the Cortex-M4 DSP instructions. The output of the head is a set of three int32 logits, converted to a scalar evaluation through a fixed-point approximation of the softmax difference $p_W - p_L$.

#todo measure how quantization impacts inference
#todo int32 accumulators make it so that the errors don't propagate with the sum
#todo explain Cortex-M4 DSP instructions?

<span style="color: #808080;">[About the hardware]</span> The memory layout is organised so that the largest component, the L1 weight matrix, is stored in the external flash and accessed through the board's memory-mapped interface, while the accumulator and the search state reside in the internal RAM. The dispatcher and the expert heads are small enough to be stored in the internal flash alongside the engine code, which avoids the latency of external memory access on the critical path. The expert heads are stored as a contiguous block indexed by the dispatcher output, so that selecting an expert amounts to offsetting a pointer into the block rather than searching a data structure. This layout keeps the per-evaluation memory access pattern predictable and cache-friendly, which matters on a device with no hardware cache.

#note: the part of the Wio Terminal is not the point of the thesis, more so of the internship. move the focus to the bucketing. remove?

<span style="color: #808080;">[Validation - precision and cost]</span> The integration is validated in two stages. First, the MoE model is run on a set of positions with known evaluations to confirm that the dispatcher selects the correct expert and that the output matches the reference implementation on the host. Second, the full engine is run on a set of test positions to measure nodes per second and search depth against the unmodified Cfish with the same base NNUE, so that the cost of the MoE extension can be isolated from the cost of the search itself. The results of these measurements are reported in Chapter 5, together with the evaluation accuracy of the resulting engine.

---

> **Note for AI**: *The parts marked with a #todo are yet to be completed. The tagged comments are NOT to be exported to the LaTeX document, and are not meant to be implemented while exporting this document to LaTeX. The gray labels are for clarity only and must not be transferred to the LaTeX. Placeholders in square brackets (e.g. `[dataset_size]`, `[W]`) must be replaced with the current values from `_ai-info_.md` when converting this document to LaTeX.*

[[5 - Results]]
