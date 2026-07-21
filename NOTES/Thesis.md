
> *Relatore: Alessio Ansuini* 

---
 
## Idee con Ansuini

"Task vectors (in weight space / representation space) T1, T2, …, T8 have specific vectors in the weight space. Is it possible / useful to define task vectors that bring you from one head to another with minimal information/performance loss? This may be the case if the head vectors live in a linear submanifold of low dimensionality (are we so lucky? Maybe not!)"

[Task vectors](https://arxiv.org/abs/2212.04089), "distance between layers" in parameter space (permutations!), [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)


## Finding Optimal Feature Combinations for Maximally-Diverse Experts

Basically, what I will try next is the following: training of a single NNUE (L1 + L2 + Output layer), then fine-tune the L2 and Output layers for every head, but with a twist; find a set of features of the board states (number of pieces, queen or no queen, bishop pair, rook pair, king position, ... `-> (30/32, 1, 0, 1, 5/8, 1/8, ...)` etc...). We re-do the fine-tuning on various combinations of features to obtain various **normalized delta vectors** (after minus before fine-tuning). We pick the sets of partitions

<div align="center">
    <img src="images/arrows.png" width="300">
</div>

Be careful: tiling of the position space must be complete and with no overlaps (like a function $f : s_{board} \rightarrow i_{bucket}$) , and the bucket must probably be of (more or less) uniform size.

bucket i = set of all board states in the database with index $f(s)==i$

expert model i trained on bucket i (**one sweep**) produces a delta vector. 

Score for f = some sort of average angular distance (**cosine similarity**) between delta vectors, but very low if the smallest distance is small.


Note: the deltas should point in all directions by default. If they don't, the training was probably not complete.

---
---

# AI-Formalized Experimental Plan: Task Vectors & Optimal Bucketing

## 1. Core Hypothesis

Task vectors (as defined in [Ilharco et al., 2022]) in weight space or representation space for different experts may lie on a low-dimensional linear subspace. If this holds, it would imply that expert heads can be efficiently represented and manipulated with minimal information/performance loss.

**Question to test**: Do the delta vectors between experts live in a linear submanifold of low dimensionality?

**Related concepts**:
- Task vectors (Ilharco et al., 2022): `τ_t = θ_ft - θ_pre` where `θ_pre` is a pre-trained model and `θ_ft` is a fine-tuned version on task `t`
- Distance between layers in parameter space (with permutation invariance considerations)
- Lottery Ticket Hypothesis (Frankle & Carbin, 2018)

---

## 2. Experimental Protocol

### 2.1 Phase 1: Train Base Model

Train a **single shared NNUE**:
- **Shared L1**: `844 → W` dense (W ∈ {128, 256})
- **Shared L2**: `2W → H` (single hidden layer, or directly to output)
- **Output**: `H → 1` with tanh output

**Loss**: MSE vs teacher `expected_reward` (Lc0 BT4 labels).

**Result**: Base model `θ_base` with parameters `(W_L1, W_L2, W_out)`.

### 2.2 Phase 2: Define Bucket Function

We seek a function:

$$f : \mathcal{S} \rightarrow \{0, 1, \dots, B-1\}$$

where:
- $\mathcal{S}$ is the set of all board states
- $B$ is the number of buckets (experts)
- The partition must be **complete** and **non-overlapping**:
  - $\bigcup_i f^{-1}(i) = \mathcal{S}$
  - $f^{-1}(i) \cap f^{-1}(j) = \emptyset \quad \forall i \neq j$

**Candidate features for $f$**:
- Number of pieces (`p`, 3-32)
- Queen presence (`q ∈ {0,1}`)
- Bishop pair presence (`bp ∈ {0,1}`)
- Rook pair presence (`rp ∈ {0,1}`)
- King position (compressed to 8 or 5 zones)
- Material imbalance
- Pawn structure (isolated/doubled counts)
- Mobility (number of legal moves)
- Any hand-crafted feature computable cheaply

Each candidate defines a different **partition** of the state space.

**Uniform bucket size**: Buckets should contain approximately equal numbers of training samples to avoid biased training.

### 2.3 Phase 3: Fine-Tune Experts (One Sweep)

For each bucket `i`:

1. **Freeze** the shared L1 weights (`W_L1`) — this keeps the representation space fixed.
2. **Fine-tune** only the L2 (`W_L2`) and Output (`W_out`) weights on bucket `i`'s training data.
3. Train for **exactly one epoch** (one sweep) on that bucket.

**Result**: Fine-tuned model `θ_i = (W_L1, W_L2 + ΔL2_i, W_out + Δout_i)`.

**Delta vector**: $\delta_i = \theta_i - \theta_{base}$ (projected only on the fine-tuned layers).

---

## 3. Selecting Optimal Feature Combinations

### 3.1 Objective

Given a candidate partition function `f`, define a **score** that measures how "well-separated" the delta vectors are.

**We want**: δ vectors for different buckets to be **as different as possible** (maximally diverse experts), but ideally lying on a low-dimensional subspace.

### 3.2 Loss/Score Function

**Score(f) = average angular distance between δ_i and δ_j**

$$\text{Score}(f) = \frac{1}{\binom{B}{2}} \sum_{i < j} d(\delta_i, \delta_j)$$

where:

$$d(\delta_i, \delta_j) = \frac{\delta_i \cdot \delta_j}{\|\delta_i\| \|\delta_j\|} \quad \text{(cosine similarity)}$$

or, if using Euclidean:

$$d(\delta_i, \delta_j) = \|\delta_i - \delta_j\|^2$$

**But careful**: If the smallest distance is very small, the score should be low (penalize clusters of similar δ vectors).

**Proposed**: Weighted average that penalizes the minimum distance:

$$\text{Score}(f) = \frac{1}{\binom{B}{2}} \sum_{i < j} d(\delta_i, \delta_j) \times \min_{i<j} d(\delta_i, \delta_j)^{-1}$$

or simpler:

$$\text{Score}(f) = \text{mean distance} - \lambda \cdot \text{min distance}$$

### 3.3 Search Strategy

Exhaustive search over all feature combinations may be infeasible if the feature space is large. Instead:

1. **Greedy forward selection**: Start with empty set, iteratively add the feature that maximizes Score(f).
2. **Random search**: Sample random feature subsets and evaluate Score(f).
3. **Genetic algorithm**: Evolve feature subsets toward higher Score(f).

---
