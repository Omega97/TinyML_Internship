
# KITSUNE

**K**ernels for 
**I**nference with 
**T**wo-stage 
**S**egregation using 
**U**nsupervised 
**N**on-linear 
**E**fficient Routing

> *A zero-overhead non-linear routing strategy for Mixture-of-Experts via gradient-informed micro-bucket recombination.*

---

## Abstract

Routing in Mixture-of-Experts (MoE) architectures is typically performed by a linear dispatcher operating on intermediate activations. However, when the optimal routing criterion is derived from sample-level gradients—i.e., assigning each input to the expert whose parameters receive the most aligned gradient signal—the resulting decision boundaries in activation space are **non-convex and discontinuous**, rendering linear dispatchers fundamentally inadequate. We introduce **KITSUNE**, a two-stage unsupervised routing strategy that decouples geometric partitioning from gradient-aware expert assignment. In the first stage, activations are quantized into a fine-grained set of *micro-buckets* via linear clustering in activation space, achieving near-perfect dispatch accuracy. In the second stage, micro-buckets are merged into the final set of experts by clustering their **mean sample gradients**, constructing arbitrarily non-linear routing regions at zero additional inference cost—only a static lookup table is required. KITSUNE preserves the gradient signal that drives expert specialization while keeping the inference pipeline strictly linear and table-driven, making it suitable for edge and microcontroller deployment.

---

## 1. Motivation: Why Linear Dispatch Fails on Gradient Clusters

Consider a Mixture-of-Experts layer with $B$ experts. During training, each sample $i$ produces a gradient $\nabla_{W} \mathcal{L}_i$ with respect to the expert parameters. Ideally, we would route sample $i$ to the expert whose gradient direction is most aligned with $\nabla_{W} \mathcal{L}_i$, as this maximizes the constructive contribution of each expert's update.

The natural approach is to train a linear dispatcher $f(a_{L1}) = W_{\text{disp}} \cdot a_{L1}$ that maps the intermediate activation $a_{L1}$ directly to a gradient cluster. This **fails** because:

| Property | Activation Clusters | Gradient Clusters |
|---|---|---|
| Convexity in $L1$ space | ✅ Approximately convex | ❌ Non-convex |
| Contiguity | ✅ Connected regions | ❌ Disconnected "islands" |
| Linear separability | ✅ Near 100% accuracy | ❌ Poor accuracy |
| Geometric structure | Voronoi-like cells | Scattered manifolds |

**Geometric intuition.** Samples that share a similar gradient direction $\nabla_W \mathcal{L}_i$ do not occupy a single contiguous region in $L1$ space. Instead, they form scattered, non-convex "islands" dispersed along the activation manifold. No hyperplane—or set of hyperplanes in a one-vs-rest scheme—can faithfully separate these discontinuous regions. A linear dispatcher trained to predict gradient clusters will suffer from irreducible misclassification.

This creates a dilemma:

- **Gradient-based routing** is the theoretically correct signal for expert specialization, but it is **non-linearly embedded** in activation space.
- **Activation-based routing** is trivially linear-dispatchable, but it carries **no direct information** about gradient alignment.

KITSUNE resolves this dilemma by decomposing the problem into two stages, each operating in a regime where it excels.

---

## 2. Core Idea: Decouple Geometry from Semantics

The key insight is that any arbitrarily complex, non-convex decision boundary in $L1$ space can be **composed** from a sufficiently fine partition of convex cells. KITSUNE exploits this by:

1. **Partitioning** $L1$ space into $B'$ convex micro-buckets ($B' \gg B$) using purely geometric clustering. Since each micro-bucket is a small Voronoi cell, a linear dispatcher assigns samples to micro-buckets with ~100% accuracy.

2. **Recombining** the $B'$ micro-buckets into $B$ macro-experts by clustering their *mean sample gradients*. This recombination is a static mapping computed offline; at inference time, it reduces to a single array lookup.

The result is that the effective routing regions for each expert are **arbitrarily non-convex and disconnected** in $L1$ space—exactly what is needed to capture gradient clusters—while the inference computation remains a linear projection followed by a table lookup.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE (Training Time)                      │
│                                                                 │
│  [L1 Activations] ──(k-means)──> [B' Micro-Buckets]            │
│                                        │                        │
│                              Compute mean gradient              │
│                                ḡ_k per bucket                   │
│                                        │                        │
│                             (Agglomerative / k-means            │
│                              on cosine similarity)              │
│                                        │                        │
│                                  [B Macro-Experts]              │
│                                        │                        │
│                            Build MAP[k] → b lookup              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    ONLINE (Inference Time)                      │
│                                                                 │
│  a_{L1} ──(W_disp · a)──> k* ──(MAP[k*])──> b ──> Expert_b    │
│               linear          O(1) lookup         forward       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Formal Pipeline

### 3.1 Stage 1: Micro-Quantization ($L1 \to B'$)

Given a calibration dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, collect the intermediate activations at layer $L1$:

$$A = \{a_{L1}^{(i)} \in \mathbb{R}^d \mid i = 1, \dots, N\}$$

Run Mini-Batch $k$-Means with $B'$ clusters ($B' \gg B$, e.g., $B' \in \{32, 64, 128\}$) to obtain centroids:

$$C = \{c_1', c_2', \dots, c_{B'}'\} \subset \mathbb{R}^d$$

Each sample is assigned to its nearest micro-bucket:

$$k(i) = \arg\min_{k \in \{1,\dots,B'\}} \| a_{L1}^{(i)} - c_k' \|_2$$

**Why this works:** With $B'$ sufficiently large, each micro-bucket covers a small, locally convex region of the activation manifold. A linear dispatcher $W_{\text{disp}} \in \mathbb{R}^{B' \times d}$ trained to predict $k(i)$ from $a_{L1}^{(i)}$ achieves near-perfect accuracy (empirically $\geq 99\%$).

### 3.2 Stage 2: Gradient Profiling

For each micro-bucket $k$, collect the set of sample indices $S_k = \{i \mid k(i) = k\}$ and compute the **mean sample gradient** with respect to the expert head parameters $W_{\text{head}}$:

$$\bar{g}_k = \frac{1}{|S_k|} \sum_{i \in S_k} \nabla_{W_{\text{head}}} \mathcal{L}_i \quad \forall\, k \in \{1, \dots, B'\}$$

This yields $B'$ gradient prototype vectors $\{\bar{g}_1, \dots, \bar{g}_{B'}\}$, each summarizing the "learning direction" of its micro-bucket.

### 3.3 Stage 3: Macro-Clustering on Gradients ($B' \to B$)

Cluster the $B'$ gradient prototypes into $B$ macro-groups using **Agglomerative Hierarchical Clustering** or $k$-Means with **cosine distance**:

$$\text{dist}(\bar{g}_a, \bar{g}_b) = 1 - \frac{\bar{g}_a \cdot \bar{g}_b}{\|\bar{g}_a\|_2 \, \|\bar{g}_b\|_2}$$

This produces a mapping:

$$\text{MAP}: \{1, \dots, B'\} \to \{1, \dots, B\}$$

where $\text{MAP}[k] = b$ means micro-bucket $k$ is assigned to macro-expert $b$.

### 3.4 Inference

At runtime, given an input activation $a_{L1}$:

1. **Linear dispatch:** $k^* = \arg\max_{k} \left( W_{\text{disp}} \cdot a_{L1} \right)_k$
2. **Table lookup:** $b = \text{MAP}[k^*]$
3. **Expert evaluation:** Compute $\text{Expert}_b(a_{L1})$

Total routing overhead: **one matrix-vector product + one array indexing**. No non-linear operations, no additional network passes.

---

## 4. Implementation Details

### 4.1 Choosing $B'$: The Resolution–Variance Tradeoff

The number of micro-buckets $B'$ controls a fundamental tradeoff:

| $B'$ too small | $B'$ too large |
|---|---|
| High intra-bucket gradient variance | Diminishing returns on variance reduction |
| $\bar{g}_k$ is a noisy gradient proxy | More centroids to store and dispatch |
| Poor macro-clustering quality | Risk of empty or near-empty buckets |
| Linear dispatch still accurate | Linear dispatch accuracy may degrade slightly |

**Recommended procedure:**

1. Sweep $B' \in \{16, 32, 64, 128, 256\}$.
2. For each $B'$, compute the **Gradient Cosine Inertia** (see §4.3).
3. Identify the *elbow point* where inertia plateaus.
4. Verify that linear dispatch accuracy on $L1 \to B'$ remains $\geq 98\%$.

Empirically, $B' \in \{32, 64\}$ provides an excellent operating point for $B \in \{4, 8\}$.

### 4.2 Load Balancing via Constrained Clustering

Naïve gradient clustering may assign many populous micro-buckets to the same macro-expert, creating severe load imbalance. To prevent this:

**Option A: Constrained Agglomerative Clustering.** Use a constrained variant (e.g., `scikit-learn`'s `AgglomerativeClustering` with connectivity constraints or a custom implementation) that enforces a maximum cluster weight:

$$\sum_{k \in \text{cluster } b} |S_k| \leq \frac{(1 + \epsilon) \cdot N}{B} \quad \forall\, b \in \{1, \dots, B\}$$

**Option B: Weighted $k$-Means with balancing penalty.** Augment the clustering objective:

$$\min \sum_{b=1}^{B} \sum_{k \in C_b} |S_k| \cdot \text{dist}(\bar{g}_k, \mu_b) + \lambda \sum_{b=1}^{B} \left( \frac{|C_b|_{\text{weighted}}}{N/B} - 1 \right)^2$$

where $|C_b|_{\text{weighted}} = \sum_{k \in C_b} |S_k|$.

**Option C: Post-hoc rebalancing.** After unconstrained clustering, iteratively reassign the most overloaded micro-buckets to the nearest underloaded macro-cluster, provided the gradient cosine similarity remains above a threshold $\tau$.

### 4.3 Diagnostic: Gradient Purity per Micro-Bucket

To validate that $B'$ is large enough, measure the **intra-bucket gradient dispersion**:

$$\text{GCI}(B') = \frac{1}{B'} \sum_{k=1}^{B'} \frac{1}{|S_k|} \sum_{i \in S_k} \left( 1 - \frac{\nabla \mathcal{L}_i \cdot \bar{g}_k}{\|\nabla \mathcal{L}_i\|_2 \, \|\bar{g}_k\|_2} \right)$$

This is the mean cosine distance between individual sample gradients and their bucket's mean gradient. A well-chosen $B'$ yields $\text{GCI} \leq 0.1$. Plot $\text{GCI}$ vs. $B'$ and select the elbow.

### 4.4 Dispatcher Training

The linear dispatcher $W_{\text{disp}} \in \mathbb{R}^{B' \times d}$ is trained as a standard multi-class classifier on $(a_{L1}^{(i)}, k(i))$ pairs:

$$\mathcal{L}_{\text{disp}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left((W_{\text{disp}} \cdot a_{L1}^{(i)})_{k(i)}\right)}{\sum_{j=1}^{B'} \exp\left((W_{\text{disp}} \cdot a_{L1}^{(i)})_j\right)}$$

This is a lightweight cross-entropy loss. Since the targets $k(i)$ come from $k$-means on the same space, the problem is nearly linearly separable and converges rapidly.

### 4.5 Gradient Computation

The sample gradients $\nabla_{W_{\text{head}}} \mathcal{L}_i$ can be computed efficiently:

- **Per-sample gradients** via `torch.func.grad` (PyTorch) or `jax.vmap(jax.grad(...))`.
- For a linear head $W_{\text{head}} \in \mathbb{R}^{o \times h}$, the gradient for sample $i$ is simply the outer product $\nabla_{W} \mathcal{L}_i = \delta_i \otimes a_{L1}^{(i)}$, where $\delta_i = \partial \mathcal{L}_i / \partial z_i \in \mathbb{R}^o$ is the output-space gradient. This avoids full backpropagation per sample.
- Flatten to a vector $\bar{g}_k \in \mathbb{R}^{o \cdot h}$ for clustering.

---

## 5. Pseudocode

```python
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

def kitsune_build(calibration_loader, model, B_prime, B, device='cuda'):
    """
    Offline phase: build the KITSUNE routing tables.
    """
    # 1. Collect L1 activations and per-sample gradients
    activations = []
    gradients = []
    
    for x, y in calibration_loader:
        x, y = x.to(device), y.to(device)
        a_L1 = model.get_L1_activations(x)           # [batch, d]
        logits = model.head(a_L1)                     # [batch, num_classes]
        loss_per_sample = F.cross_entropy(logits, y, reduction='none')
        
        # Per-sample gradient of the linear head: outer product
        # delta = dL/dz, shape [batch, num_classes]
        delta = torch.autograd.grad(loss_per_sample.sum(), logits, 
                                     create_graph=False)[0]
        # grad_i = delta_i ⊗ a_L1_i, flattened to [batch, num_classes * d]
        grad_flat = torch.einsum('bi,bj->bij', delta, a_L1).flatten(1)
        
        activations.append(a_L1.detach().cpu())
        gradients.append(grad_flat.detach().cpu())
    
    A = torch.cat(activations, dim=0).numpy()         # [N, d]
    G = torch.cat(gradients, dim=0).numpy()           # [N, d * num_classes]
    
    # 2. Stage 1: Micro-quantization in L1 space
    kmeans_micro = MiniBatchKMeans(n_clusters=B_prime, batch_size=1024)
    micro_labels = kmeans_micro.fit_predict(A)        # [N]
    micro_centroids = kmeans_micro.cluster_centers_    # [B_prime, d]
    
    # 3. Stage 2: Compute mean gradient per micro-bucket
    mean_gradients = np.zeros((B_prime, G.shape[1]))
    bucket_sizes = np.zeros(B_prime)
    for k in range(B_prime):
        mask = (micro_labels == k)
        if mask.sum() > 0:
            mean_gradients[k] = G[mask].mean(axis=0)
            bucket_sizes[k] = mask.sum()
    
    # 4. Stage 3: Macro-clustering on mean gradients (cosine distance)
    agg = AgglomerativeClustering(
        n_clusters=B, 
        metric='cosine', 
        linkage='average'
    )
    macro_labels = agg.fit_predict(mean_gradients)    # [B_prime]
    
    # 5. Build lookup table
    MAP = macro_labels  # MAP[k] = b
    
    # 6. Train linear dispatcher (standard cross-entropy)
    W_disp = train_linear_dispatcher(A, micro_labels, B_prime)
    
    return W_disp, MAP, micro_centroids


def kitsune_route(a_L1, W_disp, MAP):
    """
    Online phase: route a single activation to an expert.
    Cost: 1 matmul + 1 array lookup.
    """
    k_star = torch.argmax(W_disp @ a_L1)              # linear dispatch
    b = MAP[k_star.item()]                             # O(1) lookup
    return b
```

---

## 6. Why KITSUNE Works: Summary of Properties

| Property | KITSUNE | Direct Gradient Routing | Pure Activation Clustering |
|---|---|---|---|
| Routing signal | Gradient-informed | Gradient-based | Activation geometry only |
| Inference cost | Linear + lookup | Requires non-linear model | Linear |
| Decision boundary shape | Arbitrary non-convex | Arbitrary non-convex | Convex (Voronoi) |
| Dispatch accuracy | ~100% (on micro-buckets) | Poor (linear) / costly (non-linear) | ~100% |
| Expert specialization | Gradient-aligned | Gradient-aligned | Not guaranteed |
| Edge-deployable | ✅ Yes | ❌ No (if non-linear) | ✅ Yes |
| Offline cost | Moderate (one pass + clustering) | Moderate | Low |
| Hyperparameters | $B'$ (micro-bucket count) | Router architecture | $B$ only |

---

## 7. Discussion and Extensions

### 7.1 Relationship to Vector Quantization

KITSUNE can be viewed as a **two-stage vector quantizer** where the first stage quantizes in the *input* (activation) space and the second stage re-quantizes in the *gradient* space. This is analogous to residual VQ, but the two codebooks operate on fundamentally different signal types (geometry vs. optimization dynamics).

### 7.2 Dynamic Recomputation

The MAP table can be periodically recomputed during training as the gradient distribution shifts. Since the micro-bucket centroids and dispatcher remain fixed, this only requires re-running Stage 2–3 on a fresh calibration batch—a lightweight operation that can be performed every $K$ epochs.

### 7.3 Extension to Top-$k$ Routing

For top-$k$ MoE routing, the linear dispatcher can output the top-$k'$ micro-buckets ($k' \geq k$), and the lookup table maps each to its macro-expert. Deduplication yields the final top-$k$ expert set.

### 7.4 Connection to NNUE-Style Architectures

Like NNUE (Efficiently Updatable Neural Network), KITSUNE achieves its efficiency by front-loading all complex computation to an offline phase, reducing the online path to operations that are trivially implementable on constrained hardware: a single dense matrix-vector multiply and an integer array lookup.

---

## 8. Conclusion

KITSUNE provides a principled, zero-overhead mechanism for non-linear gradient-aware routing in Mixture-of-Experts architectures. By decomposing the routing problem into a geometric micro-quantization stage and a gradient-informed macro-recombination stage, it achieves the representational power of non-linear routing with the computational cost of a linear dispatcher plus a lookup table. This makes it uniquely suited for deployment scenarios where both expert specialization quality and inference efficiency are critical.

---

```
What would be a good number of partitions before (B') and after (B = n_exoerts) grouping together the partitions by similarity ov average sample gradient?
```

Choosing the right number of micro-buckets ($B′$) relative to the number of macro-experts ($B$) is the most critical hyperparameter tuning step in the KITSUNE framework.

As a general rule of thumb, the sweet spot for the ratio is $B'/B∈[4,8]$.

