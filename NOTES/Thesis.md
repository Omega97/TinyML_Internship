
> *Relatore: Alessio Ansuini* 

---

### Proposed Solution: Two-Stage Proxy Clustering for Expert Dispatching

**Objective** To construct a dispatcher function $g_\phi$ that partitions the state space $\mathcal{S}$ into $B$ discrete buckets, maximizing the divergence of the resulting task vectors in the weight space $\mathcal{W}$. Because the dispatcher can only observe $\mathcal{S}$ while the optimization target resides in $\mathcal{W}$, computing the exact gradient $\nabla_\phi \, S(\mathcal{D})$ is intractable. To bypass this, we propose a decoupled, two-step heuristic.

<div align="center">
    <img src="../plots/thesis_dispatcher_architecture.png" width="600">
</div>

#### Training

- **Step 1: Clustering the Task Vectors** - We first compute the instance-specific task vectors $\delta_i = \nabla_w \; \mathcal{L}_{acc}(f_\theta(s_i), v_i)$ for each data point. Then, by **clustering** them, we generate a set of labels - a partition $\mathcal D$ that explicitly groups the board states into maximally diverse clusters.
    
- **Step 2: Training the Dispatcher** - We then train a separate, **lightweight** classifier $g_\phi$ to predict these generated cluster assignments. To improve generalization and reduce computational overhead, $g_\phi$ will not operate on the raw, high-dimensional state space $\mathcal{S}$. Instead, we will map the inputs to a lower-dimensional embedding space, specifically utilizing the activations from the base model's frozen L1 layer as the input features for the dispatcher.


#### Inference

1. Compute the sparse binary representation of the board state
2. Forward pass on L1
3. Pass the activations to the *cluster assignment logits block* (no need for the softmax block)
4. Argmax on the logits to select the appropriate **expert head**
5. Forward pass on the expert head (L2 + value)

#### Conclusion 

This approach successfully decouples the weight-space clustering objective from the state-space routing mechanism. Taking the trained classifier at face value yields a highly functional dispatcher capable of assigning board positions to maximally diverse expert models without requiring end-to-end differentiable clustering.

---

See also: [Thesis Ideas](Thesis%20Ideas.md)

