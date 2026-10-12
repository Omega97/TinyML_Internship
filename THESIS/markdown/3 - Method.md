# 3 Methodology and Experimental Pipeline

---

## 3.1 Overview of the Experimental Pipeline

<span style="color: #808080;">[The Pipeline]</span> The core of this work is a systematic experimental pipeline designed to compare various neural evaluation architectures and state-space partitioning strategies under the constraints of embedded chess engines. Rather than focusing on a single novel algorithm in isolation, this thesis evaluates a battery of techniques that take inspiration from other fields, like LLMs and robotics. They span from standard baselines, like vanilla NNUE, to advanced Mixture-of-Experts (MoE) formulations. The pipeline is structured around a rigorous loop of architectural design, hyperparameter selection, large-scale training, and precise evaluation against a strong teacher signal.

<span style="color: #808080;">[The Dataset]</span> The foundation of the entire pipeline is a large-scale dataset of approximately 150 million chess positions. These positions are extracted from human games and labeled with high-quality Win-Draw-Loss (WDL) probability distributions by the Leela Chess Zero (Lc0) engine at search depth 1. This provides a rich, calibrated soft-target distribution that captures the teacher's uncertainty, serving as the ground truth for all subsequent training and evaluation.

#reference to Lc0

<span style="color: #808080;">[The Experimental Loop]</span> The experimental methodology follows three distinct phases for every model variant:

1. **Hyperparameter Selection and Scaling:** Determining the optimal architectural widths, learning rates, and data budgets. This includes running dataset scaling tests to understand when the base architectures saturate, which directly informs the data allocation for MoE expert fine-tuning.
    
2. **Training:** Executing the training procedures, which vary depending on the method (e.g., joint end-to-end training for soft MoE, versus a multi-stage pipeline of base training, clustering, and expert fine-tuning for hard MoE).
    
3. **Evaluation:** Assessing the trained models primarily through the **Cross-Entropy (CE)** against the teacher's WDL distribution on a strictly held-out test set. This metric directly quantifies how well the student model replicates the teacher's probabilistic understanding of the position, serving as the definitive criterion for model selection and comparison.

---
## 3.2 Base Evaluation Architectures
<span style="color: #808080;">[Base Models]</span> Before exploring complex routing and partitioning strategies, it is necessary to establish strong, efficient baselines. The pipeline evaluates two distinct approaches to the core evaluation function.

### 3.2.1 Standard NNUE
<span style="color: #808080;">[Standard NNUE]</span> The primary baseline is the Efficiently Updatable Neural Network (NNUE). This architecture relies on a sparse, handcrafted feature encoding (piece-square tables) mapped through a shared accumulator layer (L1), followed by a dense hidden layer (L2) and a linear output head. The dual-perspective design ensures the network always evaluates the board from the side-to-move's viewpoint. The NNUE baseline represents the state-of-the-art for resource-constrained engines, offering a highly favorable trade-off between evaluation strength and incremental update speed.

#reference to NNUE

### 3.2.2 NNUE with a Learned L1 World Model
<span style="color: #808080;">[World Model]</span> To investigate whether the handcrafted sparse accumulator can be replaced by a learned representation, the pipeline also explores a variant where the conventional L1 accumulator is substituted with a learned "world model" encoder. Instead of relying on predefined piece-square features, this architecture utilizes a feedforward neural network encoder that processes the raw board state. The hypothesis is that shifting the learning effort entirely to a shared, learned representation might yield a more adaptable spatial understanding of the board, albeit at the cost of losing the incremental update efficiency of the standard NNUE accumulator.

---
## 3.3 Mixture of Experts and State-Space Partitioning
<span style="color: #808080;">[MoE Formulations]</span> To test the hypothesis that different regions of the state space require specialized evaluation heads, the pipeline extends the base architectures with Mixture-of-Experts (MoE) formulations. In all MoE variants, the total parameter budget is distributed across multiple expert heads, and a routing mechanism directs each position to the appropriate expert(s). The pipeline explores four distinct routing and partitioning strategies, ranging from simple heuristics to fully differentiable gating networks.

#reference to MoE

### 3.3.1 Hard MoE with Piece-Count Clustering
<span style="color: #808080;">[Piece-Count]</span> The most traditional approach to NNUE bucketing uses handcrafted, domain-specific heuristics. In this variant, the state space is partitioned into discrete buckets based on the scalar piece count (the total material remaining on the board). This serves as the conventional baseline for hard MoE, representing the standard practice in engines like Stockfish. The routing is deterministic, computationally free, and requires no learned dispatcher, but it relies on the assumption that game phase (as approximated by material) is the primary driver of evaluation divergence.

#reference to Stockfish

### 3.3.2 Hard MoE with L1 Activation Clustering
<span style="color: #808080;">[L1 Clustering]</span> Moving away from handcrafted rules, this variant employs unsupervised clustering on the internal representations of the base model. By applying Mini-Batch K-Means to the frozen L1 accumulator activations, positions that "look similar" to the network's feature extractor are grouped together. A lightweight dispatcher is trained to predict these bucket assignments at inference time. This method tests whether partitioning based on the model's current knowledge (its representations) yields better specialization than handcrafted heuristics.

#reference to Mini-Batch K-Means

### 3.3.3 Hard MoE with Sample-Gradient Clustering
<span style="color: #808080;">[Gradient Clustering]</span> The core novel contribution of this thesis is a data-driven partitioning method based on the learning signal itself. Instead of clustering what the model *knows* (activations), this method clusters what the model *needs to adjust* (per-sample gradients). The pipeline for this variant consists of five stages:
1. **Base Training:** Train a reference NNUE to convergence to establish a stable parameter space.
2. **Gradient Extraction:** Compute the per-sample gradients of the loss with respect to the head parameters for every position in the dataset. These vectors are L2-normalized to focus purely on the direction of the required update.
3. **Clustering:** Apply Mini-Batch K-Means to the normalized gradient vectors to discover $B$ distinct buckets of positions with aligned learning requirements.
4. **Dispatcher Training:** Since gradients are unavailable at inference time, a lightweight linear or MLP dispatcher is trained to map the frozen L1 activations to the discovered gradient buckets.
5. **Expert Fine-Tuning:** The base head is cloned $B$ times, and each expert is fine-tuned exclusively on the positions assigned to its corresponding gradient bucket.

### 3.3.4 End-to-End Soft and Sparse MoE
<span style="color: #808080;">[End-to-End MoE]</span> Finally, the pipeline evaluates fully differentiable routing mechanisms trained jointly with the experts. This includes **Soft-Gated MoE** (where the router outputs a softmax distribution and the top-$k$ experts are blended) and **Switch MoE** (a sparse, hard top-1 routing mechanism). Both are trained end-to-end with an auxiliary load-balancing loss to prevent expert collapse. This variant tests whether a jointly optimized routing policy can discover partitions superior to the offline, two-stage clustering methods, despite the strict computational constraints of the target hardware.

#reference to Soft-Gated MoE
#reference to Switch MoE (Switch Transformer)

---
## 3.4 Training and Evaluation Protocol
<span style="color: #808080;">[Protocol Overview]</span> To ensure a fair and rigorous comparison across all architectures and routing strategies, the pipeline enforces a strict training and evaluation protocol. 

### 3.4.1 Hyperparameter Selection and Dataset Scaling
<span style="color: #808080;">[Scaling]</span> Before committing to the final MoE experiments, the pipeline includes a scaling phase to determine the data requirements of the base architectures. By training models of varying widths on progressively larger subsets of the 150M dataset, the pipeline identifies the point at which the base NNUE saturates. This scaling analysis is critical for the MoE variants: it reveals the exact data budget required per expert to avoid underfitting, and highlights the tension between the massive dataset available for base training and the fraction of data each expert receives in a hard MoE setup.

### 3.4.2 Training Procedures
<span style="color: #808080;">[Training]</span> All models are trained using the Adam optimizer with a linearly decaying learning rate. The loss function is universally the **soft cross-entropy** between the model's predicted WDL distribution and the Lc0 teacher distribution. 
#reference to Adam

- For **base models and L1-clustered/gradient-clustered hard MoE**, the training is split into a base phase and an expert fine-tuning phase (where the L1 accumulator remains frozen). 
- For **end-to-end MoE**, the experts and the gating network are updated jointly, with the load-balancing loss weighted to ensure uniform expert utilization. 
- The **piece-count** variant requires no routing training, as the buckets are fixed by definition.

### 3.4.3 Evaluation Metrics
<span style="color: #808080;">[Metrics]</span> The primary metric for model selection and comparison is the **Cross-Entropy (CE)** evaluated on a strictly held-out test set of positions. CE directly measures the divergence between the student's predicted WDL distribution and the teacher's true distribution, capturing both the accuracy and the calibrated uncertainty of the evaluation. 
Additionally, the pipeline tracks:
- **Mean Absolute Error (MAE)** on the scalar expected value ($p_W - p_L$), which correlates with the actual search behavior of the engine.
- **Routing Diagnostics:** For the hard MoE variants, the pipeline evaluates the dispatcher's Top-1 accuracy, as well as clustering agreement metrics (Adjusted Rand Index, Normalized Mutual Information) to quantify how faithfully the deployable routing signal (L1 activations) recovers the offline partition (especially for the gradient-based method).

#reference to Adjusted Rand Index and Normalized Mutual Information

---
