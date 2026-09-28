
# Introduction

#todo signal that we aim to prove that we were training NNUE MoE suboptimally

---

## 1.1 Motivation

<span style="color: #808080;">[Need / Constraint]</span> A chess engine evaluates a large number of positions during search, and the quality of this evaluation directly affects the quality of the resulting moves. On a desktop system, this evaluation can rely on relatively large neural networks. On a microcontroller, memory, computation, and latency constraints impose much tighter limits on the size and cost of the evaluation function. In the latter case, the evaluation function must be small, integer-friendly, and cheap to run at every node of an alpha-beta search. *Efficiently Updatable Neural Networks* (**NNUE**) architectures have become a widely used approach for neural evaluation in modern chess engines.

#note was more journalistic:  A chess engine spends most of its time asking the same question: *how good is this position?* On a desktop, that question can be answered by a large network; on a microcontroller, it cannot. ... 

<span style="color: #808080;">[Heuristic Gap]</span> A single shallow head must represent positions arising from substantially different tactical and positional regimes. This raises the question of whether different regions of the state space could benefit from specialized heads. Different positions may require understanding of very diverse aspects of the game. Mixture-of-Experts (**MoE**) evaluation provides a natural framework for this form of specialization: partition the state space into buckets, and assign an *expert head* to each region, while keeping a shared representation. Common chess-engine bucketing strategies rely on manually designed features such as piece count, king location, or game phase. Those rules are cheap and interpretable, but they are game-specific heuristics and are not necessarily optimal. They encode what a programmer thinks is a distinct regime, not what the model actually needs in order to specialise.

<span style="color: #808080;">[Core Idea]</span> Unsupervised alternatives exist. Clustering hidden-layer activations, for example, groups positions that look similar in the eyes of the network. The weakness of this approach lies in the fact that **similarity of representation is not the same as similarity of *learning signal***. Hidden activations characterize the representation produced by the current model, whereas gradients with respect to the head parameters directly characterize how the loss would change under parameter updates. This motivates defining the partition in terms of the parameter-space directions induced by individual training samples. This thesis is motivated by the idea that **lightweight, data-driven bucketing can provide a practical solution for on-device NNUE evaluation** while **avoiding reliance on chess-specific features**. More generally, such an approach provides a way to mix experts over a state space using information derived from the learning process.

## 1.2 Problem Statement

<span style="color: #808080;">[Problem]</span> The central problem addressed in this thesis is the design of an efficient, *data-driven* method for partitioning the state space of a chess engine into regions of specialist expertise, within the tight *inference-time* resource constraints of embedded devices.

<span style="color: #808080;">[Formal Setup]</span> More specifically, consider a standard NNUE evaluation function composed of a frozen shared representation $W_{L1}$ and a trainable head $(W_{L2}, W_{out})$. Given a dataset of positions $\mathcal{D} = \{(s_i, v_i)\}$, we can train a base model $w_{\text{base}}$. To improve upon this base model via expert specialization, we seek an algorithm that can effectively partition the state space into $B$ buckets $\{\mathcal{D}_1, \dots, \mathcal{D}_B\}$ such that fine-tuning a separate head on each bucket yields a set of specialized models with *distinct parameter updates*. We refer to the parameter difference $\delta_i = \theta_i-\theta_{\mathrm{base}}$ as a _task vector_, following the terminology commonly used for parameter-space representations of model specialization.

<span style="color: #808080;">[Key Difficulty]</span> This objective is complicated by the fact that the partition must be discovered from the training data, yet the criterion for effective specialization is expressed in parameter space through the diversity of task vectors, rather than directly in the input space. At the same time, the resulting routing mechanism must be *lightweight* enough to run on a microcontroller at every node of an alpha-beta search, ruling out expensive computations such as evaluating multiple full networks. Current heuristic bucketing strategies (e.g., based on piece count or king position) are computationally cheap but encode arbitrary *game-specific* assumptions about which positions are similar, which may not align with the learning signal relevant to head specialization.

<span style="color: #808080;">[Research Question]</span> This work, therefore, investigates whether it is possible to learn an unsupervised partition directly from instance-specific gradients—using them as a proxy for the direction in which each head should specialize—and whether such a partition can be deployed via a lightweight dispatcher that respects the memory and computational constraints of an embedded device, without sacrificing the quality of the resulting mixture-of-experts evaluation.


## 1.3 Contributions

<span style="color: #808080;">[Prior Work]</span> Recent work has explored the use of gradient directions to identify groups of samples with related optimization behavior and to construct specialized models. **ELREA** (Li et al., ICLR 2025), for example, partitions training instructions according to their gradient directions to reduce optimization conflicts, while **GradientSpace** (Sridharan et al., 2025) clusters LoRA gradients and uses a lightweight encoder-based router to enable single-expert inference. These works provide evidence that gradient-space structure can be exploited to identify forms of specialization that are not directly defined by the input space.

<span style="color: #808080;">[Domain Gap]</span> However, these approaches target large language models and operate under assumptions that differ substantially from those of embedded chess engines. In our setting, the evaluation function may be invoked at every node of an alpha-beta search, making both the computational cost of routing and the memory footprint of the experts critical constraints. Furthermore, the desired partition should be learned from the training data without relying on manually designed chess-specific features.

<span style="color: #808080;">[This Work]</span> Building on this perspective, we investigate a gradient-informed approach to state-space partitioning for NNUE evaluation. We propose a Mixture-of-Experts architecture in which positions are grouped by clustering their per-sample gradients with respect to the trainable head parameters, $(W_{L2}, W_{out})$. The resulting clusters define candidate regions for expert specialization, while the shared L1 representation is kept common across all experts. At inference time, we introduce a lightweight dispatcher that predicts the bucket assignment from the frozen L1 activations and routes each position to a single specialized head. This separates the computationally expensive discovery of the partition from the lightweight routing required during search.

<span style="color: #808080;">[Final Remark]</span> We evaluate the proposed approach on chess positions labeled with WDL values from a strong engine. The evaluation compares gradient-informed partitioning with single-head and heuristic bucketing baselines, examining both the specialization of the resulting experts and the quality and computational cost of the resulting evaluation function.

## 1.4 Thesis Outline

<span style="color: #808080;">[Next chapters]</span> The next chapter introduces the background and related work underlying the proposed approach. It reviews chess evaluation functions, with particular attention to NNUE architectures, value representations, and the computational constraints of embedded chess engines. It then introduces Mixture-of-Experts architectures, per-sample gradients, and existing approaches to state-space bucketing and gradient-based expert specialization. The discussion concludes by positioning the proposed method with respect to existing work in efficient chess evaluation, teacher-student training, and gradient-based expert specialization.

<span style="color: #808080;">[The chapters after]</span> The following chapter presents the proposed method for unsupervised state-space bucketing via sample gradients. It describes the training of the base NNUE, the computation and clustering of normalized per-sample gradients, the construction of the buckets, and the training of a lightweight dispatcher for inference-time routing. The subsequent chapter describes the experimental setup and implementation, including the dataset and teacher model, network architecture, training procedure, quantization, and integration with the Cfish engine on the target embedded hardware. The experimental results are then presented and discussed, with particular attention to expert specialization, evaluation quality, computational cost, and comparison with the selected baselines. The thesis concludes by summarizing the main findings, discussing the limitations of the approach, and outlining possible directions for future work.

#note summary of the following characters, but without referencing specifica chapter numbers.

---

> **Note for AI**: *The parts marked with a #todo are yet to be completed. The tagged comments are NOT to be exported to the Latex document, and are not meant to be implemented while exporting this document to Latex. The gray labels are for clarity only and must not be transferred to the Latex. Placeholders in square brackets (e.g. `[dataset_size]`, `[W]`) must be replaced with the current values from `_ai-info_.md` when converting this document to Latex.*

[[2 - Background]]
