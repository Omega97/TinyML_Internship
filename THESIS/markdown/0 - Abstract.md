
# Unsupervised State Bucketing for Mixture of Experts in Resource-Constrained Chess Engines

---

*"Divide and conquer."*

#note I chose this quote for the thesis because it evokes a tactical situation, not unlike a chess position, and closely resembles the goal of MoE. 

---

# Abstract

<span style="color: #808080;">[Topic / Context]</span> In the context of board games, a common approach for improving model performance is to partition the state space and allocate distinct experts to each region. While this is the central idea behind *Mixture of Experts* (MoE) architectures, the bucketing is often defined manually, requiring domain-specific expertise, and yielding potentially suboptimal partitions.

<span style="color: #808080;">[Contribution]</span> We investigate and compare several approaches to extending the *Efficiently Updatable Neural Networks* (NNUE) paradigm beyond its conventional formulation, with a particular focus on techniques that have been successfully explored in other areas of machine learning but remain relatively uncommon in NNUE-based chess engines. Mixture-of-Experts architectures are a natural extension for chess evaluation that allows multiple specialized heads to share a common NNUE representation. We also explore sample-gradient-based partitioning, using the gradients of individual training samples to identify groups of positions with similar learning requirements, an approach more commonly studied in large-scale language models and other deep learning settings. Finally, in an effort to make the NNUE architecture more applicable to other fields, we experiment with making its conventional accumulator a learned world model of the game.

<span style="color: #808080;">[Key Findings🍀)]</span> The results highlight different trade-offs between the investigated approaches. L1-based clustering combined with a Mixture-of-Experts architecture significantly improves evaluation performance, while introducing only a minor increase in memory usage and inference cost; the lightweight dispatcher accurately reproduces the learned partition. In contrast, sample-gradient-based partitioning is difficult to exploit at inference time, since the gradients depend on the unknown true value of the position, and a lightweight dispatcher struggles to approximate the resulting complex partition. Finally, the world-model approach shows that an NNUE-like architecture can be trained without feedback from a teacher engine or game outcomes, although the unsupervised training objective can reduce evaluation strength. At the same time, learning the world model makes the training process more efficient by shifting a substantial part of the learning effort to the shared representation.

---

## Scraps


<span style="color: #808080;">[Gap / Problem / Research Question]</span> ~~While unsupervised bucketing using hidden layer activations has been explored, this approach may not yield partitions that _maximise expert specialisation_. Activations capture what the model _knows_, not what it _needs_ to adjust. In this work, we ask whether the model's learning dynamics can be used to create a partition that is more effective. Our hypothesis is that clustering gradients (which encode the direction of desired weight updates) will produce buckets that are inherently better suited for expert fine‑tuning, leading to more specialised experts.~~ 

<span style="color: #808080;">[Contribution]</span> ~~We propose a novel unsupervised bucketing method based on sample-gradients — the gradients of the loss with respect to the head parameters of a base model. These vectors encode the direction in weight space that would improve the model's prediction for each individual data point. We apply the method to chess, using a standard NNUE (Efficiently Updatable Neural Network) as the base model trained on [dataset_size] positions labelled with outcome probabilities (WDL) by a strong value function (Lc0).~~ 

<span style="color: #808080;">[Methods]</span> ~~For each position, we compute the normalised sample-gradient with respect to the head parameters, and apply a clustering algorithm to define the buckets. We then train a lightweight linear dispatcher to predict the bucket from the L1 activations, enabling fast, deterministic routing at inference time. Finally, we fine‑tune the expert heads on each bucket, starting from the base model, while keeping the L1 weights frozen. This yields a set of fast, specialised experts, each adapted to a distinct region of the state space, without requiring handcrafted bucketing.~~

---

> Note for the AI: *In this abstract, the §Scraps section and gray text are for clarity only, and must not be transferred to the Latex. Placeholders in square brackets (e.g. `[dataset_size]`) must be replaced with the current values from `_ai-info_.md` when converting this document to Latex.*

[[1 - Introduction]]
