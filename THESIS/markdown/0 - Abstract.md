
# Unsupervised State Bucketing for Mixture of Experts in Resource-Constrained Chess Engines

---

*"Divide and conquer."*

#note I chose this quote for the thesis because it evokes a tactical situation, not unlike a chess position, and closely resembles the goal of MoE. 
#todo who is being quoted btw?

---

# Abstract

<span style="color: #808080;">[Topic / Context]</span> In the context of board games, a common approach for increasing model performance is to partition the state space and allocate distinct experts to each region. While this is the central idea behind *Mixture of Experts* (MoE) architectures, the bucketing is often defined manually, requiring domain-specific expertise, and yielding potentially suboptimal partitions.

<span style="color: #808080;">[Gap / Problem / Research Question]</span> While unsupervised bucketing using hidden layer activations has been explored, this approach may not yield partitions that _maximise expert specialisation_. Activations capture what the model _knows_, not what it _needs_ to adjust. In this work, we ask whether the model's learning dynamics can be used to create a partition that is more effective. Our hypothesis is that clustering gradients (which encode the direction of desired weight updates) will produce buckets that are inherently better suited for expert fine‑tuning, leading to more specialised experts. 

<span style="color: #808080;">[Contribution]</span> We propose a novel unsupervised bucketing method based on sample-gradients — the gradients of the loss with respect to the head parameters of a base model. These vectors encode the direction in weight space that would improve the model's prediction for each individual data point. We apply the method to chess, using a standard NNUE (Efficiently Updatable Neural Network) as the base model trained on [dataset_size] positions labelled with outcome probabilities (WDL) by a strong value function (Lc0). 

<span style="color: #808080;">[Methods]</span> For each position, we compute the normalised sample-gradient with respect to the head parameters, and apply a clustering algorithm to define the buckets. We then train a lightweight linear dispatcher to predict the bucket from the L1 activations, enabling fast, deterministic routing at inference time. Finally, we fine‑tune the expert heads on each bucket, starting from the base model, while keeping the L1 weights frozen. This yields a set of fast, specialised experts, each adapted to a distinct region of the state space, without requiring handcrafted bucketing.

#todo change the point to "we compare many methods AND sample gradients are one of them"

#todo summarize a bit 

<span style="color: #808080;">[Key Findings🍀)]</span> #todo hint at the results 

---

> Note for the AI: *In this abstract, the §Scraps section and gray text are for clarity only, and must not be transferred to the Latex. Placeholders in square brackets (e.g. `[dataset_size]`) must be replaced with the current values from `_ai-info_.md` when converting this document to Latex.*

[[1 - Introduction]]
