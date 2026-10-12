# REPORT — High-level results (MoE for chess)

This report distils the results under `RESULTS/` into a single narrative and
answers one question: **which routing algorithms and parameters actually worked,
and why?** All numbers are cross-entropy (CE, lower is better) on the test split;
MAE is reported where it matters. "Base" is always a dual-POV two-hidden NNUE
(`844 → W`, dual accumulator, `2W → H`, CReLU, `H → 3` WDL softmax).

---

## TL;DR — what worked

1. **On the strong reference base (`W128_H256`, 175k params, CE 0.6303), no MoE
   router beats the base.** Every committed variant lands within ±0.0003 of base
   CE; soft-gated and Switch are strictly worse; a random dispatcher is much
   worse. The *oracle* upper bound (CE 0.600–0.624) proves the experts *can*
   specialize — the bottleneck is routing, not capacity.

2. **The first real, reproducible routing gain comes on a *weaker* base
   (`W64_H128`, CE 0.6537) with longer expert fine-tuning.**
   The best cell is **L1-clustered hard MoE, linear dispatcher, K=4, ~30 expert
   epochs, 10M rows → CE 0.6499** (a −0.0037 gain over its base).

3. **The dispatcher architecture barely matters.** For L1 clusters, a *linear*
   router (Top-1 ≈ 0.97–0.99) is essentially as good as an MLP or XGBoost. The
   real limit is *what* you cluster on, not the router capacity.

4. **L1 activations are a poor proxy for gradient direction** (Top-1 ≈ 0.48–0.64
   when predicting gradient clusters). This is the core finding behind the
   KITSUNE two-stage design: gradient clusters carry the specialization signal
   but are not linearly separable from L1.

---

## 0. Reference points — base models (`RESULTS/base/`, `RESULTS/manifest.json`)

Dual-POV two-hidden NNUE scales cleanly with size:

| Model | Params | Test CE |
|---|---|---|
| W32_H64 | 31k | 0.6555 |
| W64_H128 | 71k | 0.6725 (manifest) / 0.6537 (2nd-battery test) |
| W128_H128 | 141k | 0.6387 |
| **W128_H256** (reference) | 175k | **0.6303** |
| W256_H512 | 481k | 0.6289 |

Dense baselines confirm the NNUE inductive bias (a flat FFNN over concatenated
features needs far more params to approach it):

| Model | Params | Test CE |
|---|---|---|
| Linear | 5k | 0.766 |
| FFNN H64/H128/H256 | 108k/217k/433k | 0.669/0.654/0.643 |
| FFNN2 H64/H128/H256 | 112k/233k/499k | 0.656/0.645/0.632 |

Two bases are used as MoE hosts: the **reference `W128_H256` (CE 0.6303)** and a
**smaller `W64_H128` (CE 0.6537)** for the second battery (§3).

---

## 1. Clustering (`RESULTS/clustering/`)

Three representations on the same 1M-row subsample, k-means `K ∈ {2,4,8,16}` plus
DBSCAN, and a handcrafted piece-count partition.

**Silhouette (higher = cleaner):**

| Representation | dim | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| Board (raw features) | 1688 | 0.110 | 0.073 | 0.040 | 0.016 |
| L1 activations | 256 | 0.153 | 0.109 | 0.003 | −0.007 |
| **Gradients** | 48 | **0.250** | **0.196** | 0.191 | 0.134 |
| Piece count | 1 | — | — | — | 0.606 |

- **Gradient space is the cleanest, most balanced signal** and stays healthy as
  `K` grows; L1 collapses at `K=16`; board features cluster weakly.
- **Cross-representation overlap is low** (ARI ≤ 0.19 for every pair) — board,
  L1, and gradient partitions capture disjoint structure. The piece-count
  partition is trivially separable yet aligns with *nothing* learned
  (ARI ≤ 0.19 vs board, ≤ 0.04 vs gradients).

**Verdict:** gradients are the most *informative* routing signal, but they are
the least *dispatchable* (§2) — the whole story of this project.

---

## 2. Dispatchers (`RESULTS/dispatcher/`)

How well can a router recover each cluster assignment?

| Target buckets | Router | Top-1 (k=2 → k=16) |
|---|---|---|
| L1 k-means | Linear | 0.993 → 0.967 |
| L1 k-means | MLP h64 | 0.994 → 0.971 |
| L1 k-means | XGBoost | 0.986 → 0.891 |
| L1 k-means | Decision tree | 0.959 → 0.704 |
| L1 k-means | Cosine-centroid (no training) | 0.979 → 0.940 |
| **Gradient k-means** | **Linear/MLP (input = L1)** | **0.64 → 0.48** |

- **L1 → L1-cluster is near-trivial** (Top-1 ≈ 0.97–0.99): linear, MLP and XGBoost
  are essentially tied, and hidden width (32/64/128) makes no difference.
- **L1 → gradient-cluster is hard** (Top-1 ≈ 0.48–0.64, ARI ≤ 0.37): *L1
  activations are a weak proxy for gradient direction*. This is the empirical
  basis for the KITSUNE argument that a single linear stage cannot realise
  gradient-aware routing.

**Verdict:** routing capacity is *not* the bottleneck — a linear dispatcher is
enough for whatever is linearly separable. The bottleneck is the mapping
`L1 → gradient-bucket`.

---

## 3. MoE — reference base `W128_H256` (CE 0.6303)

The committed 2M-row, 2–5-epoch battery found **no routing gain**. Base CE is
0.6303 throughout; oracle is the true `argmin_k loss_k` upper bound.

| Approach | K | MoE CE | Oracle CE |
|---|---|---|---|
| Gradient-clustered hard MoE | 2/4/8/16 | 0.6307/0.6306/0.6300/0.6301 | 0.6244/0.6160/0.6067/0.6005 |
| L1-clustered hard MoE | 2/4/8/16 | 0.6309/0.6305/0.6303/0.6300 | — |
| Piece-count routed | 8 | 0.6303 | — |
| Soft-gated (top-1 / top-2) | 2→16 | 0.6307→0.6323 / 0.6304→0.6310 | — |
| Switch (top-1, α 0.001–0.05) | 2→16 | 0.6307→0.6326 | — |
| Random dispatcher (control) | 2→16 | 0.631–0.643 | — |

Key observations:

- Only `K=8/16` gradient-clustered and `K=16` L1-clustered *edge* the base by
  ~0.0002 — noise-level.
- **The oracle improves monotonically with `K`** (0.624 → 0.600), so more experts
  *would* help if routing were perfect — it isn't.
- Soft-gated and Switch pin the gate near-uniform (entropy ≈ ln K, load
  variance ≈ 0), so experts never specialise; `α` barely matters.
- Random routing degrades with `K` (0.631 → 0.643), proving the experts drift
  from the base and that the *trained dispatcher* — not extra capacity — is what
  buys the MoE back to ≈ base.

**Verdict:** on a well-trained base there is nothing left to exploit. The
failure mode is routing, confirmed by the large oracle gap.

---

## 4. MoE — smaller base `W64_H128` (CE 0.6537) — the first real gains

A second battery of ~25 runs loads the smaller `W64_H128` checkpoint and uses
**longer expert fine-tuning** (up to 30–100 epochs) on 5M–100M rows. These runs
**do beat their base**. Sorted by MoE CE:

| Run (L1-clustered hard MoE) | K | MoE CE | vs base 0.6537 |
|---|---|---|---|
| `l1_k4_linear_ep30_10M_100k` | 4 | **0.6499** | **−0.0037** |
| `l1_k16_linear_ep100_rprop_60M` | 16 | 0.6502 | −0.0035 |
| `l1_k4_linear_ep30_5M` | 4 | 0.6505 | −0.0032 |
| `l1_k2_linear_ep30_10M` | 2 | 0.6505 | −0.0032 |
| `l1_k8_linear_ep50_rprop_25M` | 8 | 0.6512 | −0.0025 |
| `l1_k8_linear_ep100_rprop_50M` | 8 | 0.6515 | −0.0022 |

Other families on the same base:

- Switch K=2, 40 epochs, 10M: CE 0.6509 (−0.0028).
- Piece-count K=8 sweeps (1M→80M, Adam/SGD, 1–30 ep): mostly flat, best
  ~0.6516–0.6523, several *below* base.

**Verdict:** the gain is concentrated in **L1-clustered hard MoE with a linear
dispatcher**. `K=4` with ~30 expert epochs and ~10M rows is the sweet spot; going
to `K=16` or rprop/100 epochs buys almost nothing extra. Crucially, this only
materialises on a *weaker* base — the same recipe on `W128_H256` stays at parity
(§3).

---

## 5. World-model / other (`RESULTS/moe/wm_*`, `world-model-how-to.md`)

The self-supervised L1 (world-model) experiments are exploratory and largely
negative so far: InfoNCE/VICReg bases land at CE ~0.97–1.06 and a loaded
world-model L1 K=8 run collapses to CE 1.47. These are not competitive with the
standard CReLU accumulator and are not part of the "what worked" conclusion.

---

## 6. What worked best — bottom line

| Question | Answer |
|---|---|
| Best clustering space for *routing* | **L1 activations** (only thing a cheap router can recover at ~99%) |
| Best clustering space for *signal* | **Head gradients** (cleanest clusters, biggest oracle headroom) |
| Best dispatcher | **Linear** (parity with MLP/XGB for L1 buckets; width irrelevant) |
| Best MoE recipe that beats a base | **L1-clustered hard MoE, linear dispatcher, K=4, ~30 expert epochs, 10M rows, frozen L1** |
| On the reference base | Nothing beats it; oracle headroom (0.600–0.624) is unreachable via L1 routing |

The single most important parameter insight: **longer expert fine-tuning +
a weaker base** unlock the gains, not router expressiveness. The KITSUNE thesis
is the natural follow-up — decompose `L1 → (many convex micro-buckets) →
gradient-recombined macro-experts` so a linear dispatcher plus a static lookup
table can express non-convex gradient clusters that a single-stage L1 router
cannot.

---

## 7. Known gaps

- New `load_*` runs export only `summary.json`/`eval.json` — no `plots/`, no PNG,
  and no consolidated `ce.csv` table for the second battery.
- **Run 12 (well-resourced gradient K=8, 7M/head)** is stalled at the dispatcher
  (val_acc ~0.54); no `moe.pt`/`eval.json`.
- Capacity-matched dense baseline, inference NPS benchmark, and L1 sparse-weight
  export are still 🔴 in `GOAL - Results.md`.
