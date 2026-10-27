# Thesis Results


## Base models

- ✅ The dual-POV two-hidden NNUE is the best architecture per parameter: the
  reference `W128 H256` model (175k parameters) reaches a test cross-entropy of
  0.6303, within 0.002 of the largest `W256 H512` model (0.6289).
- The biggest base NNUE saturates only around 30 million training positions, but with almost the same performance of 7M positions, where it stops overfitting.

## Clustering

- Sample head gradients form the best-separated and most balanced routing
  representation (silhouette 0.250 at `B=2`, with nearly antipodal centroids).
- L1 activation clusters collapse as the number of buckets grows (silhouette
  turns negative at `B=16`); board features cluster only weakly.
- The gradient partition is nearly orthogonal to the piece-count and activation
  partitions (ARI 0.04–0.08), i.e. it captures a genuinely different structure.

## Dispatchers

- ✅ L1 clusters are near-trivially recoverable by a linear router (top-1 accuracy
  ≈ 0.97–0.99); MLP, XGBoost. 
- 🔴 Predicting gradient clusters from L1 activations is hard (top-1 ≈ 0.48–0.64,
  ARI ≤ 0.37): L1 activations are a weak proxy for gradient direction. The mapping `L1 → gradient bucket` is a bottleneck (not feasible)

## Mixture of Experts - piece-count clustering


## Mixture of Experts - L1 clustering

- ✅ On the reference base: No routing variant beats the strong base (CE 0.6303). Piece-count ties the
  base; L1-clustered MoE roughly matches it; gradient-clustered MoE edges it out
  only at `K=8`/`K=16` (0.6300), at noise level.
- Soft-gated and Switch variants pin their gates near-uniform (entropy ≈ ln K,
  load variance ≈ 0), so the experts never specialise.
- A random dispatcher is worse than the base and degrades with `K`, confirming
  the experts drift away from the generalist base.
- On weaker base: The first reproducible routing gain appears on a weaker base (`W64 H128`,
  CE 0.6537) with longer expert fine-tuning: L1-clustered hard MoE, linear
  dispatcher, `K=4`, ~30 expert epochs on 10M rows, reaches CE 0.6499
  (a −0.0037 gain over its base).
- ✅The Rprop optimizer (sign-only, with step-back on overshoot) proved markedly
  more stable than Adam for the small specialised heads.

## Oracle upper bound

- Routing each position to its minimum-loss expert reveals large headroom:
  cross-entropy falls from 0.6244 (`K=2`) to 0.6005 (`K=16`), and MAE from 0.124
  to 0.088. The experts *can* specialise — the entire gap is a routing gap, not
  a capacity gap.

## World model

- 🔴 A learned, self-supervised world-model accumulator can be trained without
  feedback from a teacher engine or game outcomes, but the unsupervised
  objective reduces evaluation strength, while shifting a substantial part of
  the learning effort to the shared representation.

## New Method

- KITSUNE (two-stage routing: L1 micro-buckets stitched offline into
  gradient-aligned macro-experts) 
- It's similar to L1 bucketing, except the way that buckets are built takes into account sample gradients
- 🆕 The two-stage premise holds at the dispatch layer: a linear router recovers
  the L1 micro-buckets near-perfectly (top-1 ≈ 0.98 at `B′=16` and `B′=32`), so
  the convex micro-buckets are trivially dispatchable as designed.
- 🆕 On the weaker base (`W64 H128`, CE 0.6537), the first two KITSUNE MoE runs
  (`B′ = 4·K`) reach CE 0.6512 (`K=4`, 30M rows) and 0.6510 (`K=8`, 60M rows) — a
  −0.0025/−0.0027 gain over the base, but *not* better than the plain L1-clustered
  hard MoE (0.6499). Gradient-stitched macro-experts have not yet beaten simple
  L1 clustering.
- 🆕 A `K=16` (B′=64) run and a longer `K=4` (90M rows) run are still in progress.