# Next tasks

> Goal (today): produce a **preliminary thesis** where the only remaining final touch is the actual results, plots, Discussion, and Conclusion. Primary focus per `GOAL.md`: **1 - Introduction**, **2 - Background**, **3 - Method**, **4 - Implementation**, and as much of **5 - Results** as possible. Content first (fix all `#todo`s), formatting later.

#note : keep in mind that the [[ai-info]] contains the guidelines 

---

## Cross-cutting (all chapters)

- [ ] Check for cross-chapter duplication and inconsistent terminology (e.g. WDL/Elo/ACPCL wording).
- [ ] Make sure `Hinton et al. (2015)` is bold in the LaTeX (there was a glitch, open `#todo make sure Hinton is bold in LaTeX`).

## 1 - Introduction

- [ ] Signal that we aim to prove NNUE MoE was trained suboptimally (open `#todo` at top).
- [ ] Add `#todo reference to NNUE`, `#todo reference to MoE`, `#todo reference to ELREA`, `#todo reference to GradientSpace`.
- [ ] Rework `1.4` outline to avoid repeating "next/following/subsequent chapter" (open `#todo`).
- [ ] Add the motivation/overview figure (`#todo figure` in `1.1`).

## 2 - Background

- [ ] Separate MCTS (AlphaZero/Lc0) from alpha-beta (Stockfish/Cfish) clearly.
- [ ] Clarify early that we use WDL + soft-CE and why.
- [ ] `2.1.3`: keep conceptual (accumulator, head, incremental update); defer dims / CReLU / 3-way WDL to `4.2`.
- [ ] `2.1.4.1`: add the centipawn caveats (uncapped, not a win probability, "centipawn ≠ 1% of a pawn").
- [ ] `2.1.5`: add one sentence on playing strength reported in Elo/ACPL (protocol in Chapter 5).
- [ ] `2.2.3`: reframe LoRA around what we *keep* rather than considering-then-discarding.
- [ ] `2.2.4`: explain "instruction-following" (open `#todo`).
- [ ] `2.3.1`: clarify distinction between $w$ and $w_{\text{head}}$ (open `#important`).
- [ ] `2.3.2`: simplify/clarify the core point (open "tough read" `#todo`).
- [ ] `2.3.3`: rephrase the `[Learning Signal]` and `[Adapting...]` paragraphs; add TAGCOS coreset-selection coverage.
- [ ] Add figures: NNUE accumulator/head, MoE routing, sample-gradients, bucketing-by-phase, search cost.
- [ ] `2.4.1`: add Kaggle FIDE & Google Efficient Chess AI Challenge info + reference.
- [ ] `2.4.2.2`: add Hexaïssa and M2CTS references.
- [ ] `2.5.1`: fill "learn more about SIMD and AVX".

## 3 - Method

- [ ] Decide placement of definitions vs. sample-gradient intro (open `#todo`).
- [ ] `3.4.5`: finish "How we deal with duplicates" (open `#todo finish`).
- [ ] `3.5.4`: state the final choice of $B$ and clustering algorithm (open `#todo`).
- [ ] `3.5.5`: fix the silhouette formula rendering; decide whether inertia is needed.
- [ ] `3.6`: update hard numbers (params, accuracy); decide `h_own` vs. concatenated dispatcher input; resolve `3.6.5` fixed vs. re-assigned assignments.
- [ ] `3.7`: replace hard numbers (per-expert data), fill learning-rate/schedule, resolve bias question.
- [ ] Cover quantization & pruning (open `#todo`).
- [ ] Cover cosine similarity / spectral clustering (open `#todo`).
- [ ] `3.8`: rephrase the "domain-specific" paragraph.

## 4 - Implementation

- [ ] `4.1`: update hyperparameters and add Lc0 reference.
- [ ] `4.1.1`: align duplicate-position handling with `3.4.5`.
- [ ] `4.1.2`: convert feature list to prose; explain king-plane compression.
- [ ] `4.2.6`: settle $W$/$H$ naming convention.
- [ ] `4.2.7`: update concrete numbers (positions, batch size, epochs).
- [ ] `4.3`: verify `torch.autograd.grad` LaTeX formatting; check the ~100k head-dimension figure; check batch size.
- [ ] `4.4`: note better high-dim clustering alternatives.
- [ ] `4.5`: settle dispatcher input (`h_own` vs. full concat) and note stability.
- [ ] `4.8`/`4.9`: soften/remove explicit Wio-Terminal references to keep the thesis focus on bucketing.
- [ ] `4.9`: explain quantization impact, int32-accumulator error behavior, Cortex-M4 DSP.

## 5 - Results (as much as possible)

- [ ] Regenerate dispatcher plots flagged with `#todo` (legend covering bars; colorbar covering confusion matrix).
- [ ] Confirm all `RESULTS/...` figure paths resolve from the build.
- [ ] Fill or explicitly mark the remaining optional rows: capacity-matched dense baseline, inference NPS benchmark, L1 sparse weights.

## Deferred (final touch, not today)

- [ ] Discussion (Chapter 6) — interpret results, limitations, generalisation, comparison.
- [ ] Conclusion (Chapter 7) — contributions, future work, closing remarks.
- [ ] Abstract key-findings + "hint at the results" + quote attribution.
- [ ] Appendix A — extra tables/plots/implementation notes.
- [ ] Full LaTeX export (placeholders replaced, gray labels stripped, tags removed).
