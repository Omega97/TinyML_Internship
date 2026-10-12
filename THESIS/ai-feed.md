# AI Feed — Alignment check: thesis `markdown/` vs `notes/REPORT-9-oct.md`

## Verdict

The thesis chapters are **numerically aligned** with `REPORT-9-oct.md` on every
reported table (base models, dense baselines, clustering, dispatchers, and the
MoE batteries on both bases). Two real gaps remain, plus a couple of minor
narrative/consistency issues:

1. The **world-model results** in REPORT §5 are not reported anywhere in
   Chapter 5.
2. `W64_H128` has two different CE values (0.6725 vs 0.6537) that are never
   reconciled in one place.
3. The Abstract overstates the L1-clustered MoE gain.

---

## Aligned ✅ (numbers match the report)

- **Base-model sweep** (Table 5.1) = REPORT §0:
  `W32_H64` 0.6555, `W64_H128` 0.6725, `W128_H128` 0.6387,
  **`W128_H256` (reference) 0.6303**, `W256_H512` 0.6289.
- **Dense baselines** (Table 5.9) = REPORT §0:
  Linear 0.766; FFNN 0.669 / 0.654 / 0.643; FFNN2 0.656 / 0.645 / 0.632.
- **Clustering** (Table 5.2) = REPORT §1:
  gradients best-separated (silhouette 0.250 / 0.196 / 0.191 / 0.134), L1
  collapses at K=16, board weak, piece-count 0.606.
- **Dispatchers** (Tables 5.5–5.7) = REPORT §2:
  L1→L1 linear 0.993 → 0.967 (near-trivial); L1→gradient ≈ 0.64 → 0.48
  (ARI ≤ 0.37).
- **MoE, reference base** (Tables 5.10–5.14) = REPORT §3:
  no router beats base 0.6303; oracle 0.6244 → 0.6005; soft-gated/Switch pin
  the gate near-uniform; random router degrades.
- **MoE, weaker base** (Table 5.15) = REPORT §4:
  `l1_k4_linear_ep30_10M_100k` → 0.6499 (−0.0037); rprop / K=16 add little.
- **KITSUNE** (5.6) = REPORT "follow-up": results still pending in both.
- **Capacity-matched dense baseline, inference NPS, L1 sparse weights**:
  pending in both (thesis 5.1.4; REPORT §7).

---

## Misaligned / gaps ⚠️

1. **World-model results are absent from Chapter 5.**
   REPORT §5 has concrete (negative) numbers — InfoNCE/VICReg bases CE ~0.97–1.06,
   loaded world-model L1 K=8 collapses to CE 1.47. Chapter 5 has no world-model
   section; the only world-model text lives in the Abstract
   (`0 - Abstract.md:18`) and Introduction (`1 - Introduction.md:24`). The
   Abstract already states a world-model "Key Finding", so the supporting numbers
   must be added to Chapter 5 to back it up.

2. **`W64_H128` CE has two unreconciled values.**
   REPORT gives 0.6725 (manifest) and 0.6537 (2nd-battery test). The thesis uses
   0.6725 in Table 5.1 (dismissed as an "outlier" in 5.4.3) but 0.6537 as the
   second-battery base in 5.5.4. The two are never tied together in one place.

3. **Abstract overstates the L1-clustered result.**
   `0 - Abstract.md:18` says L1-clustered MoE "significantly improves evaluation
   performance". The actual result (REPORT §3–4 and 5.5.3) is that L1-clustered
   MoE only *matches* the reference base and gains just −0.0037 on the weaker
   base. Recommend softening to "matches / slightly improves on a weaker base".

---

## Minor notes

- **7M vs 2M budget.** Thesis 5.1.5 mentions a 7M-per-expert budget while 5.5.1
  uses the 2M preliminary pack — consistent with REPORT §3's "2M-row battery",
  but the 7M target is thesis-only (not present in the report).
- **REPORT §7 "Run 12" (7M/head gradient K=8, stalled dispatcher ~0.54)** is an
  internal detail and correctly omitted from the thesis.
