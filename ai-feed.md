# ai-feed — Discrepancies: RESULTS/ vs `THESIS/markdown/5 - Results.md` (§5.1–5.3)

Scope: only the already-written sections — 5.1 Experimental Setup, 5.2 Clustering, 5.3 Dispatcher. `RESULTS/GOAL.md` is treated as the authoritative description of the current results. Every number below was checked against the actual artifacts (`RESULTS/clustering/*.csv/.json`, `RESULTS/dispatcher/**/stats.csv`, `RESULTS/base/*/source.json`, `RESULTS/manifest.json`).

Legend: 🔴 factual error / must fix · 🟠 methodological or comparability problem · 🟡 minor / cosmetic.

---

## 5.1 Experimental Setup

Resolved — all items below are now fixed and marked verified for later review.

- [x] ✅ **§5.1.4 "dual-hidden FFNN … not yet trained"** — FIXED. Added the three `FFNN2_H{64,128,256}` rows to Table 5.1 (112,451 / 233,091 / 498,947 params; CE 0.656 / 0.645 / 0.632) and reworded the paragraph so only the capacity-matched dense baseline (H=512) remains "not yet trained".
- [x] ✅ **Dataset-size inconsistency across sources** — FIXED. §5.1.2's 150,815,697 is confirmed correct (matches the slice-meta sum and `GOAL.md` Run 1 corpus count). Stale values corrected: `GOAL.md` §1 now reads $150.8\times 10^6$ (was $120\times 10^6$) and `_ai-info_.md` `[dataset_size]` now 150.8M (was 105M; `[dataset_size_unique]` left as a recompute TODO).
- [x] ✅ **§5.1.6 typo "lassification"** — FIXED → "classification".
- [x] ✅ **§5.1.6 stray whitespace "WDL   distribution"** — FIXED → single space.
- [x] ✅ **§5.1.4 incomplete sentence** — FIXED. Now names the dual-hidden FFNN among the dense baselines.

✅ **Verified correct (unchanged):** hardware (DGX Spark / GB10 / 128 GB / CUDA 13.0 / PyTorch 2.12 / Python 3.12 / bf16 + TF32), 429 slices, 844-d = 716 + 128, test_fraction 1% → 1,507,940 test positions, training protocol (Adam 1e-2→1e-3, 100 epochs, 512×10,000), and the 7×10⁶-per-expert budget all match the repo.

---

## 5.2 Clustering

Resolved — all items below are now fixed and marked verified.

- [x] ✅ **"Targeting eight clusters" wording** — FIXED. §5.2.1 now reads "DBSCAN is applied with `min_samples` fixed at $80$ and $\varepsilon$ selected on a quantile grid to recover a small number of clusters (at most eight)", matching `requested_k=8` and no longer implying a hard target.
- [x] ✅ **DBSCAN fit in different spaces** — FIXED (per-representation presentation). §5.2.3 and the Table 5.3 caption now state that board/L1 DBSCAN is fit in a 48-d PCA while gradients are in the native 48-d, and that the rows are "not directly comparable across representations".
- [x] ✅ **"Stability" centroid-cosine numbers** — VERIFIED CORRECT. Re-computed from `clustering/b{2,4,8}/centroids.npy`: mean off-diagonal centroid cosine is −0.938 / −0.307 / −0.128, matching the thesis exactly. The earlier "null in diagnostics.json" concern was a red herring — the values are stored in `centroids.npy`, not `diagnostics.json`.
- [x] ✅ **B=4 centroid separation glossed over** — FIXED. The stability paragraph now states explicitly that the B=4 centroids drift from −0.229 (1M) to −0.307 (2M) while B=2 and B=8 are essentially unchanged, instead of "in line with".
- [x] ✅ **Mixes two experiments (provenance)** — FIXED. The stability paragraph already states "fitted on two million positions" vs the "one-million-position" tables; the DBSCAN and k-means captions now also state their sample sizes explicitly.
- [x] ✅ **Table numbering** — FIXED. Renumbered sequentially 5.1–5.8 (clustering 5.2–5.4, dispatcher 5.5–5.8); all in-text references and captions updated, and the top-of-chapter DeepSeek todo marked resolved.

- [x] ✅ **Silhouette subsample mismatch** — FIXED. Changed `run_clustering_battery.py` to use a 10,000-point subsample for every representation and re-ran the battery (`--skip-embeddings`); board/L1 silhouettes are now on 10k points, matching gradients/DBSCAN/piece-count. New values (vs the old 2k ones): board 0.110 / 0.073 / 0.040 / 0.016 (was 0.112 / 0.075 / 0.041 / 0.019); L1 0.153 / 0.109 / 0.003 / −0.007 (was 0.154 / 0.106 / 0.005 / −0.005). Table 5.2 and its caption updated; the "2,000-point vs 10,000-point" disclosure removed.

✅ **Verified correct (unchanged):** Table 5.2 (all 22 k-means rows: silhouettes, cosine distances, min/max shares), Table 5.3 (DBSCAN clusters 3/2/5, bucket shares, noise 39.6%/57.2%/75.2%, silhouettes), and Table 5.4 (all six ARI/NMI agreement pairs at B=8) match `clustering/stats.csv` / `summary.json` / `overlap.csv` exactly.

---

## 5.3 Dispatcher

Resolved — text-level issues are fixed and marked verified; the plot-regeneration items remain open.

- [x] ✅ **Centroid routers scored against the wrong labels** — FIXED (disclosed). §5.3.3 now states that argmin-cosine routing scored against Euclidean k-means labels is a lower bound on recoverability, and that scoring against spherical (cosine) k-means would raise it. (Re-scoring against spherical k-means is deferred — requires re-running the centroid dispatcher.)
- [x] ✅ **Transductive leakage in dispatcher training** — FIXED (disclosed). §5.3.1 now notes the k-means labels are computed on the full 1M subsample before the 90/10 split, so the reported validation accuracy is transductive rather than a fully independent estimate.
- [x] ✅ **Gradient-target routing reported twice** — FIXED. §5.3.3 now surfaces Run 2's linear gradient dispatcher (2M rows, val acc 0.60/0.56/0.48) in the prose and states it is a separate setup from the 1M MLP sweep (Table 5.7). DeepSeek todo marked resolved.
- [x] ✅ **Routing-error severity metric never computed** — FIXED (removed). The §5.3.2 paragraph and its DeepSeek todo are deleted, since no artifact computes the diagnostic.
- [x] ✅ **§5.3.3 gradient-target range overstated** — FIXED. "0.48–0.49" → "0.47–0.49" (the h=32 B=16 row is 0.475).
- [x] ✅ **§5.3.3 error-structure claim unverified** — FIXED. The unverifiable "diffuse, not adjacent" confusion-matrix claim is removed; the paragraph now keeps only the verified macro-F1 vs top-1 observation. DeepSeek todo removed.

- [ ] 🟡 **Plot regeneration todos still open** — line 220 ("legend covers bars") and line 253 ("colorbar covers confusion matrix"). Requires fixing the plotting code and re-running the dispatcher scripts; not done.

✅ **Verified correct (unchanged):** Table 5.5 (centroid), Table 5.6 (linear/MLP/tree/XGBoost → L1), Table 5.7 (MLP gradient sweep, all h×B), and Table 5.8 (piece-count ARI/NMI) all match their CSVs exactly, including the dummy (0.727→0.172 L1; 0.560→0.134 gradient) and chance baselines, and the §5.3.1 overhead arithmetic (2W·B+B = 2,056 at B=8; <40k params for h=128).

---

## Cross-cutting

🟡 The figure-path-resolution `#todo DeepSeek` note at the top of the chapter is still unresolved (the table-renumbering note was resolved during the §5.2 fix).

🟡 §5.1.7 and §5.2.1 agree that the gradient representation is 48-d (66,563-d head gradient, Gaussian-projected to 48-d — confirmed by `sample_gradients.py`), but double-check that "head parameters (W=128, H=256)" uses the same definition as Chapter 4 (`P_head = 2W·H + H + H·3 + 3 = 66,563`).
