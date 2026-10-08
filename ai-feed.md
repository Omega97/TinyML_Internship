# ai-feed — State of the results (MoE for chess)

**Goal:** implement a variety of MoE approaches for chess position evaluation, and export the results as **plots + numeric tables**. Reference base = dual-POV two-hidden NNUE `W128_H256` (175k params, test CE **0.6303**).

## Where we are

The full MoE battery is implemented and logged in `GOAL - Results.md` (Runs 1–11, all green) plus an in-progress Run 12. Ten approaches exist. The **committed** battery (2M-row, 2–5-epoch cells on the reference base) found **no routing gain** — every router lands within ±0.0003 of base CE. The oracle upper bound, however, shows large recoverable headroom, so the bottleneck is routing, not expert capacity.

### Committed results (reference base CE 0.6303, 2M rows)

| Approach | MoE CE | vs base | Notes |
|---|---|---|---|
| Piece-count routed (K=8) | 0.6303 | = | per-bucket expert CE ≈ base on every bucket |
| L1-clustered hard (K=2/4/8/16) | 0.6309/0.6305/0.6303/0.6300 | ≈ | dispatcher Top-1 0.98–1.00, no signal |
| Gradient-clustered hard (K=2/4/8/16) | 0.6307/0.6306/0.6300/0.6301 | ≈ | L1→gradient proxy weak (Top-1 0.64→0.48) |
| Soft-gated end-to-end (top-1/2) | 0.6304–0.6323 | ≈/worse | gate pinned near-uniform (entropy ≈ ln K) |
| Switch (sparse top-1, α 0.001–0.05) | 0.6307–0.6326 | worse | no CE gradient on the hard argmax gate |
| Random dispatcher control | 0.6311–0.6429 | worse | degrades with K → experts genuinely drift |
| **Oracle (argmin-k loss)** | **0.6244/0.6160/0.6067/0.6005** | **big** | monotonic headroom the routers never reach |

### New (uncommitted) runs — first real routing gain

A second battery of ~25 runs (`RESULTS/moe/load_dual_h128_H128_512x8198_e1000_*`, Oct 1–2) loads the **smaller W64_H128 checkpoint** (measured base CE **0.6537** on 100k test) and uses longer expert fine-tuning (up to 30 epochs). These **do** beat their base:

- L1-clustered hard MoE, linear dispatcher, 30 expert epochs, 10M rows: **K=2 CE 0.6505**, **K=4 CE 0.6499** (base 0.6537); K=4 oracle 0.6174.
- Switch K=2, 40 epochs, 10M: **CE 0.6509** vs base 0.6537.
- Piece-count K=8 sweeps (1M→80M rows, Adam/SGD, 1–30 ep): mostly flat; best ~0.6516–0.6523, several degrade below base.

On the **reference** base (`load_dual_h128_H256_e200_bpe512_bs10000_*`) Switch still does not beat it (K=2 0.63036 vs 0.63026; K=4 0.63117; K=8 0.63204). So the gain so far is tied to a weaker base + longer fine-tuning.

## Gaps against the goal (plots & tables)

- **New runs export only `summary.json`/`eval.json`** — no `plots/` and no PNG (`expert_ce_by_bucket`, `moe_vs_base_ce`), and no consolidated table (`ce.csv`-style) for the second battery. This is the main unfinished deliverable.
- **Run 12 (well-resourced gradient K=8, 7M/head) is stalled**: 56M-row gradient cache, 8-cluster k-means, and the 57 GB L1 tensor are built (`RESULTS/moe/wellresourced_k8/`), but the dispatcher stopped at epoch 020 (val_acc ~0.54) — no `moe.pt`, no `eval.json`, no summary.
- Still 🔴 in `GOAL - Results.md`: capacity-matched dense baseline, inference NPS benchmark, L1 sparse weights.
- `scripts/moe-training-UI/` is the live training driver (Switch technique, SGD expert optimizer, settings persistence — uncommitted). It feeds the `load_*` run names above.
