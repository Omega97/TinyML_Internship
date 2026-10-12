# Table 5.10 — Dispatcher-routed hard MoE

Dispatcher-routed hard MoE (piece-count, L1-clustered, gradient-clustered) on the 50,000-position test subsample. The reference base attains CE 0.6303 and MAE 0.1326. No routing variant beats the base: piece-count ties it, L1-clustered roughly matches it, and gradient-clustered edges it only at K=8/16 (0.6300) at noise level. 'Dispatcher Top-1' is the MLP dispatcher's accuracy against its own k-means pseudo-labels.
