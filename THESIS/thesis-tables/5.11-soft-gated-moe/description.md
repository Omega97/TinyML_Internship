# Table 5.11 — End-to-end soft-gated MoE

End-to-end soft-gated MoE test cross-entropy (top-1 and top-2 gating), on the 50,000-position subsample. Base CE 0.6303. The best configuration is top-2 with K=2 (0.6304); gate entropy stays near ln K and load variance near 0, so the gate never specialises and the experts remain near-clones of the base head.
