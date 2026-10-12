# Table 5.13 — Oracle upper bound

Oracle upper bound on the gradient-clustered experts, on the 50,000-position subsample. Routing each position to its true argmin-loss expert reveals large monotonic headroom: oracle CE falls from 0.6244 (K=2) to 0.6005 (K=16) and MAE from 0.124 to 0.088. 'Nearest-centroid CE' uses the cosine/nearest-centroid rule; the realised dispatcher sits near 0.63. The gap is a routing gap, not a capacity gap.
