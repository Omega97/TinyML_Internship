# Table 5.2 — Mini-Batch K-Means clustering diagnostics

Mini-Batch K-Means clustering diagnostics for the three representations (board features, L1 activations, sample gradients) and the piece-count reference, on one million training positions. Silhouette is estimated on a 10,000-point subsample; the share columns give the smallest/largest bucket as a fraction of the sample. Gradients are the best-separated (silhouette 0.250 at B=2), L1 collapses as B grows, and piece-count's high silhouette (0.61) is an artifact of its one-dimensional nature.
