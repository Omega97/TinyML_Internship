# Table 5.3 — DBSCAN results

DBSCAN results, fitted on a 100,000-point subsample with min_samples=80 and epsilon selected on a quantile grid. The board and L1 runs are fit in a 48-dimensional PCA of the data, the gradient run in the native 48 dimensions, so the rows are not directly comparable across representations. DBSCAN recovers few clusters and flags a large fraction of the sample as noise (up to 75.2% for gradients).
