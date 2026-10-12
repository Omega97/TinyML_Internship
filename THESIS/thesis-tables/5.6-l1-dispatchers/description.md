# Table 5.6 — Learned L1-to-L1 dispatchers

Learned L1-to-L1 dispatchers on one million positions (90/10 split), scored against the L1 Mini-Batch k-means labels. Recovery of the L1 partition is essentially trivial for the neural routers: the linear dispatcher reaches 0.993 (B=2) to 0.967 (B=16), statistically tied with the MLP; XGBoost and the decision tree are progressively weaker.
