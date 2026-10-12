# Table 5.7 — MLP dispatchers predicting gradient buckets

MLP dispatchers (hidden width h) trained to predict the gradient k-means buckets from L1 activations, on one million positions (90/10 split). Recovery is hard: top-1 falls from 0.64 (B=2) to 0.48 (B=16), and ARI stays at or below 0.37. This is the central negative result: L1 activations are a weak proxy for gradient direction.
