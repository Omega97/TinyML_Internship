
> My goal from now on is to improve the results of the thesis, and create a scientific paper about the findings.

---

## Plan

The new data (compressed and packaged) is in `data\processed\board_eval_2`.
The thesis is `thesis.md`.

1) Focus on implementing properly the KITSUNE method.
2) Make the paper about the findings.

The main things to correct are the following:

- Train the base model (NNUE with a $H$ neuron L1 and a $H$ neuron L2) on all the data. The loss is the cross-entropy between the *student model* and the *teacher* WDL probabilities.
    
- The KITSUNE method partitions the input space of board representations (L1 activations) into $B'$ mini-clusters. ==🟠Contrary to the thesis tests $B'$ should be a very high number, like 1000==.
    
- Then, the method computes the *average sample gradient* (ASGs) for every mini-cluster, and uses it to *stitch* together similar clusters. ==🟠Not included in the thesis is the fact that this step should ensure cluster balance==. The ASGs are quickly clustered, and the **centroids** are used to compute the actual $B$ partitions (one for each expert) with the following ==🟠balancing procedure==:
	1) Iterate over all centroids many times, and assign the closest unused mini-cluster to that centroid, until $N*\alpha$ mini-clusters were assigned.
	2) Assign the rest of the mini-clusters to the closest centroid. 
	This balances the clusters, while leaving some room for reflecting the structure of the distribution.
	 
- Now we have the $B$ regions of the L1 space, one for each expert. The final step is to fine-tune the experts on their respective partitions.


We want to make sure that we implemented correctly this method. We want to have data about the gradient magnitude in each partition before the fine-tuning step. Then we will show also the difference between the base L1 clustering and KITSUNE.


**Stretch goal**: With 1000 mini-clusters, the dispatcher is the biggest part of the model. Let's try pruning the dispatcher to make it smaller and faster.