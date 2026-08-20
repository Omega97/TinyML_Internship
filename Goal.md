
# Chess bot project


> *Building a small end efficient chess bot, that can run on extremely limited hardware. The following is a clear description of the steps to complete the project* 

---

## 1 - Building the dataset

- [ ] Download many games from the web (bot games, human games, puzzles, ...).
      
- [ ] Download a strong value function (possibly one that returns the expected reward from any given state, like Lc0).
       
- [ ] Create a JSON fen-value-visits database of all of the games, where you have:
	- the **FEN** code of the position, 
	- the **value** given by the value function, and 
	- the number of **visits** of the position (how many times was found in the dataset).

---

## 2 - Training the model

- [ ] Train a **NNUE** value function $f_w(s)=v$ that is a FFNN with sparse activations and two hidden layers (256 neuron per layer, for example):
	- **input**: the board state, standard representation.
	- **L1**: the first hidden layer is called the "accumulator", and it doubles down as the embedded representation of the board states (param $w^{(1)}$).
	- **L2**: second hidden layer (param $w^{(2)}$).
	- **output**: the expected reward (proba win - proba lose) of the position (param $w^{(3)}$).

---

## 3 - Fine-tuning

- [ ] **Embeddings**: A forward pass of the NNUE on the database will provide the embedding $h_i$ for each position.
     
- [ ] **Task vectors**: Compute the *task vector* $\delta = \nabla_{w^{(head)}} \; \mathcal L_{acc}\,(f_w (s), \hat v)$ for each position, where $w^{(head)} = (w^{(2)}, w^{(3)})$ are the parameters of the expert head. These vectors tell you how the model would like to adapt to learn each position-value pair.
     
- [ ] **Clustering**: Apply a clustering algorithm (like *k-Means*) with $B$ clusters on the normalized task vectors $\hat \delta_i$ to obtain the classification $(\hat \delta, b)$ . Each cluster groups together similar *task vectors*, making it easier for the expert heads to learn them.
     
- [ ] **Dispatcher**: Train a minimal model $g_\phi​(h)=\text{softmax}(W_\phi\, h​)$ to predict the class $b_i$ based on the L1 activations $a^{(1)}$ (the bias component is implied). _The dispatcher is not used during training of the experts — it is only for inference routing._
    
- [ ] **Fine-tuning**: The board positions are clustered based on the model's needs rather than the representation of the states (or the states themselves, for that matter). A final fine-tuning step on each bucket of states, starting from the initial model $f_w$ , will yield a model $f^{(i)}_{w'}$ that should outperform the original model.

---

## 4 - Inference

- [ ] The inference step works as follows:
	1) Run the first half of the model. The activations $a^{(1)}$ of the accumulator layer double down as the representation of the state.
	2) Predict the class of the position based on the first part of the dispatcher $b = \text{argmax}(W_\phi\, h)$
	3) Run the *expert head* selected by the dispatcher to compute the model output $v$.

*Note: in spirit of the NNUE architecture, you are supposed to recycle the activations of the L1 layer when evaluating similar board states. This lets you save compute, as bulk of the computation is then only in the L2 layer.* 

---
