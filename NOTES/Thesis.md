
> *Relatore: Alessio Ansuini* 

---
 
## Idee con Ansuini

"Task vectors (in weight space / representation space) T1, T2, …, T8 have specific vectors in the weight space. Is it possible / useful to define task vectors that bring you from one head to another with minimal information/performance loss? This may be the case if the head vectors live in a linear submanifold of low dimensionality (are we so lucky? Maybe not!)"

[Task vectors](https://arxiv.org/abs/2212.04089), "distance between layers" in parameter space (permutations!), [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)


## Finding Optimal Feature Combinations for Maximally-Diverse Experts

Basically, what I will try next is the following: training of a single NNUE (L1 + L2 + Output layer), then fine-tune the L2 and Output layers for every head, but with a twist; find a set of features of the board states (number of pieces, queen or no queen, bishop pair, rook pair, king position, ... `-> (30/32, 1, 0, 1, 5/8, 1/8, ...)` etc...). We re-do the fine-tuning on various combinations of features to obtain various **normalized delta vectors** (after minus before fine-tuning). We pick the sets of partitions

<div align="center">
    <img src="images/arrows.png" width="300">
</div>

Be careful: tiling of the position space must be complete and with no overlaps (like a function $g : s_{board} \rightarrow i_{bucket}$) , and the bucket must probably be of (more or less) uniform size.

bucket i = set of all board states in the database with index $g(s)==i$

expert model i trained on bucket i (**one sweep**) produces a delta vector. 

Score for f = some sort of average angular distance (**cosine similarity**) between delta vectors, but very low if the smallest distance is small.


Note: the deltas should point in all directions by default. If they don't, the training was probably not complete.

---

**==Important idea for the thesis== (Omar + Ansuini):** freeze shared L1 from a single base value function model $\hat f$; for each candidate partition $g : s_{\mathrm{board}} \mapsto i_{\mathrm{bucket}}$ , fine-tune only L2 + output **one sweep** per bucket and record normalized **delta (task) vectors** $\delta_i = \theta_i - \theta_{\mathrm{base}}$ (fine-tuned layers only). Score $f$ by how **diverse** the $\delta_i$ are (mean angular distance / cosine, with a penalty if any pair is nearly collinear). Partition must be **complete and non-overlapping**, buckets roughly uniform in sample count.

==Idea==: Alternatively, perform one sweep of the dataset of $(s_i, v_i)$ pairs to compute (once) the deltas $\delta_i = \nabla_w \; \mathcal L_{acc}(\hat f(s_{base}), v_i)$ , defined as the gradient of the loss wrt the value function weights. Then, train a dispatcher $g(s_{board})$ (FFNN, no hidden layer, softmax activation) to partition the dataset so to maximize the average intra-cluster distance $S$ (minus the average within-cluster distance). The point is to have a simple dispatcher that is optimized to group board positions in maximally-diverse clusters. 
The probabilistic nature of the softmax makes the training possible, but during inference we get rid of that block, and only pick the highest-value activation to assign the class (same result but faster).

**Candidate features for $g$:** the entire board, piece count, king location, queen presence, bishop/rook pair, material imbalance, pawn structure, mobility, etc. Search over combinations (greedy / random / GA) rather than exhaustive enumeration.

---

Base model $\hat{f}_{\text{base}} = (W_{L1}, W_{L2}, W_{\text{out}})$ 
**Freeze L1 weights** $W_{L1}$ (shared representation fixed).

distance $d(\delta_i, \delta_j)$

number of buckets/experts $B$ 

**Bucket function** $g : \mathcal{S} \rightarrow \{1, 2, \dots, B\}$ where:

Diversity metric $S(\mathcal{D})$, for example intra-cluster distance between centroid $c_i$
$$S(\mathcal{D}) = \frac{1}{\binom{B}{2}} \sum_{i<j} d(c_i, c_j)$$

---
---

# Tesi di Laurea: Task Vectors per l'Apprendimento Multi-Expert in Reti NNUE


## 1. Idea Centrale

Partendo da un **modello base** NNUE addestrato su tutte le posizioni, proponiamo di creare un insieme di **esperti specializzati**, dove ogni esperto viene addestrato su un sottoinsieme (bucket) di posizioni. La sfida è trovare la **partizione ottimale** dello spazio degli stati tale che gli esperti risultanti siano massimamente **diversi** e complementari.

L'ipotesi chiave è che i **task vectors** (vettori nello spazio dei pesi che rappresentano il cambiamento indotto dal fine-tuning su un bucket) possano essere usati come segnale guida per progettare la partizione. Bucket che producono task vectors **ortogonali** tra loro genereranno esperti con specializzazioni complementari.

---

## 2. Definizioni e Notazione

- $\mathcal{S}$: spazio di tutte le posizioni scacchistiche.
- $B$: numero di bucket/esperti.
- $g: \mathcal{S} \rightarrow \{1, \dots, B\}$: funzione di partizione.
- $\theta_{\text{base}}$: pesi del modello base, addestrato su tutto il dataset.
- $\theta_i$: pesi dell'esperto $i$, ottenuto fine-tuning di $W_{L2}$ e $W_{out}$ (con $W_{L1}$ **congelato**) sul bucket $i$ per **un singolo sweep**.
- $\delta_i = \theta_i - \theta_{\text{base}}$: task vector per l'esperto $i$.

---

## 3. Metrica di Diversità

Definiamo la metrica $S(\mathcal{D})$ che quantifica la diversità dei task vectors indotti da una partizione:

$$S(\mathcal{D}) = \frac{1}{\binom{B}{2}} \sum_{i=1}^{B} \sum_{j=i+1}^{B} d(\delta_i, \delta_j)$$

dove $d(\delta_i, \delta_j)$ è la **distanza del coseno**:

$$d_{\text{cos}}(\delta_i, \delta_j) = 1 - \frac{\delta_i \cdot \delta_j}{\|\delta_i\| \|\delta_j\|}$$

L'obiettivo è **massimizzare $S(\mathcal{D})$**, ovvero trovare una partizione che produca task vectors il più possibile ortogonali.

---

## 4. Due Approcci per la Ricerca della Partizione Ottimale

### Approccio A: Ricerca Euristica su Feature Semplici
- Definire un insieme di **feature candidate** per $g$: materiale, presenza di donne, coppia di alfieri/torri, posizione del re, struttura pedonale, mobilità, etc.
- Per ogni combinazione di feature, costruire una griglia (discretizzazione) e valutare $S(\mathcal{D})$.
- Selezionare la partizione con $S$ massimo (ricerca greedy/genetica/randomizzata).

### Approccio B: Apprendimento del Dispatcher (End-to-End)
- Addestrare una rete leggera $g_\phi(s)$ (MLP con softmax) per **apprendere la partizione** ottimale.
- Il gradiente di $\phi$ viene stimato per massimizzare $S(\mathcal{D})$ (ottimizzazione bilevel con tecniche tipo REINFORCE o Gumbel-Softmax).
- **Inferenza**: si usa $g_\phi$ in modalità **argmax**, scartando la softmax per velocità.

---

## 5. Metodologia Sperimentale

1. **Addestramento Base**: Addestrare $\theta_{\text{base}}$ su un grande dataset di posizioni.
2. **Generazione Task Vectors**: Per una data partizione, per ogni bucket:
   - Congelare $W_{L1}$.
   - Fine-tuning di $W_{L2}$ e $W_{out}$ per un singolo sweep.
   - Registrare $\delta_i = \theta_i - \theta_{\text{base}}$.
3. **Ottimizzazione**: Applicare approccio A o B per trovare la partizione che massimizza $S(\mathcal{D})$.
4. **Validazione**: Valutare le performance degli esperti su dataset di hold-out e confrontare con il modello base.

---

## 6. Risultati Attesi e Contributi

1. Una nuova metrica basata su task vectors per quantificare la diversità tra esperti.
2. Due metodologie (euristica e appresa) per ottimizzare la partizione dello spazio degli stati.
3. Evidenza empirica sulla struttura lineare dei task vectors in NNUE.
4. Un modello multi-expert che supera le performance del singolo modello base.

---

## 7. Collegamento con l'Idea Iniziale (World Models / TinyML)

Il principio alla base di questo lavoro — **aggiornamento incrementale e diversificazione delle rappresentazioni** — è lo stesso che abbiamo identificato come potenziale ponte tra NNUE e World Models. Questa tesi potrebbe quindi servire come **caso di studio** per una metodologia più ampia, applicabile anche all'ottimizzazione di World Models per deploy su microcontrollori.

---
