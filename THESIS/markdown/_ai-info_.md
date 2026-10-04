
> *Note: This folder contains the text of the Master's thesis in chess AI. This and all the subsequent documents are meant to be transposed to LaTex in pdf form.*

[[related-work]]

---

## Notes for the AI agents

- professional, scientific tone
- mind the sources!
- no lists or bullet points unless absolutely necessary
- watch out for duplicate information and redundancy - do not state the same exact thing in two different chapters
- not too many "we", "our", and "us" - sometimes you can just refer to the algorithm/method instead
- not too many em dashes (option: en dashes among spaces)
- watch out for formula glitches ("BB" instead of "$B$", "\(a+b\)" instead of "$a+b$")
- check for inconsistencies: chapters in the wrong place, wrong summary title
- check "thesis-writer" skill if available
- ~~strikethrough~~ text is not meant to be exported to the LaTex file
- <span style="color: #808080;">[Each paragraph must have a clear point]</span>, marked in brackets in gray at the beginning like this
- **bald** and *italic* are meant to be transposed to the LaTex, unless stated otherwise

---

## Hyperparameters

> *Note for the AI: refer to this file for every hyperparameter and variable. When converting markdown to LaTeX, replace every square-bracket placeholder (e.g. `[dataset_size]`, `[W]`) with the **current value** from the table below. Do not leave placeholders in the LaTeX.*

| Placeholder             | Meaning                                                       | Current value                                       |
| :---------------------- | :------------------------------------------------------------ | :-------------------------------------------------- |
| `[dataset_size]`        | Dataset size (FEN positions, with duplicates)                 | 150M                                                |
| `[dataset_size_unique]` | Dataset size (unique FEN positions)                           | #todo recompute                                     |
| `[test_fraction]`       | Percentage of the dataset held out as test set                | 1%                                                  |
| `[batch_size]`          | Training batch size                                           | 10k                                                 |
| `[lr_start]`            | Initial learning rate                                         | $10^{-2}$                                           |
| `[lr_end]`              | Final (decayed) learning rate                                 | $10^{-3}$                                           |
| `[d_in]`                | Input / sparse-feature dimension                              | 844                                                 |
| `[W]`                   | L1 accumulator width **per perspective** (dual NNUE: `[W]`×2) | 128                                                 |
| `[H]`                   | L2 hidden width                                               | 256                                                 |
| `[model_size]`          | total number of model parameters                              | 70k                                                 |
| `[keep_prob]`           | Probability of keeping any sampled position in the dataset    | 5%                                                  |
| `[n_params]`            | Total trainable parameters of the target architecture         | *derive from `[d_in]`, `[W]`, `[H]` at export time* |
| `[P_head]`              | Flattened head-parameter dimension (L2 + output)              | *derive from `[W]`, `[H]` at export time*           |

Target model architecture: $d_{\text{in}}=[d_in]$, dual NNUE, L1 accumulator layer $[W]\times 2$ neurons, L2 $[H]$ neurons, output WDL for the current player.

---

## Formatting Citations

La guida ufficiale UniTS stabilisce che:

- la bibliografia deve contenere **tutte e sole le pubblicazioni utilizzate e citate** nella tesi;
    
- le citazioni bibliografiche complete vanno riportate alla fine;
    
- le fonti vanno ordinate **alfabeticamente per autore**;
    
- nel testo si usa il sistema **autore–anno**, non numerico `[1]`, `[2]`, ecc. ([Corsi UniTS](https://corsi.units.it/sites/default/files/media/documents/Guida%20per%20la%20redazione%20tesi_7-2024.pdf?utm_source=chatgpt.com "Università degli Studi di Trieste"))
    

Gli esempi ufficiali sono:

|Caso|Citazione nel testo|
|---|---|
|1 autore|`(Rossi, 2024)`|
|2 autori|`(Rossi & Bianchi, 2024)`|
|≥3 autori|`(Rossi et al., 2024)`|
|più studi|`(Rossi, 2023; Rossi & Bianchi, 2024; Verdi et al., 2024)`|
|stesso cognome|`(Rossi E., 2023; Rossi R., 2024)`|
|stesso autore, anni diversi|`(Rossi 2023, 2024)`|

([Corsi UniTS](https://corsi.units.it/sites/default/files/media/documents/Guida%20per%20la%20redazione%20tesi_7-2024.pdf?utm_source=chatgpt.com "Università degli Studi di Trieste"))

Quindi, **per la tua tesi DSAI, il formato ufficialmente indicato è sostanzialmente APA/author–date**.

#### Esempio pratico

Nel testo:

> Neural networks can be trained using gradient-based optimization (Goodfellow et al., 2016).

e non:

> Neural networks can be trained using gradient-based optimization [17].

In bibliografia:

> Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press.

La guida UniTS raccomanda inoltre di usare un reference manager come **Zotero, RefWorks o EndNote**, proprio per mantenere coerente la bibliografia. ([Corsi UniTS](https://corsi.units.it/sites/default/files/media/documents/Guida%20per%20la%20redazione%20tesi_7-2024.pdf?utm_source=chatgpt.com "Università degli Studi di Trieste"))

#### Però c'è un dettaglio importante

La pagina ufficiale specifica del tuo corso DSAI **non prescrive uno stile bibliografico più specifico**: richiede che la tesi sia corredata da un'adeguata bibliografia. ([dsai.units.it](https://dsai.units.it/wp-content/uploads/2023/04/Regolamento-Didattico-DSAI-23-24.pdf?utm_source=chatgpt.com "Regolamento didattico del Corso di Laurea Magistrale in"))

Quindi io userei **la guida generale UniTS come norma di riferimento**, anziché copiare lo stile di un'altra tesi DSAI.

**Non riesco però ancora a vedere il file della tesi dello studente che dici di aver allegato in questo turno.** Se me lo alleghi qui, posso fare il confronto preciso tra:

1. **stile ufficiale UniTS**;
    
2. **stile effettivamente usato nella tesi dello studente**;
    
3. **stile attuale della tua bibliografia**;
    

e dirti esattamente cosa cambiare, inclusi `et al.`, maiuscole, corsivo, DOI, URL, conference papers, arXiv, articoli di journal, libri, ecc. 📚

[Guida UniTS per la redazione della tesi (PDF)](https://corsi.units.it/sites/default/files/media/documents/Guida%20per%20la%20redazione%20tesi_7-2024.pdf?utm_source=chatgpt.com)

---

## References

### 1. Chess, AlphaZero, NNUE

- **AlphaZero — Silver et al. (2018)**  
    [AlphaZero paper — arXiv](https://arxiv.org/abs/1712.01815?utm_source=chatgpt.com)  
    Utile per: AlphaZero, valore scalare vv, self-play, MCTS, confronto con alpha-beta. ([arXiv](https://arxiv.org/abs/1712.01815?utm_source=chatgpt.com "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"))
    
- **Stockfish — official repository**  
    [Stockfish GitHub](https://github.com/official-stockfish/Stockfish?utm_source=chatgpt.com)  
    Utile per: Stockfish, NNUE, alpha-beta engine, implementazione di riferimento. ([GitHub](https://github.com/official-stockfish/Stockfish?utm_source=chatgpt.com "GitHub - official-stockfish/Stockfish: A free and strong UCI chess engine · GitHub"))
    
- **Stockfish NNUE documentation**  
    [Official Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html?utm_source=chatgpt.com)  
    Questa è probabilmente **una delle fonti più importanti per la tua sezione NNUE**: descrive accumulator, sparse input, incremental update, low-precision/integer inference, ecc. ([Stockfish Docs](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html?utm_source=chatgpt.com "NNUE | Stockfish Docs"))
    
- **Stockfish NNUE-PyTorch**  
    [Stockfish NNUE-PyTorch GitHub](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md?plain=1&utm_source=chatgpt.com)  
    Utile se descrivi training, architettura, quantizzazione o implementazione pratica delle NNUE.
    
- **Stockfish neural networks**  
    [Official Stockfish NNUE networks](https://github.com/official-stockfish/networks?utm_source=chatgpt.com)  
    Utile se parli dei network NNUE effettivamente distribuiti da Stockfish. ([GitHub](https://github.com/official-stockfish/networks/blob/master/README.md?utm_source=chatgpt.com "networks/README.md at master · official-stockfish/networks · GitHub"))
    
- **YaneuraOu**  
    [YaneuraOu GitHub](https://github.com/yaneurao/yaneuraou?utm_source=chatgpt.com)  
    Utile per la storia/origine della NNUE nello shogi. ([GitHub](https://github.com/yaneurao/yaneuraou?utm_source=chatgpt.com "GitHub - yaneurao/YaneuraOu: YaneuraOu is the World's Strongest Shogi engine(AI player) , WCSC29 1st winner , educational and USI compliant engine. · GitHub"))
    
- **Yu Nasu — original NNUE implementation**  
    [Yu Nasu NNUE GitHub](https://github.com/ynasu87/nnue?utm_source=chatgpt.com)  
    È particolarmente interessante se vuoi citare **la fonte originale dell'architettura NNUE**, invece di citare soltanto la documentazione Stockfish. ([GitHub](https://github.com/ynasu87/nnue?utm_source=chatgpt.com "GitHub - ynasu87/nnue: Efficiently Updatable Neural-Network-based evaluation functions for computer shogi · GitHub"))
    
- **English translation of the original NNUE paper**  
    [NNUE paper — English translation](https://github.com/asdfjkl/nnue?utm_source=chatgpt.com)  
    Utile per una citazione accademica dell'origine della NNUE.
    

---

### 2. Mixture of Experts

- **Sparsely-Gated Mixture-of-Experts — Shazeer et al. (2017)**  
    [Shazeer et al., Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538?utm_source=chatgpt.com)  
    Questa è probabilmente la **reference fondamentale per introdurre il concetto moderno di MoE**. ([arXiv](https://arxiv.org/abs/1701.06538?utm_source=chatgpt.com "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"))
    
- **Switch Transformer — Fedus et al. (2021)**  
    [Switch Transformers](https://arxiv.org/abs/2101.03961?utm_source=chatgpt.com)  
    Utile per sparse top-1 routing, gating e distinzione tra numero totale di parametri e computazione per input. ([arXiv](https://arxiv.org/abs/2101.03961?utm_source=chatgpt.com "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"))
    
- **GLaM — Du et al. (2022)**  
    [GLaM — PMLR](https://proceedings.mlr.press/v162/du22c.html?utm_source=chatgpt.com)  
    Utile per MoE su larga scala e sparse activation. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v162/du22c.html?utm_source=chatgpt.com "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"))
    

---

### 3. LoRA e gradient-based specialization

- **LoRA — Hu et al. (2021)**  
    [LoRA paper — arXiv](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com)  
    Reference fondamentale per la tua sezione 2.2.3. ([arXiv](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com "LoRA: Low-Rank Adaptation of Large Language Models"))
    
- **ELREA — Li et al. (ICLR 2025)**  
    [ELREA — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e8711daef520be07cb9852390c673de8-Abstract-Conference.html?utm_source=chatgpt.com)  
    **Molto importante per la tua tesi**, perché utilizza proprio gradient directions per partizionare i dati e creare esperti LoRA. ([proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e8711daef520be07cb9852390c673de8-Abstract-Conference.html?utm_source=chatgpt.com "Ensembles of Low-Rank Expert Adapters"))
    
- **GradientSpace — Sridharan et al. (2025)**  
    [GradientSpace — arXiv](https://arxiv.org/abs/2512.06678?utm_source=chatgpt.com)  
    Probabilmente **la reference più vicina al tuo lavoro**: gradient clustering → expert specialization → lightweight router. ([arXiv](https://arxiv.org/abs/2512.06678?utm_source=chatgpt.com "GradientSpace: Unsupervised Data Clustering for Improved Instruction Tuning"))
    
- **TAGCOS — Zhang et al. (2024)**  
    [TAGCOS — arXiv](https://arxiv.org/abs/2407.15235?utm_source=chatgpt.com)  
    Molto utile per giustificare l'idea che **per-sample gradients possano essere usati come rappresentazione per il clustering**. ([arXiv](https://arxiv.org/abs/2407.15235?utm_source=chatgpt.com "TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data"))
    
- **Gradient Atoms — Rosser (2026)**  
    [Gradient Atoms — GitHub](https://github.com/jrosseruk/gradient_atoms?utm_source=chatgpt.com)  
    Interessante per la tua related work sui gradienti come fonte di struttura/interpretabilià. ([GitHub](https://github.com/jrosseruk/gradient_atoms?utm_source=chatgpt.com "GitHub - jrosseruk/gradient_atoms: Unsupervised discovery and steering of model behaviors via sparse decomposition of training gradients. · GitHub"))
    

---

### 4. Deep learning / fondamenti

- **Goodfellow, Bengio & Courville — Deep Learning**  
    [Deep Learning — official online book](https://www.deeplearningbook.org/?utm_source=chatgpt.com)  
    Puoi usarlo per le parti generali su neural networks, gradient descent, backpropagation, loss functions, ecc. ([Deep Learning Book](https://www.deeplearningbook.org/?utm_source=chatgpt.com "Deep Learning"))
    
- **MIT Press — Deep Learning**  
    [Deep Learning — MIT Press](https://mitpress.mit.edu/9780262337373/deep-learning/?utm_source=chatgpt.com)  
    Preferibile se vuoi una reference bibliografica editoriale ufficiale del libro. ([MIT Press](https://mitpress.mit.edu/9780262337373/deep-learning/?utm_source=chatgpt.com "Deep Learning"))
    

---

### 5. Riferimenti aggiuntivi (verificati online)

- **Hinton et al. (2015)** — knowledge distillation  
    Hinton, G. E., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv preprint arXiv:1503.02531. ([arXiv](https://arxiv.org/abs/1503.02531 "Distilling the Knowledge in a Neural Network"))

- **Kawata et al. (2025)** — MoE latent-cluster structure  
    Kawata, R., Matsutani, K., Kinoshita, Y., Nishikawa, N., & Suzuki, T. (2025). *Mixture of Experts Provably Detect and Learn the Latent Cluster Structure in Gradient-Based Learning*. *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*, 267, 29390–29448. ([PMLR](https://proceedings.mlr.press/v267/kawata25a.html "Mixture of Experts Provably Detect and Learn the Latent Cluster Structure in Gradient-Based Learning"))

- **Rousseeuw (1987)** — silhouette score  
    Rousseeuw, P. J. (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis*. *Journal of Computational and Applied Mathematics, 20*, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

- **Rodriguez & Laio (2014)** — Density Peak Clustering  
    Rodriguez, A., & Laio, A. (2014). *Clustering by fast search and find of density peaks*. *Science, 344*(6191), 1492–1496. https://doi.org/10.1126/science.1242072

- **Helfenstein et al. (2024)** — M2CTS  
    Helfenstein, F., Blüml, J., Czech, J., & Kersting, K. (2024). *Checkmating One, by Using Many: Combining Mixture of Experts with MCTS to Improve in Chess*. arXiv preprint arXiv:2401.16852. ([arXiv](https://arxiv.org/abs/2401.16852 "Checkmating One, by Using Many: Combining Mixture of Experts with MCTS to Improve in Chess"))

- **Ngo & Do (2026)** — Hexaïssa  
    Ngo, B., & Do, N. H. K. (2026). *Hexaïssa: Standing on Giants’ Shoulders—Routing the Best Chess Engines with Mixture-of-Experts and Latent Reward Learning*. *Proceedings of the AAAI Conference on Artificial Intelligence, 40*(29), 24532–24540. https://doi.org/10.1609/aaai.v40i29.39636

---

### 📌 La lista che userei effettivamente nella tua tesi

Se vuoi evitare una bibliografia enorme e mantenere **solo le reference realmente necessarie**, per il tuo Chapter 2 partirei da queste:

1. **Silver et al. (2018)** — AlphaZero
    
2. **Nasu (2018)** — NNUE
    
3. **Stockfish NNUE documentation** — implementazione/architettura NNUE
    
4. **Shazeer et al. (2017)** — Mixture of Experts
    
5. **Fedus et al. (2021)** — Switch Transformer
    
6. **Hu et al. (2021)** — LoRA
    
7. **Li et al. (2025)** — ELREA
    
8. **Sridharan et al. (2025)** — GradientSpace
    
9. **Zhang et al. (2024)** — TAGCOS
    
10. **Rosser (2026)** — Gradient Atoms
    
11. **Goodfellow et al. (2016)** — Deep Learning
    
12. **Stockfish** — se descrivi concretamente l'engine
    
13. **YaneuraOu** — se vuoi documentare la storia/origine NNUE.
    

### Una precisazione importante

Non metterei **semplicemente tutti questi URL nella bibliografia**. Per una tesi scientifica è meglio usare, quando disponibile, **il DOI o la pagina ufficiale del paper/proceedings** come riferimento bibliografico; GitHub/documentation va usato per software e dettagli implementativi.

Inoltre, nel tuo caso vedo che il testo della tesi contiene già alcuni `#todo reference...`, per esempio su **Stockfish, YaneuraOu, MCTS, ELREA, GradientSpace e TAGCOS**. Quindi questa lista può diventare direttamente una **checklist delle reference mancanti del Chapter 2**.

Se vuoi, nel prossimo passo posso prendere **tutte le citazioni presenti nel tuo Chapter 2** e costruirti una tabella `Citation → paper → DOI/arXiv URL → dove citarla nel testo → BibTeX`, pronta da importare in Zotero/LaTeX.

---