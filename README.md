
# Report del Progetto
**Progetto di Tirocinio all'ICTP**

> Dettagli progetto: [PROJECT.md](PROJECT.md)

> Note progetto: [NOTES/notes.md](NOTES/notes.md)

---

## 22 Aprile

Meeting all'ufficio del prof. Zennaro

### Obbiettivo del Progetto

> Caricare un'AI di scacchi su un componente di edge-computing. L'obbiettivo è avere un prodotto funzionante, che bilancia prestazioni e consumo. 
> Misureremo l'Elo della policy contro una policy di Elo noto e comparabile (molto efficiente, le partite durano meno di un secondo). 
> Se il chip dovesse essere in grado di gestire una *tree search*, dovremmo modellare il consumo del chip come $overhead + consumo/nodo * nodi$ . 
> Inoltre, la *tree search* userebbe una *value function*. Possiamo misurare la forza di questa in modo simile, trasformandola però prima in una *policy* (valutando ciascuna mossa possibile).
> Per completare il prodotto, bisogna valutare il sistema di input dello stato del gioco, e output dell'azione del bot. 

#### Spunti teorici
- Valutare le performance di una rete neurale quantizzata e distillata (a vari livelli di compressione) rispetto al bot originale, e ricavare una "legge fisica"
- Do AI models have an inner representation of the chessboard?
- How good is a latent vector representation of the board?


#### Di cosa abbiamo discusso col professore
1. Xiao setup
2. [EdgeImpulse](https://www.edgeimpulse.com) e [SenseCraft](https://sensecraft.seeed.cc)
3. [LiteRT](https://ai.google.dev/edge/litert) e PyTorch workflow
4. OTII
5. [MLSysBook AI kits](https://mlsysbook.ai/kits/)

---

## 23-26 Aprile
- TinyML Framework
- NNUE
- Model Compression
- Knowledge Distillation

(See [NOTES/notes.md](NOTES/notes.md) for details.)

---

## 27 Aprile - 3 Maggio
- Xiao Setup
- Edge Impulse
- SenseCraft
- LiteRT
- PyTorch Workflow
- Power Measurement with OTII
- MLSysBook AI Kits

(See [NOTES/notes.md](NOTES/notes.md) for details.)

### Repo work
- basic repo structure
- [download_data.py](scripts/download_data.py)
- [test_data.py](tests/test_data.py)
- [example_game.py](examples/example_game.py)

---

## 4-10 Maggio
- FEN
- Value functions
- Policy functions

(See [NOTES/notes.md](NOTES/notes.md) for chess/hardware/software notes; [PROJECT.md](PROJECT.md) for model details.)

### Repo work
- [example_fen.py](examples/example_fen.py)
- [test_policy_inference.py](tests/test_policy_inference.py)
- [featurizer.py](src/tinymlinternship/datasets/featurizer.py)

---

## 25-31 Maggio

- **Models**: list of model candidates
- **MCU Deployment**: what is it about?
- **Quantization**: How to perform (can it be done in Python??)
- MicroPython
- CircuitPython: what is it?
- **Power Profiling (OTII)**: how to
- TinyTorch not on hardware? How it works? does it work?

---

#core
