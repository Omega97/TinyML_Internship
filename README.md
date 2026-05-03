
# Report del Progetto
**Progetto di Tirocinio all'ICTP**

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

#todo: Review Papers on state of the art about compressing chess engine into small chip

> Vedi >> [dettagli progetto](PROJECT.md) <<

---

## 23-26 Aprile

- [TinyML Framework](NOTES.md#tinyml)
- [NNUE](NOTES.md#NNUE)
- [Model Compression](NOTES.md#tecniche-di-compressione-del-modello)
- [Knowledge Distillation](NOTES.md#knowledge-distillation-stockfish--nn)

---

## 27 Aprile - 3 Maggio

- [Xiao Setup](NOTES.md#xiao-setup)
- [Edge Impulse](NOTES.md#edge-impulse)
- [SenseCraft](NOTES.md#sensecraft)
- [LiteRT](NOTES.md#litert)
- [PyTorch Workflow](NOTES.md#pytorch-workflow)
- [Power Measurement with OTII](NOTES.md#otii-otii-arc)
- [MLSysBook AI Kits](NOTES.md#mlsysbook-ai-kits)

### Repo work

- [download_data.py](scripts/download_data.py)
- [test_data.py](tests/test_data.py)
- [view_game.py](tests/view_game.py)

---
