
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

> Vedi >> [note projetto](PROJECT.md) <<

---

## 23 Aprile


### 1. TinyML

Il framework [TinyML](https://github.com/marcozennaro/ICTP-UNU-TinyML) , sviluppato dall’ICTP di Trieste nasce con l’obiettivo di 
rendere l’intelligenza artificiale accessibile su hardware a basso consumo e costo contenuto. 
A differenza dei framework commerciali, si concentra sulla portabilità estrema e sulla 
semplificazione del codice C++ per permettere l’esecuzione di modelli complessi 
(come le reti neurali) su dispositivi con risorse limitate.

- [Intro to TinyML](https://github.com/marcozennaro/ICTP-UNU-TinyML/blob/main/Intro_TinyML.pdf)
- [Presentation of TinyML projects](https://github.com/marcozennaro/ICTP-UNU-TinyML/blob/main/Ellie%20Cai_Seeed%20TinyML%20for%20SDGs_20240417.pdf)


### 2. Tecniche di Compressione del Modello

Per far stare un motore di scacchi in un chip XIAO, è necessario ridurre le dimensioni 
del modello senza crolli drastici nell'Elo.

- **Quantizzazione (4-bit):** Processo che riduce la precisione dei pesi della rete neurale. Passare da pesi a 32-bit (float) a **4-bit (integer)** permette di ridurre la memoria occupata di circa 8 volte. Sebbene introduca del "rumore", le moderne tecniche di _Quantization-Aware Training_ (QAT) minimizzano la perdita di precisione.
    
- **Pruning:** Tecnica che consiste nel rimuovere i parametri (connessioni tra neuroni) che contribuiscono meno al risultato finale. In una rete per scacchi, molti pesi vicini allo zero possono essere eliminati, rendendo il modello più "sparso" e leggero.

### 3. Knowledge Distillation (Stockfish → NN)

Utilizzeremo un *teacher* (Stockfish) su un database di partite per allenare un 
modello studente con prestazioni più basse, ma molto più compatto.

---

## 27 Aprile - 3 Maggio


### 1. Xiao Setup

Si riferisce alla configurazione della serie [**Seeed Studio XIAO**](https://www.seeedstudio.com/xiao-series-page?srsltid=AfmBOoqiE4YsyhFz3buz-LnmXRlaxTLG7TspRI0r9wwwiZS_L8oZfuOl), 
una famiglia di microcontrollori estremamente compatti (grandi quanto un pollice). 
Vedi **XIAO ESP32-S3 Sense** o il **XIAO nRF52840 Sense**.

**Setup:** Prevede l'installazione delle Board Manager su Arduino IDE o PlatformIO 
e la configurazione dei sensori integrati (microfono, accelerometro o fotocamera) 
per raccogliere dati in tempo reale.


### 2. Edge Impulse

[EdgeImpulse](https://www.edgeimpulse.com) è la piattaforma standard del settore per lo sviluppo di TinyML. 
Ti permette di gestire l'intero ciclo di vita del modello: acquisizione dati, 
estrazione delle feature (DSP), addestramento e deployment in formato libreria C++ ottimizzata.


### 3. SenseCraft

[SenseCraft](https://sensecraft.seeed.cc): È un ecosistema software di Seeed Studio (spesso no-code o low-code) 
che facilita il deployment di modelli AI pronti all'uso su dispositivi edge, permettendo 
di visualizzare i risultati delle inferenze direttamente tramite browser o applicazioni dedicate.


### 4. LiteRT

[**LiteRT:**](https://ai.google.dev/edge/litert?hl=it) È il nuovo brand e framework di Google per l'IA on-device 
(l'evoluzione di **TensorFlow Lite**). È progettato per essere più veloce e universale, 
supportando accelerazioni hardware specifiche (NPU/GPU).
   

### 5. PyTorch Workflow

[**PyTorch Workflow:**](https://docs.pytorch.org/tutorials/index.html) Poiché molti modelli di ricerca sono scritti in PyTorch, 
il workflow prevede l'uso di [`litert-torch`](https://github.com/google-ai-edge/litert-torch) per convertire i modelli `.pt` 
direttamente in formato `.tflite`. Questo elimina la necessità di passaggi intermedi 
complessi e permette di far girare modelli PyTorch su microcontrollori.


### 6. OTII (Otii Arc)

È lo strumento che abbiamo menzionato per misurare il consumo energetico. 
L'[**Otii Arc**](https://www.qoitech.com/otii-arc-pro/) di Qoitech funge sia da alimentatore che da analizzatore 
di corrente ad alta precisione (al nanosecondo).

**Utilità:** Ti permette di vedere in tempo reale quanta energia consuma il 
chip XIAO mentre calcola una mossa di scacchi, aiutandoti a ottimizzare il 
codice per la durata della batteria.


### 7. MLSysBook AI Kits

[MLSysBook AI Kits](https://mlsysbook.ai/kits/contents/getting-started.html) si riferisce ai kit hardware didattici associati 
al libro [_"Machine Learning Systems"_](https://mlsysbook.ai/book/) di **Vijay Janapa Reddi** (Harvard).

**Contenuto:** Sono kit progettati per imparare a distribuire modelli ML 
su hardware reale ("TinyML kits"), affrontando problemi di latenza, 
memoria e affidabilità dei sistemi.

---
