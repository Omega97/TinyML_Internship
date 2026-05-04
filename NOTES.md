
### TinyML

Il framework [TinyML](https://github.com/marcozennaro/ICTP-UNU-TinyML) , 
sviluppato dall’ICTP di Trieste nasce con l’obiettivo di rendere l’intelligenza artificiale 
accessibile su hardware a basso consumo e costo contenuto. 
A differenza dei framework commerciali, si concentra sulla portabilità estrema e sulla 
semplificazione del codice C++ per permettere l’esecuzione di modelli complessi 
(come le reti neurali) su dispositivi con risorse limitate.

- [Intro to TinyML](https://github.com/marcozennaro/ICTP-UNU-TinyML/blob/main/Intro_TinyML.pdf)
- [Presentation of TinyML projects](https://github.com/marcozennaro/ICTP-UNU-TinyML/blob/main/Ellie%20Cai_Seeed%20TinyML%20for%20SDGs_20240417.pdf)


### NNUE

[NNUE](https://www.chessprogramming.org/NNUE) (Efficiently Updatable Neural Network) is 
a neural network architecture specifically designed for board game engines running on CPUs. 
Originally invented for Shogi by Yu Nasu in 2018 and later popularized in chess by Stockfish 
in 2020, NNUE revolutionized traditional alpha-beta engines by replacing hand-crafted 
evaluation functions with a compact neural network while maintaining extremely high speed.

[Basic NNUE](https://www.chessprogramming.org/NNUE#Basic_NNUE): The most basic form 
of an NNUE network consists of three layers: an input layer of length 768 
(768 = 6 pieces x 2 colors x 64 squares), one hidden layer of arbitrary size, and 
an output layer consisting of one neuron, representing the evaluation of the position. 
A NNUE network also commonly consists of two perspectives. That is, two hidden layers 
representing both sides are concatenated into a single hidden layer of twice the length, 
before being forwarded to the output layer. 

Read also: [NNUE Stockfish](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)

### Tecniche di Compressione del Modello

Per far stare un motore di scacchi in un chip XIAO, è necessario ridurre le dimensioni 
del modello senza crolli drastici nell'Elo.

- **Quantizzazione (4-bit):** Processo che riduce la precisione dei pesi della rete neurale. Passare da pesi a 32-bit (float) a **4-bit (integer)** permette di ridurre la memoria occupata di circa 8 volte. Sebbene introduca del "rumore", le moderne tecniche di _Quantization-Aware Training_ (QAT) minimizzano la perdita di precisione.
    
- **Pruning:** Tecnica che consiste nel rimuovere i parametri (connessioni tra neuroni) che contribuiscono meno al risultato finale. In una rete per scacchi, molti pesi vicini allo zero possono essere eliminati, rendendo il modello più "sparso" e leggero.


### Knowledge Distillation (Stockfish → NN)

Utilizzeremo un *teacher* (Stockfish) su un database di partite per allenare un 
modello studente con prestazioni più basse, ma molto più compatto.


### Xiao Setup

Si riferisce alla configurazione della serie [**Seeed Studio XIAO**](https://www.seeedstudio.com/xiao-series-page?srsltid=AfmBOoqiE4YsyhFz3buz-LnmXRlaxTLG7TspRI0r9wwwiZS_L8oZfuOl), 
una famiglia di microcontrollori estremamente compatti (grandi quanto un pollice). 
Vedi **XIAO ESP32-S3 Sense** o il **XIAO nRF52840 Sense**.

**Setup:** Prevede l'installazione delle Board Manager su Arduino IDE o PlatformIO 
e la configurazione dei sensori integrati (microfono, accelerometro o fotocamera) 
per raccogliere dati in tempo reale.


### Edge Impulse

[EdgeImpulse](https://www.edgeimpulse.com) è la piattaforma standard del settore per lo sviluppo di TinyML. 
Ti permette di gestire l'intero ciclo di vita del modello: acquisizione dati, 
estrazione delle feature (DSP), addestramento e deployment in formato libreria C++ ottimizzata.


### SenseCraft

[SenseCraft](https://sensecraft.seeed.cc): È un ecosistema software di Seeed Studio (spesso no-code o low-code) 
che facilita il deployment di modelli AI pronti all'uso su dispositivi edge, permettendo 
di visualizzare i risultati delle inferenze direttamente tramite browser o applicazioni dedicate.


### LiteRT

[**LiteRT:**](https://ai.google.dev/edge/litert?hl=it) È il nuovo brand e framework di Google per l'IA on-device 
(l'evoluzione di **TensorFlow Lite**). È progettato per essere più veloce e universale, 
supportando accelerazioni hardware specifiche (NPU/GPU).
   

### PyTorch Workflow

[**PyTorch Workflow:**](https://docs.pytorch.org/tutorials/index.html) Poiché molti modelli di ricerca sono scritti in PyTorch, 
il workflow prevede l'uso di [`litert-torch`](https://github.com/google-ai-edge/litert-torch) per convertire i modelli `.pt` 
direttamente in formato `.tflite`. Questo elimina la necessità di passaggi intermedi 
complessi e permette di far girare modelli PyTorch su microcontrollori.


### OTII (Otii Arc)

È lo strumento che abbiamo menzionato per misurare il consumo energetico. 
L'[**Otii Arc**](https://www.qoitech.com/otii-arc-pro/) di Qoitech funge sia da alimentatore che da analizzatore 
di corrente ad alta precisione (al nanosecondo).

**Utilità:** Ti permette di vedere in tempo reale quanta energia consuma il 
chip XIAO mentre calcola una mossa di scacchi, aiutandoti a ottimizzare il 
codice per la durata della batteria.


### MLSysBook AI Kits

[MLSysBook AI Kits](https://mlsysbook.ai/kits/contents/getting-started.html) si riferisce ai kit hardware didattici associati 
al libro [_"Machine Learning Systems"_](https://mlsysbook.ai/book/) di **Vijay Janapa Reddi** (Harvard).

**Contenuto:** Sono kit progettati per imparare a distribuire modelli ML 
su hardware reale ("TinyML kits"), affrontando problemi di latenza, 
memoria e affidabilità dei sistemi.

---

### FEN string

FEN (Forsyth-Edwards Notation) is a standard string representation of a chess position. 
It has 6 fields, separated by spaces:

$$
  \underbrace{\text{rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR}}_{\substack{\textbf{Piece Placement} \\ \text{(Ranks 8 to 1)}}}
  \;
  \underbrace{\text{b}}_{\substack{\textbf{Active Color} \\ \text{(w or b)}}}
  \;
  \underbrace{\text{KQkq}}_{\substack{\textbf{Castling} \\ \text{Availability}}}
  \;
  \underbrace{\text{e6}}_{\substack{\textbf{En Passant} \\ \text{Target}}}
  \;
  \underbrace{\text{0}}_{\substack{\textbf{Halfmove} \\ \text{Clock}}}
  \;
  \underbrace{\text{2}}_{\substack{\textbf{Fullmove} \\ \text{Number}}}
$$
