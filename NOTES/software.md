# Software Notes 🤖

### TinyML

Il framework [TinyML](https://github.com/marcozennaro/ICTP-UNU-TinyML) , 
sviluppato dall’ICTP di Trieste nasce con l’obiettivo di rendere l’intelligenza artificiale 
accessibile su hardware a basso consumo e costo contenuto. 
A differenza dei framework commerciali, si concentra sulla portabilità estrema e sulla 
semplificazione del codice C++ per permettere l’esecuzione di modelli complessi 
(come le reti neurali) su dispositivi con risorse limitate.

- [Intro to TinyML](https://github.com/marcozennaro/ICTP-UNU-TinyML/blob/main/Intro_TinyML.pdf)
- [Presentation of TinyML projects](https://github.com/marcozennaro/ICTP-UNU-TinyML/blob/main/Ellie%20Cai_Seeed%20TinyML%20for%20SDGs_20240417.pdf)

---

### Tecniche di Compressione del Modello

Per far stare un motore di scacchi in un chip XIAO, è necessario ridurre le dimensioni 
del modello senza crolli drastici nell'Elo.

- **Quantizzazione (4-bit):** Processo che riduce la precisione dei pesi della rete neurale. Passare da pesi a 32-bit (float) a **4-bit (integer)** permette di ridurre la memoria occupata di circa 8 volte. Sebbene introduca del "rumore", le moderne tecniche di _Quantization-Aware Training_ (QAT) minimizzano la perdita di precisione.
    
- **Pruning:** Tecnica che consiste nel rimuovere i parametri (connessioni tra neuroni) che contribuiscono meno al risultato finale. In una rete per scacchi, molti pesi vicini allo zero possono essere eliminati, rendendo il modello più "sparso" e leggero.

---

### Knowledge Distillation (Stockfish → NN)

Utilizzeremo un *teacher* (Stockfish) su un database di partite per allenare un 
modello studente con prestazioni più basse, ma molto più compatto.

---

### PyTorch Workflow

[**PyTorch Workflow:**](https://docs.pytorch.org/tutorials/index.html) Poiché molti modelli di ricerca sono scritti in PyTorch, 
il workflow prevede l'uso di [`litert-torch`](https://github.com/google-ai-edge/litert-torch) per convertire i modelli `.pt` 
direttamente in formato `.tflite`. Questo elimina la necessità di passaggi intermedi 
complessi e permette di far girare modelli PyTorch su microcontrollori.

---

### SenseCraft

[SenseCraft](https://sensecraft.seeed.cc): È un ecosistema software di Seeed Studio (spesso no-code o low-code) 
che facilita il deployment di modelli AI pronti all'uso su dispositivi edge, permettendo 
di visualizzare i risultati delle inferenze direttamente tramite browser o applicazioni dedicate.

---

### TensorFlow Lite

**TensorFlow Lite** (spesso abbreviato **TFLite**) è il framework ufficiale di Google per eseguire modelli di Machine Learning **direttamente sui dispositivi** (on-device), invece di mandare i dati al cloud.

È stato progettato specificamente per ambienti con **risorse limitate**: smartphone, Raspberry Pi, microcontrollori (MCU) e dispositivi IoT.

---

### LiteRT

[**LiteRT:**](https://ai.google.dev/edge/litert?hl=it) È il nuovo brand e framework di Google per l'IA on-device 
(l'evoluzione di **TensorFlow Lite**). È progettato per essere più veloce e universale, 
supportando accelerazioni hardware specifiche (NPU/GPU).

---

### Quantization: *How to perform it?*

Quantization means converting the weights (and sometimes activations) of the neural network from **32-bit floating point (float32)** to lower precision (INT8 or INT4/4-bit).

**Benefits for your project:**

- Dramatically reduces model size (4x–8x smaller)
- Much faster inference on MCU
- Much lower power consumption (critical for OTII measurements)
- Essential to fit a chess model on the XIAO

Quantization can (and should) be done **entirely in Python**. It's actually the easiest and most effective stage to do on your laptop before deploying to the XIAO.

---

### MicroPython

**MicroPython** è una versione **leggera e ottimizzata** del linguaggio Python 3, progettata per girare direttamente sui microcontrollori (MCU) con risorse molto limitate.

È stata creata nel 2014 da Damien George e rappresenta il progetto “originale”. CircuitPython è un fork (una derivazione) di MicroPython fatto da Adafruit per renderlo ancora più semplice per principianti e per i loro hardware.

---

### CircuitPython

**CircuitPython** è una versione di **Python** pensata specificamente per **microcontrollori** (MCU). È stata creata da **Adafruit** come fork di MicroPython, con l’obiettivo di rendere la programmazione di hardware il più semplice e “Pythonica” possibile.

Invece di programmare in C++, scrivi codice **direttamente in Python**. Quando colleghi la scheda (es. XIAO ESP32-S3) al computer, appare come una **chiavetta USB** chiamata CIRCUITPY. Modifichi il file code.py e il codice viene eseguito **automaticamente** → niente compilazione, niente flash complicato.

---

### TinyTorch

**TinyTorch** è un **framework educativo** open-source creato da **Vijay Janapa Reddi** (Harvard) e dal team di MLSysBook.

È essenzialmente un **"PyTorch fatto da zero"** per scopi didattici. Invece di usare PyTorch come una scatola nera (import torch), con TinyTorch **tu implementi tu stesso** i pezzi fondamentali del framework:

- Tensor
- Autograd (backpropagation)
- Optimizers (SGD, Adam…)
- Layer (Linear, Conv, ecc.)
- Training loop

È il compagno pratico del libro **"Machine Learning Systems"**.

*Nota: TinyTorch è pensato per girare sul tuo computer (laptop normale, anche senza GPU), **NON sui microcontrollers**.*

---

### Chess ESP32

**"Chess ESP32"** si riferisce a diversi progetti che eseguono un **motore di scacchi direttamente su ESP32** (incluso il tuo XIAO ESP32-S3).

Sono progetti open-source che trasformano l’ESP32 in un vero e proprio **computer scacchistico** embedded: il microcontrollore genera mosse, valuta posizioni e gioca a scacchi in modo autonomo (senza bisogno di un PC potente).

Progetti principali:
- esp32-chess-engine (Sergey Urusov)
- Dog (Folkert van Heusden)

---

[← Back to Notes index](notes.md)
