# Hardware Notes 🖥

### OTII (Otii Arc) 🔋

È uno strumento per misurare il consumo energetico. 
L'**Otii Arc** di Qoitech funge sia da alimentatore che da analizzatore 
di corrente ad alta precisione (al nanosecondo).

**Utilità:** Ti permette di vedere in tempo reale quanta energia consuma il 
chip XIAO mentre calcola una mossa di scacchi, aiutandoti a ottimizzare il 
codice per la durata della batteria.

#### Power profiling
L’**Otii Arc** (e Arc Pro) di Qoitech è uno degli strumenti migliori al mondo per misurare con altissima precisione il consumo energetico di dispositivi embedded come il tuo **XIAO ESP32-S3**.

Permette di:
- Alimentare il dispositivo (fino a 5V / 3A)
- Misurare corrente, tensione e potenza con risoluzione fino a **nA** (nanoampere)
- Registrare il profilo energetico nel tempo
- Calcolare l’**energia totale** consumata (mJ o µWh) per una singola mossa di scacchi

 https://www.youtube.com/watch?v=ztKSD2Q5-wQ
 https://www.youtube.com/watch?v=7efMvu2wCaA

---

### Edge Impulse

[EdgeImpulse](https://www.edgeimpulse.com) è la piattaforma standard del settore per lo sviluppo di TinyML. 
Ti permette di gestire l'intero ciclo di vita del modello: acquisizione dati, 
estrazione delle feature (DSP), addestramento e deployment in formato libreria C++ ottimizzata.

---

### MLSysBook AI Kits

[MLSysBook AI Kits](https://mlsysbook.ai/kits/contents/getting-started.html) si riferisce ai kit hardware didattici associati 
al libro [_"Machine Learning Systems"_](https://mlsysbook.ai/book/) di **Vijay Janapa Reddi** (Harvard).

**Contenuto:** Sono kit progettati per imparare a distribuire modelli ML 
su hardware reale ("TinyML kits"), affrontando problemi di latenza, 
memoria e affidabilità dei sistemi.

---

### Xiao Setup

Si riferisce alla configurazione della serie [**Seeed Studio XIAO**](https://www.seeedstudio.com/xiao-series-page?srsltid=AfmBOoqiE4YsyhFz3buz-LnmXRlaxTLG7TspRI0r9wwwiZS_L8oZfuOl), 
una famiglia di microcontrollori estremamente compatti (grandi quanto un pollice). 
Vedi **XIAO ESP32-S3 Sense** o il **XIAO nRF52840 Sense**.

**Setup:** Prevede l'installazione delle Board Manager su Arduino IDE o PlatformIO 
e la configurazione dei sensori integrati (microfono, accelerometro o fotocamera) 
per raccogliere dati in tempo reale.

---

[← Back to Notes index](notes.md)
