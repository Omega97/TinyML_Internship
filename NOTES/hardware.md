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

### Wio Terminal D51R

The **Wio Terminal** is a compact, feature-rich, all-in-one development board ideal for rapid IoT prototyping, interactive projects, and portable HMI applications. 🚀

- **MCU**: Microchip **ATSAMD51P19** (ARM Cortex-M4F @ 120 MHz, boostable to 200 MHz) with 512 KB flash, 192 KB RAM, and 4 MB external SPI flash.
- **Wireless**: Realtek **RTL8720DN** supporting **dual-band Wi-Fi (2.4/5 GHz)** and **Bluetooth/BLE 5.0**.
- **Display**: 2.4-inch TFT LCD (320×240 resolution).
- **Built-in Peripherals**: LIS3DHTR IMU (accelerometer), light sensor, microphone, buzzer, IR emitter, microSD slot, 5-way joystick, and three buttons.
- **Expansion**: 2× Grove ports + Raspberry Pi-compatible 40-pin GPIO + USB-C (with OTG).

**Software Support**: Excellent compatibility with **Arduino** (primary), **CircuitPython**, and **MicroPython**, making it accessible for both beginners and advanced users.

#### Strengths & Use Cases
Its combination of a color screen, wireless connectivity, rich sensors, and small form factor (~72 × 57 × 12 mm) makes it particularly strong for:
- Wireless sensors and data loggers
- Portable dashboards and HMIs
- TinyML/edge AI experiments
- Interactive gadgets and educational projects

#### Porte  e Interfacce Esterne Principali

| Porta / Connettore          | Nome / Descrizione                           | Posizione / Note      |
| --------------------------- | -------------------------------------------- | --------------------- |
| **USB Type-C**              | USB Type-C (Power + Data + OTG)              | Lato inferiore        |
| **Grove Port 1** (sinistra) | **Grove I²C** (principale, multifunzione)    | Fondo, lato sinistro  |
| **Grove Port 2** (destra)   | **Grove Digital/Analog/PWM** (configurabile) | Fondo, lato destro    |
| **40-pin GPIO**             | **Raspberry Pi compatible 40-pin header**    | Retro del dispositivo |
| **microSD Card Slot**       | Slot per scheda microSD (fino a 16 GB)       | Lato inferiore        |
| **FPC Connector**           | Connettore FPC 20-pin (per display interno)  | Interno               |

---

[← Back to Notes index](notes.md)
