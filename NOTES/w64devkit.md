
## 🛠️ Guida Rapida a w64devkit

### 1. Che cos'è w64devkit?

**w64devkit** è un ambiente di sviluppo portatile C/C++ per Windows in un unico pacchetto leggero. Non richiede installazione né diritti di amministratore.

- **Cosa include:** `gcc`, `g++`, `make`, `gdb`, la suite di utility **BusyBox** (con `sh`, `ls`, `grep`, `sed`, `cat`, ecc.) e `vim`.
    
- **Come si avvia:** Basta fare doppio clic su **`w64devkit.exe`**.
    

### 2. Navigazione nel File System

La shell di w64devkit è basata su BusyBox (`sh`), **non** è una macchina virtuale né WSL. Interagisce direttamente con i file di Windows.

- **Posizione iniziale:** Quando apri `w64devkit.exe`, la shell parte direttamente nella tua home utente Windows (`C:\Users\tuo_utente`).
    
- **Spostarsi tra le cartelle:** Usa la sintassi dei percorsi relativa o con slash Unix (`/`):
    
    Bash
    
    ```
    # Navigazione relativa (consigliata)
    cd PycharmProjects/TuoProgetto/src
    
    # Cambiare disco o percorso assoluto
    cd /c/Users/tuo_utente/Progetto
    cd /d/AltroDisco/Cartella
    ```
    

### 3. Compilazione e Build (Make & GCC)

Per compilare progetti C/C++ usando un `Makefile`:

Bash

```
# 1. Pulisci eventuali build precedenti
make clean

# 2. Compila sfruttando tutti i core della CPU (-j)
make -j

# 3. Compila specificando flag o target particolari (es. architettura)
make build ARCH=x86-64-avx2 -j
```

### 4. Dritte e Risoluzione Problemi Frequenti

#### 💡 Problema 1: Conflitti di nomi con le API Windows (`windows.h`)

Quando compili sorgenti C/C++ cross-platform su Windows, gli header di sistema (`windows.h`, `minwindef.h`) possono definire macro invadenti che vanno in conflitto con `enum` o variabili del tuo codice (es. `THREAD_EXIT`, `THREAD_RESUME`, `max`, `min`).

- **Soluzione A:** Nel `Makefile`, prova a passare l'opzione per MinGW (es. `COMP=mingw`).
    
- **Soluzione B:** Fai l'`#undef` esplicito nei file header sorgente subito dopo le inclusioni di sistema:
    
    C
    
    ```
    #ifdef THREAD_EXIT
    #undef THREAD_EXIT
    #endif
    ```
    

#### 💡 Problema 2: Scaricare il pacchetto corretto

- **Attenzione alle Release:** Da GitHub scarica sempre il file zip pre-compilato (es. `w64devkit-X.Y.Z.zip`).
    
- Evita i file marcati come _Source Code (zip/tar.gz)_, poiché contengono solo i sorgenti del devkit e non l'eseguibile `w64devkit.exe` né i binari di `gcc`.
    

#### 💡 Problema 3: Esecuzione diretta di script ed eseguibili

- Per lanciare un programma o un binario appena compilato nella cartella corrente, ricordati il prefisso `./`:
    
    Bash
    
    ```
    ./cfish.exe
    ```
    
- Per passare comandi rapidi a `w64devkit` da CMD/PowerShell senza aprire l'interfaccia grafica:
    
    DOS
    
    ```
    C:\w64devkit\w64devkit.exe make -C C:\percorso\progetto\src
    ```
    

### 5. Cheat Sheet Comandi Veloci

|**Comando**|**Descrizione**|
|---|---|
|`ls -la`|Lista tutti i file (inclusi i nascosti) con dettagli|
|`pwd`|Mostra la cartella corrente in formato Unix|
|`which gcc`|Rileva quale compilatore sta usando l'ambiente|
|`make -j`|Compila in parallelo usando tutti i thread della CPU|
|`clear`|Pulisce la schermata della console|
