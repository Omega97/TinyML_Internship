
## Come connettere il tuo bot Cfish/SARDINE a Lichess

Questa guida ti mostra come far giocare il tuo motore (Cfish, e in futuro il motore SARDINE personalizzato) su Lichess, in modo da poterlo sfidare comodamente dal telefono.

Useremo `lichess-bot`, un ponte open-source che collega il motore UCI all'API di Lichess.

---

### Panoramica dell'architettura

```
Telefono (Lichess App)   <--->   Server Lichess   <--->   PC (lichess-bot)   <--->   Motore (Cfish/SARDINE sul Wio)
```

- **Lichess** ospita la partita e la rende visibile sul tuo telefono.
- **`lichess-bot`** (sul PC) fa da ponte: riceve la posizione da Lichess, la passa al motore, e invia la mossa scelta.
- **Il motore** (Cfish sul PC, o SARDINE sul Wio) calcola la mossa.

Per ora, il motore gira sul PC. La sezione finale spiega come passare al Wio.

---

### Fase 1: Creare un account bot su Lichess

1. **Crea un nuovo account** su [Lichess.org](https://lichess.org).
   - Usa un account **separato** dal tuo personale.
   - **Non giocare partite manuali** con questo account, altrimenti non potrai più trasformarlo in bot.

2. **Genera un token API**:
   - Accedi al nuovo account.
   - Vai su **Preferences → API access tokens**.
   - Clicca su **"+ Create a new token"**.
   - Dagli un nome (es. `SARDINE-bot`) e spunta **tutti i permessi** relativi al gioco.
   - Clicca su **"Create"** e **copia subito il token** (non sarà più visibile).

3. **Trasforma l'account in "BOT"**:
   - Apri un terminale sul PC.
   - Esegui questo comando, sostituendo `IL_TUO_TOKEN` con il token appena copiato:

     ```bash
     curl -X POST https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer IL_TUO_TOKEN"
     ```

   - Se la risposta è `{"ok":true}`, l'account è ora un bot.

---

### Fase 2: Installare e configurare `lichess-bot`

1. **Installa Python** (versione 3.9 o successiva).

2. **Clona il repository**:
   ```bash
   git clone https://github.com/lichess-bot-devs/lichess-bot
   cd lichess-bot
   ```

3. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configura il file `config.yml`**:
   - Copia il file di esempio:
     ```bash
     cp config.yml.default config.yml
     ```
   - Apri `config.yml` con un editor di testo.
   - Trova la sezione `engine` e imposta il percorso del motore:

     ```yaml
     engine:
       dir: /percorso/alla/cartella/del/motore
       name: cfish.exe
       protocol: uci
       working_dir: /percorso/alla/cartella/del/motore
       uci_options:
         Hash: 16
         Threads: 1
     ```

     - **Se stai usando Cfish su PC**: imposta `dir` e `working_dir` sulla cartella `src/cfish/` del tuo progetto.
     - **Se stai usando SARDINE su Wio**: per ora tieni Cfish sul PC (vedi Fase 4 per il passaggio).

   - Inserisci il token API nella sezione `token`:
     ```yaml
     token: IL_TUO_TOKEN
     ```

   - (Opzionale) Imposta `allow_abort: false` per evitare che il bot abbandoni partite troppo lunghe.

---

### Fase 3: Avviare il bot

1. **Assicurati che il motore funzioni**: testalo con il comando `go depth 5` (vedi la guida su Cfish).

2. **Avvia `lichess-bot`** dalla sua cartella:
   ```bash
   python lichess-bot.py
   ```

3. **Verifica la connessione**:
   - Dovresti vedere un messaggio di conferma e il bot si metterà in attesa di sfide.
   - Vai su Lichess, cerca il nome del tuo bot e sfidalo! La partita apparirà anche sul telefono.

---

### Fase 4: Far girare il motore sul Wio Terminal

Per ora il motore gira sul PC. Per usare il Wio, hai due strade:

#### Opzione 1 (Consigliata): Bridge remoto
- Fai girare `lichess-bot` sul PC, ma configuralo per usare un motore che comunica con il Wio via seriale o rete.
- Dovrai scrivere un piccolo adattatore che:
  1. Riceve la posizione da `lichess-bot`.
  2. La invia al Wio (via USB seriale o WiFi).
  3. Riceve la mossa dal Wio e la restituisce a `lichess-bot`.

#### Opzione 2: Eseguire `lichess-bot` sul Wio
- Il Wio Terminal non ha abbastanza risorse per eseguire Python e `lichess-bot` insieme al motore.

#### Opzione 3 (semplice per test): Usare un ponte locale
- Tieni `lichess-bot` sul PC, ma punta il motore a un eseguibile che è un wrapper per il Wio.
- Questo wrapper:
  1. Avvia il motore sul Wio (es. via seriale).
  2. Inoltra i comandi UCI dal PC al Wio e le risposte dal Wio al PC.

**Consiglio pratico**: Per la fase di test, tieni il motore sul PC. Quando SARDINE sarà pronto, scrivi un piccolo bridge in Python che usa `pyserial` per comunicare con il Wio e lo usi come motore in `lichess-bot`.

---

### Risoluzione dei problemi

| Problema | Soluzione |
|----------|-----------|
| **Il bot non viene sfidato** | Assicurati che l'account sia stato promosso a bot e che il token sia corretto. |
| **Errore di connessione** | Verifica che il PC abbia accesso a Internet e che il firewall non blocchi le connessioni. |
| **Il motore non risponde** | Controlla il percorso del motore in `config.yml` e che l'eseguibile sia avviabile da riga di comando. |
| **Partita annullata per timeout** | Aumenta `Move Overhead` in `config.yml`. |
| **Il bot gioca mosse strane** | Verifica che il motore sia configurato correttamente (es. `Hash` e `Threads` adeguati). |

---

### Riferimenti utili

- [Documentazione ufficiale di `lichess-bot`](https://github.com/lichess-bot-devs/lichess-bot/wiki)
- [Guida su come creare un token OAuth](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-create-a-Lichess-OAuth-token)
- [Guida su come configurare il motore](https://github.com/lichess-bot-devs/lichess-bot/wiki/Setup-the-engine)

