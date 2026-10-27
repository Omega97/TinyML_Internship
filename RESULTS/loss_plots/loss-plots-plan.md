# Loss landscape: random, L1, KITSUNE

Piano per i plot della presentazione. Non genera ancora gli script né i plot.

## Cosa deve mostrare la figura

Stesso modello, stesso punto nello spazio dei pesi, tre partizioni diverse. Per ogni bucket si disegna la cross-entropy media muovendo i pesi della head lungo due assi ortogonali:

- asse x: direzione del gradiente della loss di quel bucket
- asse y: una direzione casuale ortogonale al gradiente

L’ipotesi è corretta, con un significato preciso. La derivata della superficie rispetto a x, nell’origine, è la norma del gradiente medio del bucket. Dentro un bucket L1 i gradienti dei singoli campioni puntano in direzioni diverse e si cancellano, quindi la norma è piccola e la superficie è piatta lungo x. KITSUNE raggruppa i micro-bucket per similarità del gradiente medio, quindi la cancellazione è minore, la norma è più grande e la discesa lungo x è più ripida. Lungo y la derivata nell’origine è zero per tutti e tre i metodi, perché y è ortogonale al gradiente: il contrasto da leggere è la pendenza lungo x.

La partizione casuale è la baseline. Ogni bucket è un campione della stessa distribuzione, quindi il suo gradiente medio resta vicino al gradiente globale del pack. Sul checkpoint già allenato quel gradiente è piccolo, e la riga casuale deve essere la più piatta. L’ordine atteso sulla mediana di `||g||` è casuale ≤ L1 < KITSUNE.

## Scelte fissate

- Checkpoint unico: `dual_h128_H256_e200_bpe512_bs10000/best.pt` (il reference `W128_H256`, CE 0.6303). I pesi non vengono fine-tunati. Ogni pannello parte dagli stessi pesi.
- Dati: il pack `data/processed/board_eval/moe/moe_b4_2m` (2_000_000 righe), lo stesso su cui sono stati fittati i bucket L1.
- Spazio dei pesi: solo la head, cioè `l2.weight`, `l2.bias`, `head.weight`, `head.bias`. L1 resta congelata. È lo spazio in cui il clustering dei gradienti e il fine-tuning degli esperti sono definiti. Per `W=128`, `H=256` sono 66_563 parametri.
- Loss: la stessa del training, soft cross-entropy STM WDL (`ce_and_mae` in `src/tinymlinternship/nnue/moe_pipeline.py`), ridotta a media. Non la somma: la dimensione del bucket non deve gonfiare il gradiente.
- `B = 4` per la slide. I label L1 esistono anche per 2, 8 e 16; lo script accetta `--buckets` ma il deliverable della presentazione è `B=4`.
- Partizione casuale: ogni riga del pack va in un bucket uniforme in `{0, …, B−1}`, seed 0. Nessun clustering. I label si salvano in `RESULTS/loss_plots/labels/random_b4.npy`.
- KITSUNE: `B' = 8B = 32` micro-bucket L1, poi stitch in `B` macro-bucket. Nel repo non ci sono label KITSUNE, solo `RESULTS/KITSUNE/kitsune_moe.csv`. Lo script li ricostruisce su questo checkpoint e su questo pack. Se esiste un file di label esterno, `--kitsune-labels` lo usa al posto della ricostruzione.
- I label L1 si rileggono da `RESULTS/moe/l1_clustered_k4/labels.npy` (2_000_000 righe, allineate al pack). Non si rifà il k-means L1.

## Perché la scala deve essere comune

La superficie di un bucket è

`L_b(α, β) = CE( θ + α · s · u_b + β · s · v_b )`

dove `u_b = g_b / ||g_b||`, `g_b = ∇_θ L_b(θ)`, e `v_b` è gaussiano, reso ortogonale a `u_b` e rinormalizzato. `s` è uno scalare unico per tutti i pannelli e per tutti e tre i metodi.

`s = 0.05 · ||θ_head||`. Quindi `α = ±1` è uno spostamento del 5% della norma della head, nella stessa metrica euclidea per ogni pannello. La pendenza nell’origine vale `s · ||g_b||`: a parità di `s`, un gradiente più grosso si vede come una discesa più ripida.

Non normalizzare i filtri alla Li et al. 2018. Quella normalizzazione serve quando entrambi gli assi sono direzioni casuali. Qui deformerebbe `u_b`, che deve restare esattamente il gradiente, e renderebbe `||u||` e `||v||` diversi.

Non scalare l’asse x con `1/||g_b||`. Quella scelta forza pendenza 1 in ogni pannello e cancella il fenomeno.

`v_b` è diverso per ogni pannello: l’ortogonalità è rispetto al gradiente di quel bucket. Il seed è `hash(metodo, bucket)` e va salvato. In slide va detto che l’asse y non è una feature condivisa.

## Calcolo, senza materializzare i gradienti da 66k

1. Campionare `n = 8192` righe per bucket, seed 0. Se un bucket è più piccolo si usano tutte le righe e si annota `n`. Lo stesso sottocampione serve sia per il gradiente sia per la griglia: così l’asse x è il gradiente vero della superficie disegnata, e l’origine non risulta inclinata.
2. Una sola passata L1 sul sottocampione. Salvare `h` (concat STM, `2W`) e il target WDL. Da lì in poi si valuta solo `head_from_h`.
3. Un backward della CE media dà `g_b` sui quattro tensori della head. Appiattire nello stesso ordine di `state_dict` della head.
4. Costruire `u_b` e `v_b`.
5. Griglia `31 × 31`, `α, β ∈ [-1, 1]`. Per ogni nodo, sommare la perturbazione ai pesi della head (o una forward funzionale), calcolare la CE media, ripristinare i pesi. Sono circa 12 × 961 forward di 8192 righe sulla sola head: trascurabile anche su CPU.
6. Salvare `ΔL = L(α, β) − L(0, 0)`, non la CE assoluta. I bucket partono da loss diverse; il colore deve leggere la pendenza, non il livello.

## Come si ottengono i bucket KITSUNE

Sul pack intero, con lo stesso modello:

1. Attivazioni L1 (`l1_concat`), come in `build_l1` di `scripts/run_l1_clustered_moe.py`.
2. Mini-batch k-means a `B' = 32` in quello spazio. Sono celle convesse.
3. Per ogni micro-bucket, un backward della CE media sulle sue righe (a chunk). Si ottengono 32 vettori di 66_563, non un gradiente per campione. È il `ḡ_k` della nota `THESIS/markdown/notes/KITSUNE.md`.
4. Normalizzare quei vettori e fare k-means euclideo a `B = 4` (equivale al coseno). La mappa micro → macro assegna ogni riga del pack.
5. Salvare i label in `RESULTS/loss_plots/labels/`.

Il dispatcher non serve: la figura usa l’assegnazione offline.

## Figura

Cartella: `RESULTS/loss_plots/`.

File principale: `landscape_random_l1_kitsune_b4.png` e `.pdf`.

- Griglia 3 × 4. Riga 1: partizione casuale. Riga 2: L1 clustering. Riga 3: KITSUNE. Colonne: bucket 0–3.
- `imshow`, colormap `viridis`, interpolazione bilineare.
- Una sola colorbar, etichetta `Δ cross-entropy`, condivisa dai 12 pannelli. `vmin`/`vmax` dai percentili 2 e 98 di tutti i pannelli insieme, così un outlier non schiaccia la scala.
- Origine marcata con una croce.
- Asse x: `gradient direction`. Asse y: `random orthogonal direction`. Tick solo a −1, 0, 1.
- In ogni pannello, in piccolo: `n` e `||g||`.
- Contorni sottili scuri su 5 livelli, per leggere la pendenza anche a colpo d’occhio.
- Font grandi, pensati per una slide. DPI 200 per il PNG.

Accanto, `slopes.csv`: metodo, bucket, `n`, CE nell’origine, `||g||`, pendenza `s·||g||`. È il controllo numerico dell’ipotesi. La mediana di `||g||` deve rispettare casuale ≤ L1 < KITSUNE. Se non succede, la figura va comunque prodotta e il csv lo dice: non si ritocca la scala per forzare l’effetto.

Sidecar `scale.json`: checkpoint, pack, `B`, `B'`, `s`, `||θ_head||`, seed della partizione casuale, seed delle direzioni, griglia.

Per ogni pannello, `grids/<metodo>_b4_bucket<k>.npz` con `alpha`, `beta`, `delta_loss`, `grad_norm`. I metodi sono `random`, `l1`, `kitsune`. Rigenerare il PNG non richiede di ricalcolare la griglia.

## Script

`scripts/plot_loss_landscapes.py`, lanciato con `py -3.12`.

Riutilizza `load_dual_hidden_checkpoint`, `load_train_pack`, `ce_and_mae`, `fit_minibatch_kmeans`. Non aggiunge dipendenze. Backend Matplotlib `Agg`.

Argomenti: `--buckets` (default 4), `--micro-factor` (default 8), `--n-per-bucket` (default 8192), `--grid` (default 31), `--scale-fraction` (default 0.05), `--random-seed` (default 0), `--kitsune-labels` opzionale, `--out-dir` default `RESULTS/loss_plots`.

## Cosa non fare

- Non perturbare L1, e non usare i centroidi del proiettore a 48 dimensioni come direzione di plot: non vivono nello spazio dei pesi.
- Non confrontare pannelli con scale `s` diverse o con colorbar separate.
- Non plottare gli esperti già fine-tunati. Il punto è la loss del modello base ristretta al bucket.
- Non interpretare l’asse y come una direzione condivisa tra i bucket.
- Non bilanciare a mano le dimensioni dei bucket casuali: l’assegnazione uniforme con seed fisso è la baseline, e `n` annotato assorbe uno sbilanciamento residuo.
