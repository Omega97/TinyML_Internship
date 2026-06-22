
## TODOs

- [x] Tutorial on how to load model on *Wio terminal*
- [x] pacchetto chiamato **Seeed SAMD Boards**
- [x] **Make a script to generate model structure and parameters**
- [x] **Modify the .mio to import the model correctly**
- [x] performance report excel
- [x] Fix calling time metric
- [ ] improve performance of the NN 
- [ ] integer **Lookup Table (LUT)** vs on-the-fly float conversion
- [ ] Reddit projects: explore for interesting projects
- [ ] Provare il metodo dei non-linear probes per vedere se un grosso modello è in grado di ricostruire lo stato del gioco (per scacchi) 🤖
- [ ] Thesis on Explainable tiny chess and World Models (TODO find name opposite of Magnus, which means big) Can models also build an internal board?
- [ ] **Data Pipeline**: to refine and decide (preprocessing, final dataset)
- [ ] Review Papers on state of the art about compressing chess engine into small chip
- [ ] Chess esp32: check what is it
- [ ] Provare a trainare un mini bot di scacchi 🤖
- [ ] competing paradigms for machine intelligence: what is this about?
- [ ] ICGA: what is it
- [ ] chessprogramming.org Discord
- [ ] r/chessprogramming
- [ ] World models to compress state info and facilitate training? 🌍


---

#todo fix .ino

### 1. 🚀 Massive Speed Optimization (Sparse Layer 1)
*   **The Change:** In Layer 1, instead of multiplying every weight by the input (most of which are `0.0f`), the code now checks `if (x[j] > 0.5f)` and only adds the weight if a piece is actually present on that square.
*   **The Impact:** Since a chess board only has ~32 pieces out of 768 possible features, this skips ~95% of the multiplications in the first layer. This significantly boosts your **Evals/sec** beyond the previous 2.16M limit.

### 2. 🧠 Accuracy Restoration (Fixed Binarization Bug)
*   **The Change:** Remove the `(h1[j] > 0.5f ? 1 : 0)` logic that was forcing hidden activations to be only `0` or `1`. The new code keeps activations as `float` between layers.
*   **The Impact:** Your neural network now preserves the "magnitude" of its thoughts. Previously, it was essentially making random coin-flip decisions; now it can accurately evaluate complex positional advantages.

### 3. 🛡️ Critical Sign Bug Fix
*   **The Change:** Add explicit casting to `(int8_t)` before converting weights to `float` (e.g., `(float)((int8_t)pgm_read_byte(...))`).
*   **The Impact:** `pgm_read_byte` returns an unsigned value (0–255). Without this cast, a negative weight like `-5` was being read as `251`, completely breaking the math. The model now correctly uses negative weights for penalizing bad positions.

### 4. 📐 Dynamic Architecture Support
*   **The Change:** Replace hardcoded loop limits (like `j < 768`) with macros defined in your header files (`FC1_IN_DIM`, `FC1_OUT_DIM`, etc.).
*   **The Impact:** You can now switch between **Nano**, **Tiny**, **Small**, and **Medium** models just by changing one `#define` line at the top of the file. The C++ code automatically adapts to the new layer sizes without manual editing.

### 5. 📊 Real-Time Health Monitoring
*   **The Change:** Add a `freeRam()` function and integrated it into the LCD/Serial output.
*   **The Impact:** You can now see exactly how much RAM is left while the stress test runs. This is crucial for the next step (adding a Negamax search), as you need to ensure the recursion stack doesn't cause a memory overflow.