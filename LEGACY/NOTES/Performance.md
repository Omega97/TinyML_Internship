
## Value function on Wio Terminal

*Updated 2026-06-24 — honest benchmark (volatile sink, interval EMA, sparse L1).*


| name   | architecture   | latency (ms) | evals/s | storage (bytes) | space% |
| ------ | -------------- | ------------ | ------- | --------------- | ------ |
| nano   | 768→16→8→1     | 1.4          | ~714    | 74312           | 14%    |
| tiny   | 768→32→16→1    | 2.8          | ~357    | 87016           | 17%    |
| small  | 768→64→32→1    | 5.8          | ~172    | 113192          | 22%    |
| medium | 768→128→64→1   | 11.4         | ~88     | 168632          | 33%    |
| big    | 768→256→64→1   | 22.6         | ~44     | 275320          | 54%    |
| huge   | 768→512→64→1   | 45.0         | ~22     | 488568          | 96%    |

*Note: Maximum internal flash is 507904 bytes. Latency measured on-device (Wio Terminal, starting FEN).*

### Hw–sw synergy (2026-06-24)

- Latency scales **~2× per model tier** (1.4 → 2.8 → 5.8 → 11.4 → 22.6 → 45.0 ms). Throughput is no longer flat — benchmark now measures real `evaluate()` calls.
- **Dominant cost:** `pgm_read_byte` from PROGMEM (flash bus stalls). Wider layers → more weight fetches → slower evals. FPU is rarely the limit.
- **Sparse L1 helps:** L1 skips reads where `x[j] == 0` (~32 active squares). L2/L3 still read every weight regardless of activation — likely the next win for medium+ models.
- **Memory:** Footprint scales linearly with params (~60 KB sketch overhead). Huge fits at ~96% flash.

### Observations (historical)

- Memory scales linearly with model size, with an overhead of about 60kB
- The biggest model we can run is approximately `768→512→64→1`

- **The Bottleneck:** Every time your code executes `pgm_read_byte(&fc1_w[...])`, the CPU has to wait for the Flash controller to fetch that byte. This takes several clock cycles.
- **The Loop Overhead:** Your C++ `for` loops have overhead: incrementing the counter (`j++`), checking the condition (`j < 768`), and jumping back to the start of the loop.
- **The FPU is Idle:** Because the CPU spends most of its time _waiting_ for data from Flash, the FPU (which can do a multiply-add in 1-2 cycles) is often sitting idle.

Seeed Wio Terminal 
- CPU frequency = 120 MHz
- RAM = 192 KB
- Flash memory = 500 kB

> **Analogy:** Imagine you are a chef (the CPU) who can chop vegetables incredibly fast (the FPU). But your ingredients (weights) are in a warehouse across town (Flash Memory). No matter if you are making a small salad (Nano) or a large stew (Medium), you spend 90% of your time driving to the warehouse and back. The chopping speed doesn't change because the _driving_ is the limit.

