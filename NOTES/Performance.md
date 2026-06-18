
## Value function on Wio Terminal


|        | used storage <br>space <br>(bytes) | space% | performance<br>(1e6 evals / s) |
| ------ | ---------------------------------- | ------ | ------------------------------ |
| nano   | 74312                              | 14%    | 2.16                           |
| tiny   | 87016                              | 17%    | 2.16                           |
| small  | 113192                             | 22%    | 2.16                           |
| medium |                                    | 33%    | 2.16                           |

*Note: Maximum is 507904 bytes.*


Observations: 
- Memory scales linearly with model size, with an overhead of about 60kB
- The bottlenecks are the memory bandwidth and loop overhead.
- The biggest model we can run is approximately `768→512→64→1`

- **The Bottleneck:** Every time your code executes `pgm_read_byte(&fc1_w[...])`, the CPU has to wait for the Flash controller to fetch that byte. This takes several clock cycles.
- **The Loop Overhead:** Your C++ `for` loops have overhead: incrementing the counter (`j++`), checking the condition (`j < 768`), and jumping back to the start of the loop.
- **The FPU is Idle:** Because the CPU spends most of its time _waiting_ for data from Flash, the FPU (which can do a multiply-add in 1-2 cycles) is often sitting idle.

Seeed Wio Terminal 
- CPU frequency = 120 MHz
- RAM = 192 KB
- Flash memory = 500 kB

> **Analogy:** Imagine you are a chef (the CPU) who can chop vegetables incredibly fast (the FPU). But your ingredients (weights) are in a warehouse across town (Flash Memory). No matter if you are making a small salad (Nano) or a large stew (Medium), you spend 90% of your time driving to the warehouse and back. The chopping speed doesn't change because the _driving_ is the limit.

