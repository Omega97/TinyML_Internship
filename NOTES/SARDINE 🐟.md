
# SARDINE Blueprint
***S**mall **A**rtificial **R**AM-restricted **D**eep **I**ntelligent **N**eural **E**ngine*


### Components

- Multiple NNUE networks
- Policy? What type
- Monte Carlo Search Engine?

---

### Memory management

Total memory: **507 kB**

| **Component**                   | **Percentage** | **Core Function / Target Structure**                           |
| ------------------------------- | -------------- | -------------------------------------------------------------- |
| **Transposition Table (TT)**    | 70%            | Caches evaluated states to prevent search tree redundancy.     |
| **Search Stack & Accumulators** | 15%            | Scratchpad for ply states and incremental NNUE updates.        |
| **NNUE Value Network**          | 10%            | Quantized `int8` weights for non-linear positional evaluation. |
| **Algorithmic Policy**          | 5%             | Piece-type compressed History and Killer move tables.          |

---

### Runtime

...
