# SARDINE — Mission

Build a playable chess bot that runs **entirely on-device** on the Seeed **Wio Terminal** (120 MHz, **192 KB RAM**, **~512 KB flash**): neural evaluation + shallow alpha-beta, maximizing **Elo per byte**. No cloud, no GPU.

| Target | Value |
| ------ | ----- |
| Elo | ≥ **1700** (match gate) |
| Move time | ~**1 s** |
| Eval | Bucketed micro **NNUE** (shared sparse L1 + expert heads) |
| Search | Alpha-beta + quiescence + pruning (v1) |

**Spec:** [NOTES/SARDINE Engine Blueprint.md](NOTES/SARDINE%20Engine%20Blueprint.md)  
**Status / checklist:** [PROJECT.md](PROJECT.md) · [TODOs.md](TODOs.md)

Session work is logged under local `DAILY-NOTES/` (gitignored).

**SARDINE** — *Small but mighty.* 🐟
