# hce-d2-qsearch

**Created:** 2026-07-07  
**Status:** stable baseline

## Value function

| Field | Value |
|-------|--------|
| Backend | `hce` |
| Checkpoint | — |

## Search

| Field | Value |
|-------|--------|
| Algorithm | alpha-beta negamax v0.3 |
| Depth | 2 |
| Quiescence | on |

## Commands

```bash
py -3.12 scripts/run_engine.py --eval hce --depth 2
py -3.12 scripts/record_engine_game.py --eval hce --depth 2 --headless --max-plies 80
```

## Notes

Hand-crafted eval; useful baseline before NNUE. No checkpoint dependency.