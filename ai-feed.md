```bash
(.venv) C:\Users\monfalcone\PycharmProjects\TinyMLInternship>py -3.12 scripts/eval_bot_acpl.py --eval hce --depth 1 --max-plies 80 --no-quiescence --sf-movetime-ms 100 --verbose
pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html
Self-play game 1/1 (SARDINE 0.3.0 (hce, depth 1, qsearch=off))...
  80 half-moves, result *

=== Game ===
Moves analysed: 80
ACPL:           275.0 cp  (σ=1016.4)
Elo heuristic:  400  (range 400–2627)
  formula: Elo ≈ 2855 - ACPL × 10

Worst moves (top 5 CPL):
  ply 40 white: a1a2 (best h3h4) CPL=6391 [+0 vs +6391 cp]
  ply 39 white: a2a1 (best h3h4) CPL=6285 [+0 vs +6285 cp]
  ply 35 black: a4a5 (best a4b5) CPL=1999 [-8115 vs -6116 cp]
  ply 37 black: a4a5 (best a4b5) CPL=1014 [-6883 vs -5869 cp]
  ply 5 white: d4f6 (best e2e4) CPL=741 [-367 vs +374 cp]
```

```bash
(.venv) C:\Users\monfalcone\PycharmProjects\TinyMLInternship>py -3.12 scripts/record_engine_game.py --eval hce --depth 2 --headless --max-plies 80 --output images/hce_d2_game.gif
pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html
Playing engine self-play (SARDINE 0.3.0 (hce, depth 2))...
```

```bash

```