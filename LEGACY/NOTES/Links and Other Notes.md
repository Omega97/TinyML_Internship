
## Links

- [x] [Get Started with Wio Terminal](https://wiki.seeedstudio.com/Wio-Terminal-Getting-Started/)
- [x] [LabZero: a chess engine from zero in one working day](https://lichess.org/@/carlok/blog/labzero-a-chess-engine-from-zero-in-one-working-day/3VuU4oU7)
- [ ] [Cfish](https://talkchess.com/forum3/viewtopic.php?p=800832&sid=bdce3f751f10e3fc56a9f03e31d5e002#p800832)


## Other Notes

`cfish` : Activate the Cfish engine from console.

- [ ] iterative deepening, 
- [x] alpha-beta search, 
- [x] quiescence search, 
- [ ] aspiration windows, 
- [ ] null-move pruning, 
- [ ] late move reductions, 
- [ ] killer move ordering
- [ ] history move ordering, 
- [ ] SEE-based capture ordering, 
- [x] transposition table. 

Features:
- material, 
- tapered piece-square tables, 
- bishop pair, 
- pawn structure, 
- rook-file, 
- king-safety
- Lazy SMP mode

Checks:
- unit tests for engine internals;
- perft checks against known positions;
- random-game fuzzing;
- legality checks through `python-chess`;
- cross-checks against independent Rust chess libraries;
- UCI protocol smoke tests;
- tournament smoke and gauntlet scripts;
- benchmark scripts that save logs and PGNs.