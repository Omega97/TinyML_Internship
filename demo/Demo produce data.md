
## How to produce data

```powershell
# Generate data
cd C:\Users\monfalcone\PycharmProjects\TinyMLInternship

# Parse games — n included, m excluded (n=0 is the start of the dump)
# Default --dropout 0.90 keeps ~1 ply in 10 (stem …_0-10_d90). Also writes features.npz.
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10 --dropout 0
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 0 10 --max-draw 0.30
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 100000 105000 --dropout 0 --max-moves 10

# Larger slice
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 30000 32000

# Count unique EPDs vs total slice rows (does not write the join)
py -3.12 -u scripts/count_fen_value_visits.py

# Join all datasets
py -3.12 -u scripts/join_fen_value_visits.py

# Add/overwrite STM wdl on existing JSON (no dump re-parse). Default --depth 1.
py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits --in-place
py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_puzzles --depth 2 --in-place
```