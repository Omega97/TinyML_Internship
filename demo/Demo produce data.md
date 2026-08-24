
## How to produce data

```powershell
# Generate data
cd C:\Users\monfalcone\PycharmProjects\TinyMLInternship

# Games [20000, 22000) — n included, m excluded
py -3.12 -u scripts/lichess_dump_to_fen_value_visits.py 30000 32000

# Join all datasets
py -3.12 -u scripts/join_fen_value_visits.py

# Improve depth of the search (overwrite file)
py -3.12 -u scripts/relabel_fen_value_visits_lc0.py data/processed/board_eval/fen_value_visits/fen_value_visits_lichess_puzzles/fen_value_visits_lichess_puzzles.json --depth 2 --in-place
```