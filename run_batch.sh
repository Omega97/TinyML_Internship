#!/usr/bin/env bash
set -euo pipefail

# Controlla se sono stati passati meno di 3 argomenti
if [ "$#" -lt 3 ]; then
    echo "Errore: Mancano uno o più argomenti obbligatori."
    echo "Uso corretto: $0 <numero_iniziale> <numero_finale> <step_size>"
    echo "Esempio: $0 12000000 12600000 200000"
    exit 1
fi

START="$1"
END="$2"
STEP="$3"

current=$START

while (( current < END )); do
    next=$(( current + STEP ))
    if (( next > END )); then
        next=$END
    fi

    echo "==> Esecuzione range: $current -> $next"
    python3.12 -u scripts/lichess_dump_to_fen_value_visits-gpu.py "$current" "$next" --dropout 0.95 --backend cuda-fp16 --nn-batch 512

    current=$next
done
