#!/usr/bin/env bash
# Launch the two missing KITSUNE MoE runs sequentially (avoid OOM on shared GPU).
set -u
cd /home/omar/jupyterlab/TinyML_Internship
PY=/home/omar/jupyterlab/venv/bin/python

echo "=== [1/2] KITSUNE K=16 B'=64 60M ==="
$PY scripts/run_kitsune_headless.py --k 16 --max-rows 60 2>&1

echo
echo "=== [2/2] KITSUNE K=4 B'=16 90M (rerun) ==="
$PY scripts/run_kitsune_headless.py --k 4 --max-rows 90 2>&1

echo
echo "=== BATTERY DONE ==="
