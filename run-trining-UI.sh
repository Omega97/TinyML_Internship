#!/usr/bin/env bash
# Double-click to launch the MoE Training UI.
set -e

# Change directory to the root where this script resides
cd "$(dirname "$0")"

# Use relative path to the venv if present, or fallback to python3
if [ -f "./venv/bin/python" ]; then
  PYTHON_BIN="./venv/bin/python"
elif [ -f "../venv/bin/python" ]; then
  PYTHON_BIN="../venv/bin/python"
else
  PYTHON_BIN="/home/omar/jupyterlab/venv/bin/python"
fi

exec "$PYTHON_BIN" scripts/moe-training-UI/run.py
