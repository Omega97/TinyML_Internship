#!/usr/bin/env bash
# Double-click to launch the MoE Training UI.
set -e
cd "$(dirname "$0")"
exec /home/omar/jupyterlab/venv/bin/python scripts/moe-training-UI/run.py
