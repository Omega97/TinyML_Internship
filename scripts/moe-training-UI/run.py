#!/usr/bin/env python3
"""Launch the MoE training UI.

Usage (from the repo root)::

    python scripts/moe-training-UI/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import main

if __name__ == "__main__":
    raise SystemExit(main())
