"""Prepend project src + legacy src to sys.path for archived pre-SARDINE tooling."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

for _rel in ("src", "legacy/pre-sardine/src"):
    _path = str(_ROOT / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)