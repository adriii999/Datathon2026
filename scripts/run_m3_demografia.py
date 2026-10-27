"""Pipeline completo de M3 · Demografía.

Ejecutable desde cualquier directorio:
    python scripts/run_m3_demografia.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.m3_demografia.train_export import entrenar  # noqa: E402

if __name__ == "__main__":
    entrenar()
