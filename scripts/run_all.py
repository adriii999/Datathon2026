"""Ejecuta los dos pipelines del proyecto en orden y comprueba la salida.

    python scripts/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
MODELOS = REPO / "oracle" / "models"

EXITOS = [
    ("run_m3_demografia.py", "m3_demografia.onnx"),
    ("run_m4_capacidad.py", "m4_capacidad.onnx"),
]


def main() -> int:
    fallos = []
    for script, onnx in EXITOS:
        print(f"\n=== {script} ===")
        r = subprocess.run([sys.executable, str(SCRIPTS / script)])
        if r.returncode != 0:
            fallos.append(script)
            print(f"[run_all] FALLÓ {script}")
            break
    print("\n=== RESULTADO ===")
    ok = True
    for script, onnx in EXITOS:
        p = MODELOS / onnx
        existe = p.exists() and p.stat().st_size > 0
        print(f"  {'OK ' if existe else 'FALTA'}  {p}")
        ok &= existe
    return 0 if (ok and not fallos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
