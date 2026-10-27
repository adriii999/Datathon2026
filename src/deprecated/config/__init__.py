from functools import lru_cache
from pathlib import Path

import yaml

from .thresholds import *
from .weights import *

# Fecha de decisión del MVP: única fuente de verdad. Debe ser >= la fecha de
# disponibilidad del último snapshot consumido (Etapa 1: 2026-10-25). El filtro
# punto-en-tiempo la usa para garantizar que ningún dato posterior entra al panel.
FECHA_DECISION_MVP = "2026-10-25"

@lru_cache(maxsize=1)
def cargar_params() -> dict:
    """
    Carga `params.yaml` desde la raíz del repositorio (o desde este paquete como
    fallback). Centraliza hiperparámetros para que editar params.yaml tenga efecto
    real y no quede como configuración muerta.
    """
    candidatos = [
        Path(__file__).resolve().parents[3] / "params.yaml",
        Path(__file__).resolve().parent / "params.yaml",
    ]
    for ruta in candidatos:
        if ruta.is_file():
            contenido = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
            if not isinstance(contenido, dict):
                raise ValueError(f"params.yaml inválido en {ruta}: se esperaba un objeto YAML")
            return contenido
    return {}
