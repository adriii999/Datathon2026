"""Carga del único dataset de entrada del proyecto.

Todo dato usado por M3 y M4 proviene de:
    data/dataset_155_municipios_2010-2025.xlsx
o de rutas del propio repositorio. Nada más.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd

# .../Datathon2026/src/common/ -> repo = parents[2]
REPO = Path(__file__).resolve().parents[2]

XLSX = REPO / "data" / "dataset_155_municipios_2010-2025.xlsx"

DIR_MODELOS = REPO / "oracle" / "models"
DIR_SQL = REPO / "oracle" / "sql"
DIR_DOC = REPO / "doc"

ANIO_BASE = 2025      # último año observado de la serie
ANIO_MIN = 2010       # primer año observado
ANIO_MAX_PROYECCION = 2035


def sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cargar_panel() -> pd.DataFrame:
    """Panel apilado municipio-año 2010-2025 (2.480 filas)."""
    df = pd.read_excel(XLSX, sheet_name="Panel_2010_2025")
    df = df.rename(
        columns={
            "Código IEM": "cod_iem",
            "Código INE": "cod_ine",
            "Municipio": "municipio",
            "Año": "anio",
        }
    )
    df["cod_ine"] = df["cod_ine"].astype(str).str.zfill(5)
    df["anio"] = df["anio"].astype(int)
    return df.sort_values(["cod_ine", "anio"]).reset_index(drop=True)


def cargar_municipios() -> pd.DataFrame:
    """Estática municipal (155 filas): geografía y accesibilidad."""
    df = pd.read_excel(XLSX, sheet_name="Municipios")
    df["cod_ine"] = df["cod_ine"].astype(str).str.zfill(5)
    return df


def series_poblacion(panel: pd.DataFrame) -> pd.DataFrame:
    """Matriz municipio x año con la serie de población (sin huecos)."""
    serie = (
        panel.pivot(index="cod_ine", columns="anio", values="poblacion_total")
        .sort_index()
        .sort_index(axis=1)
    )
    assert not serie.isna().any().any(), "La serie de población no puede tener huecos"
    return serie
