"""
pipeline_datos_reales.py - Etapa 1: integra las fuentes REALES de vivienda y demografía
en el panel validado.

Enriquece data/validated/municipios_panel_validado.csv con cuatro columnas observadas:

  - valor_catastral_residencial_eur_m2 : DG Catastro (vía CKAN CAM), 2026
  - anio_valor_catastral               : año de referencia del valor catastral
  - migracion_neta_externa              : INE EVR (vía CKAN CAM), media anual 2017-2021
  - crecimiento_pob_pct_2020_2025      : INE Padrón 2881 (nult=6)

Regla de honestidad: no se elimina la parte calibrada del panel; se añade la observada
y fuente_datos declara ambas. El dataset_version pasa a v1.1.0-F2.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.lodcore.data.quality_gates import ejecutar_quality_gates
from src.lodcore.ingest.ckan_vivienda_migracion import (
    IngestorViviendaMigracionReal,
    SNAP_MIGRACIONES,
    SNAP_PADRON_6Y,
    SNAP_VALOR_CATASTRAL,
)

logger = logging.getLogger("lodcore.pipeline_datos_reales")

FUENTE_DATOS_V111 = (
    "Población: INE Padrón continuo 2025 (tabla 2881 API) | "
    "Equipamientos: CKAN CAM (Sanidad/Educación en ALTA) | "
    "Valor catastral residencial: DG Catastro vía CKAN CAM (2026) | "
    "Migración externa: INE EVR vía CKAN CAM (media anual 2017-2021) | "
    "Crecimiento 2020-2025: INE Padrón 2881 (nult=6) | "
    "Resto de dotación de servicios, vivienda y precios: SINTÉTICO calibrado (seed=2026)"
)


def _fecha_de_snapshots() -> str:
    """Deriva la fecha de disponibilidad de los snapshots realmente consumidos."""
    rutas = [
        Path("data/raw/ckan") / SNAP_VALOR_CATASTRAL,
        Path("data/raw/ckan") / SNAP_MIGRACIONES,
        Path("data/raw/ine") / SNAP_PADRON_6Y,
    ]
    fechas = []
    for r in rutas:
        m = re.search(r"(\d{8})", r.name)
        if m and r.exists():
            fechas.append(m.group(1))
    if not fechas:
        raise ValueError("No se pudo derivar la fecha de los snapshots de la Etapa 1")
    return f"{fechas[0][:4]}-{fechas[0][4:6]}-{fechas[0][6:]}"


def ejecutar_pipeline_datos_reales(actualizar_panel: bool = True) -> Dict[str, Any]:
    """Integra valor catastral, migración externa y crecimiento real en el panel."""
    logger.info("=== ETAPA 1: INTEGRACIÓN DE DATOS REALES (VIVIENDA + DEMOGRAFÍA) ===")

    panel_path = Path("data/validated/municipios_panel_validado.csv")
    if not panel_path.exists():
        raise FileNotFoundError(
            f"No existe el panel base '{panel_path}'. La Etapa 1 enriquece un panel "
            "existente; no lo crea desde cero."
        )
    df = pd.read_csv(panel_path)
    df["codigo_ine"] = df["codigo_ine"].astype(str).str.zfill(5)

    ing = IngestorViviendaMigracionReal()
    vc = ing.procesar_valor_catastral()
    mg = ing.procesar_migraciones_externas()
    pb = ing.procesar_crecimiento_padron()

    # Cobertura exigida: los 20 municipios del panel deben tener las tres fuentes.
    faltan = {
        "valor_catastral": [c for c in df["codigo_ine"] if c not in vc],
        "migracion_externa": [c for c in df["codigo_ine"] if c not in mg],
        "crecimiento_padron": [c for c in df["codigo_ine"] if c not in pb],
    }
    if any(faltan.values()):
        raise ValueError(f"Cobertura incompleta de las fuentes reales: {faltan}")

    n_vc = n_mg = n_pb = 0
    for i, row in df.iterrows():
        cod = row["codigo_ine"]
        if cod in vc:
            df.at[i, "valor_catastral_residencial_eur_m2"] = vc[cod]["valor_catastral_residencial_eur_m2"]
            df.at[i, "anio_valor_catastral"] = vc[cod]["anio_valor_catastral"]
            n_vc += 1
        if cod in mg:
            df.at[i, "migracion_neta_externa"] = mg[cod]["migracion_neta_externa"]
            n_mg += 1
        if cod in pb:
            df.at[i, "crecimiento_pob_pct_2020_2025"] = pb[cod]["crecimiento_pob_pct_2020_2025"]
            n_pb += 1

    # Procedencia y fechas: derivadas de los snapshots consumidos, no constantes ocultas
    fecha = _fecha_de_snapshots()
    df["origen"] = "OBSERVADO+SIMULADO"
    df["metodo"] = "PANEL_MUNICIPAL_INTEGRADO_F2"
    df["dataset_version"] = "v1.1.0-F2"
    df["fecha_disponible"] = fecha
    df["fecha_carga"] = fecha
    df["fecha_referencia"] = fecha
    df["fecha_calculo"] = datetime.now().replace(microsecond=0).isoformat()
    df["fuente_datos"] = FUENTE_DATOS_V111

    qg = ejecutar_quality_gates(df)
    logger.info(
        "Quality Gates superados: %s (%d municipios)", qg["status"], qg["filas_validadas"]
    )

    if actualizar_panel:
        out_val = Path("data/validated/municipios_panel_validado.csv")
        out_json = Path("data/output/municipios_comunidad_madrid_completo.json")
        out_val.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_val, index=False, encoding="utf-8")
        records = df.to_dict(orient="records")
        out_json.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Panel v1.1.0-F2 persistido en %s (%d columnas) y payload JSON regenerado",
            out_val, len(df.columns),
        )

    logger.info("=== ETAPA 1 FINALIZADA: 20/20 MUNICIPIOS CON DATOS REALES ===")
    return {
        "status": qg["status"],
        "dataset_version": "v1.1.0-F2",
        "fecha_disponible": fecha,
        "columnas_totales": int(len(df.columns)),
        "municipios": int(len(df)),
        "cobertura": {
            "valor_catastral_residencial": n_vc,
            "migracion_neta_externa": n_mg,
            "crecimiento_pob_2020_2025": n_pb,
        },
        "quality_gates": qg,
    }


if __name__ == "__main__":
    res = ejecutar_pipeline_datos_reales(actualizar_panel=True)
    print(json.dumps(res, indent=2, ensure_ascii=False))
