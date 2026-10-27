"""
pipeline_179.py - Capa REAL de los municipios de Madrid (E4).

Construye un dataset separado del panel de 20, con SOLO datos oficiales:

  codigo_ine, nombre, poblacion_actual, anio_padron,
  num_centros_salud, num_centros_educativos,
  valor_catastral_residencial_eur_m2, anio_valor_catastral,
  migracion_neta_externa, crecimiento_pob_pct_2020_2025

Genera dos salidas:
  - panel_179_real: los 179 municipios de la Comunidad de Madrid
  - panel_menores_50k_real: solo los que tienen menos de 50.000 habitantes
    (el ámbito del reto)

NO inventa las columnas calibradas del panel de 20 (precios, medicos, plazas,
vivienda, suelo, transporte...). Esa capa es para el mapa y el diagnostico,
no para entrenar el modelo de 92 columnas.

Los centros sin dato en CKAN se rellenan con 0 (municipio sin centros en la
fuente), nunca se dejan como nulos.

Uso:
  python -m src.lodcore.ingest.pipeline_179
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.lodcore.ingest.ckan_madrid import IngestorCKANMadrid
from src.lodcore.ingest.ckan_vivienda_migracion import (
    IngestorViviendaMigracionReal,
    SNAP_PADRON_6Y,
)
from src.lodcore.ingest.ine_padron import IngestorINEPadron

logger = logging.getLogger("lodcore.pipeline_179")

RAW_CKAN = Path("data/raw/ckan")
RAW_INE = Path("data/raw/ine")
OUT_CSV = Path("data/validated/panel_179_real.csv")
OUT_JSON = Path("data/output/panel_179_real.json")
OUT_CSV_50K = Path("data/validated/panel_menores_50k_real.csv")
OUT_JSON_50K = Path("data/output/panel_menores_50k_real.json")

# Ámbito del reto: municipios de menos de 50.000 habitantes
UMBRAL_POBLACION = 50000

FUENTE_DATOS = (
    "Población: INE Padrón 2881 | "
    "Equipamientos: CKAN CAM (sanidad/educación en ALTA) | "
    "Valor catastral: DG Catastro vía CKAN CAM | "
    "Migración externa: INE EVR vía CKAN CAM (2017-2021) | "
    "Crecimiento: INE Padrón 2020-2025"
)


def _ultimo_snapshot(patron: str) -> Path:
    rutas = sorted(RAW_CKAN.glob(patron))
    if not rutas:
        raise FileNotFoundError(f"No hay snapshots que cumplan {patron} en {RAW_CKAN}")
    return rutas[-1]


def ejecutar_pipeline_179(escribir: bool = True) -> Dict[str, Any]:
    logger.info("=== CAPA REAL DE 179 MUNICIPIOS ===")

    # 1. Población: padrón de 16 años (contiene el último año)
    snap_padron = RAW_INE / SNAP_PADRON_6Y
    if not snap_padron.exists():
        raise FileNotFoundError(f"No existe {snap_padron}")
    padron = IngestorINEPadron().procesar_poblacion_madrid(
        json.loads(snap_padron.read_text(encoding="utf-8"))
    )

    # 2. Equipamientos (CKAN)
    ing_ckan = IngestorCKANMadrid()
    snap_san = _ultimo_snapshot("ckan_madrid_sanidad_snapshot_*.json")
    snap_edu = _ultimo_snapshot("ckan_madrid_educacion_snapshot_*.json")
    sanidad = ing_ckan.procesar_sanidad(
        json.loads(snap_san.read_text(encoding="utf-8"))
    )
    educacion = ing_ckan.procesar_educacion(
        json.loads(snap_edu.read_text(encoding="utf-8")), solo_reglada=True
    )

    # 3. Vivienda, migración y crecimiento (ya integrados en E1)
    ing_vm = IngestorViviendaMigracionReal()
    valor_catastral = ing_vm.procesar_valor_catastral()
    migracion = ing_vm.procesar_migraciones_externas()
    crecimiento = ing_vm.procesar_crecimiento_padron()

    # 4. Construir el panel
    filas = []
    for codigo, datos in sorted(padron.items()):
        filas.append(
            {
                "codigo_ine": codigo,
                "nombre": datos["nombre_ine"],
                "poblacion_actual": datos["poblacion_oficial"],
                "anio_padron": datos["anio_padron"],
                "num_centros_salud": sanidad.get(codigo, {}).get("num_centros_salud_ckan"),
                "num_centros_educativos": educacion.get(codigo, {}).get("num_centros_educativos_ckan"),
                "valor_catastral_residencial_eur_m2": valor_catastral.get(codigo, {}).get("valor_catastral_residencial_eur_m2"),
                "anio_valor_catastral": valor_catastral.get(codigo, {}).get("anio_valor_catastral"),
                "migracion_neta_externa": migracion.get(codigo, {}).get("migracion_neta_externa"),
                "crecimiento_pob_pct_2020_2025": crecimiento.get(codigo, {}).get("crecimiento_pob_pct_2020_2025"),
            }
        )

    df = pd.DataFrame(filas)

    # Centros sin dato en CKAN = 0 centros en la fuente (nunca nulos)
    for col in ("num_centros_salud", "num_centros_educativos"):
        df[col] = df[col].fillna(0).astype(int)

    df["origen"] = "OBSERVADO"
    df["metodo"] = "PANEL_179_CAPA_REAL"
    df["dataset_version"] = "v1.1.0-F2"
    df["fecha_disponible"] = "2026-10-27"
    df["fecha_calculo"] = datetime.now().replace(microsecond=0).isoformat()
    df["fuente_datos"] = FUENTE_DATOS

    # Capa del reto: municipios de menos de 50.000 habitantes
    df_50k = df[df["poblacion_actual"] < UMBRAL_POBLACION].copy()
    df_50k["metodo"] = "PANEL_MENORES_50K_CAPA_REAL"

    # 5. Cobertura real
    clave = [
        "num_centros_salud",
        "num_centros_educativos",
        "valor_catastral_residencial_eur_m2",
        "migracion_neta_externa",
        "crecimiento_pob_pct_2020_2025",
    ]
    cobertura = {c: int(df[c].notna().sum()) for c in clave}
    completos = int(df[clave].notna().all(axis=1).sum())

    if escribir:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False, encoding="utf-8")
        OUT_JSON.write_text(
            json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        df_50k.to_csv(OUT_CSV_50K, index=False, encoding="utf-8")
        OUT_JSON_50K.write_text(
            json.dumps(df_50k.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    resumen = {
        "municipios_179": int(len(df)),
        "municipios_menores_50k": int(len(df_50k)),
        "poblacion_menores_50k": int(df_50k["poblacion_actual"].sum()),
        "columnas": int(len(df.columns)),
        "nulos_totales_179": int(df.isna().sum().sum()),
        "nulos_totales_50k": int(df_50k.isna().sum().sum()),
        "cobertura_por_fuente": cobertura,
        "municipios_con_todas_las_fuentes": completos,
        "salida_csv": str(OUT_CSV),
        "salida_json": str(OUT_JSON),
        "salida_csv_50k": str(OUT_CSV_50K),
        "salida_json_50k": str(OUT_JSON_50K),
    }
    logger.info("Capa 179: %s municipios · cobertura %s", len(df), cobertura)
    return resumen


if __name__ == "__main__":
    print(json.dumps(ejecutar_pipeline_179(), indent=2, ensure_ascii=False))
