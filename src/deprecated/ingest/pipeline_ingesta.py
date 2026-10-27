"""
pipeline_ingesta.py - Orquestador de la Fase F1 de Ingesta Oficial y Snapshot Inmutable.
Integra datos reales de INE (Padrón continuo tabla 2881) y CKAN Comunidad de Madrid
(Recursos Sanitarios y Centros Educativos en ALTA), cruzando por código INE de 5 dígitos,
con procedencia honesta OBSERVADO+SIMULADO y validación previa a la escritura en disco.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.lodcore.data.quality_gates import ejecutar_quality_gates
from src.lodcore.ingest.ckan_madrid import IngestorCKANMadrid
from src.lodcore.ingest.ine_padron import IngestorINEPadron

logger = logging.getLogger("lodcore.pipeline_ingesta")

def ejecutar_pipeline_ingesta_f1(actualizar_panel: bool = True) -> Dict[str, Any]:
    """
    Ejecuta el pipeline de la Fase F1 con honestidad metodológica y linaje riguroso:
    1. Descarga y persiste snapshots inmutables de INE y CKAN Madrid sin mutar los ya existentes.
    2. Procesa y normaliza por código INE de 5 dígitos (28xxx) aplicando filtros de centros activos en ALTA.
    3. Cruza con el panel municipal piloto (20 municipios).
    4. Garantiza trazabilidad de linaje (origen = OBSERVADO+SIMULADO).
    5. Recalcula indicadores dependientes (ratios sanitarios y educativos).
    6. Valida con Quality Gates ANTES de persistir en disco.
    """
    logger.info("=== INICIANDO F1: INGESTA DE FUENTES OFICIALES REALES ===")

    # 1. Ingesta INE Padrón
    ingestor_ine = IngestorINEPadron()
    path_snap_ine, raw_ine = ingestor_ine.descargar_y_guardar_snapshot(overwrite=False)
    datos_ine = ingestor_ine.procesar_poblacion_madrid(raw_ine)
    logger.info("INE: %d registros municipales procesados", len(datos_ine))

    # 2. Ingesta CKAN Madrid
    ingestor_ckan = IngestorCKANMadrid()
    path_snap_san, raw_san = ingestor_ckan.descargar_y_guardar_snapshot_sanidad(overwrite=False)
    datos_san = ingestor_ckan.procesar_sanidad(raw_san)

    path_snap_edu, raw_edu = ingestor_ckan.descargar_y_guardar_snapshot_educacion(overwrite=False)
    datos_edu = ingestor_ckan.procesar_educacion(raw_edu, solo_reglada=True)
    logger.info("CKAN: Sanidad (%d municipios), Educación (%d municipios en ALTA)", len(datos_san), len(datos_edu))

    # 3. Cargar panel municipal base (única fuente de verdad)
    panel_validado_path = Path("data/validated/municipios_panel_validado.csv")
    if not panel_validado_path.exists():
        raise FileNotFoundError(
            f"No existe el panel base '{panel_validado_path}'. La ingesta enriquece "
            "un panel ya existente; no lo crea desde cero."
        )

    df = pd.read_csv(panel_validado_path)
    df["codigo_ine"] = df["codigo_ine"].astype(str).str.zfill(5)

    # La fecha de disponibilidad se deriva de los snapshots que realmente se
    # han consumido, no de una constante. Se toma la más reciente de las tres
    # fuentes: el panel no está completo hasta que llega la última.
    fechas_snapshot = []
    for snap in (path_snap_ine, path_snap_san, path_snap_edu):
        m_fecha = re.search(r"(\d{8})\.json$", snap.name)
        if m_fecha:
            fechas_snapshot.append(
                datetime.strptime(m_fecha.group(1), "%Y%m%d").date()
            )
    fecha_real_snapshot = (
        max(fechas_snapshot).isoformat() if fechas_snapshot else date.today().isoformat()
    )

    for idx, row in df.iterrows():
        cod_ine = row["codigo_ine"]

        # Cruce estricto por código INE (Población INE oficial)
        if cod_ine in datos_ine:
            df.at[idx, "poblacion_actual"] = datos_ine[cod_ine]["poblacion_oficial"]
            df.at[idx, "anio_padron"] = datos_ine[cod_ine]["anio_padron"]

        # Recuentos de sanidad oficial (Atención Primaria física en término municipal)
        if cod_ine in datos_san:
            cs = datos_san[cod_ine]["num_centros_salud_ckan"]
            cons = datos_san[cod_ine]["num_consultorios_ckan"]
            total_centros = cs + cons
            if total_centros > 0:
                df.at[idx, "num_centros_salud"] = total_centros

        # Recuentos de educación reglada en ALTA oficial
        if cod_ine in datos_edu:
            centros_edu = datos_edu[cod_ine]["num_centros_educativos_ckan"]
            if centros_edu > 0:
                df.at[idx, "num_centros_educativos"] = centros_edu

        # Recálculo coherente de ratios derivados
        pob = float(df.at[idx, "poblacion_actual"])
        medicos = float(df.at[idx, "num_medicos_ap"])
        if medicos > 0:
            df.at[idx, "hab_por_medico"] = round(pob / medicos, 0)
            df.at[idx, "medicos_x1000"] = round((medicos / pob) * 1000, 4)

        num_cs = float(df.at[idx, "num_centros_salud"])
        df.at[idx, "centros_salud_x10k"] = round((num_cs / pob) * 10000, 4)

        plazas = float(df.at[idx, "num_plazas_escolares"])
        alumnos = float(df.at[idx, "num_alumnos_matriculados"])
        if plazas > 0:
            df.at[idx, "ocupacion_escolar_pct"] = round((alumnos / plazas) * 100, 2)
            df.at[idx, "tasa_holgura_escolar"] = round(1.0 - (alumnos / plazas), 4)

        num_ce = float(df.at[idx, "num_centros_educativos"])
        if alumnos > 0:
            df.at[idx, "centros_edu_x1000_alumnos"] = round((num_ce / alumnos) * 1000, 4)

    # Metadatos obligatorios y transparentes de procedencia (Fase F1 y F2)
    df["origen"] = "OBSERVADO+SIMULADO"
    df["metodo"] = "PANEL_MUNICIPAL_INTEGRADO_F2"
    df["dataset_version"] = "v1.1.0-F2"
    df["fecha_disponible"] = fecha_real_snapshot
    df["fecha_carga"] = fecha_real_snapshot
    df["fecha_referencia"] = fecha_real_snapshot
    # Timestamp real de ejecución (no una hora inventada). Es el único campo
    # que cambia entre ejecuciones del mismo snapshot.
    df["fecha_calculo"] = datetime.now().replace(microsecond=0).isoformat()
    df["fuente_datos"] = (
        "Población: INE Padrón continuo 2025 (tabla 2881 API) | "
        "Equipamientos: CKAN CAM (Sanidad/Educación en ALTA) | "
        "Dotación servicios, vivienda y precios: SINTÉTICO calibrado (seed=2026)"
    )

    # 4. Validar calidad ANTES de persistir en disco
    qg_result = ejecutar_quality_gates(df)
    logger.info("Quality Gates superados antes de guardar: %s (%d municipios)", qg_result["status"], qg_result["filas_validadas"])

    if actualizar_panel:
        out_val = Path("data/validated/municipios_panel_validado.csv")
        # El JSON de data/output/ es el payload que consume la consolidación F2
        # y el frontend; serializa el mismo panel, no lo duplica en CSV.
        out_json = Path("data/output/municipios_comunidad_madrid_completo.json")

        out_val.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_val, index=False, encoding="utf-8")

        records = df.to_dict(orient="records")
        out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Panel persistido en data/validated y data/output (JSON) tras validación")

    logger.info("=== PIPELINE DE INGESTA F1 FINALIZADO CON ÉXITO ===")
    return {
        "status": "PASSED",
        "snapshot_ine": str(path_snap_ine),
        "snapshot_ckan_sanidad": str(path_snap_san),
        "snapshot_ckan_educacion": str(path_snap_edu),
        "municipios_validados": len(df),
        "quality_gates": qg_result
    }

if __name__ == "__main__":
    res = ejecutar_pipeline_ingesta_f1(actualizar_panel=True)
    print(json.dumps(res, indent=2, ensure_ascii=False))
