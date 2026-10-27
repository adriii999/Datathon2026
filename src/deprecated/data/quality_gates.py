"""
quality_gates.py - Puertas de calidad para el panel municipal validado (Fase F2).
Evita Great Expectations en Oracle (bug ORA-00907 con FROM DUAL) y valida tipos,
rangos y no-nulidad antes de permitir el paso a los 5 modelos.
"""

import pandas as pd
from typing import List, Dict, Any

REQUIRED_MUNICIPAL_COLUMNS = [
    "id_municipio",
    "codigo_ine",
    "nombre",
    "comarca",
    "poblacion_actual",
    "superficie_km2",
    "tiempo_medio_madrid_minutos",
    "precio_m2_compra",
    "num_medicos_ap",
    "num_plazas_escolares",
    "num_alumnos_matriculados",
    "hab_por_medico",
    "ocupacion_escolar_pct",
    "score_sanidad",
    "score_educacion",
    "score_conectividad",
    "score_vivienda",
    "indice_capacidad_global",
    "personas_acogida_efectiva",
    "factor_limitante",
    # Trazabilidad obligatoria (AGENTS.md): sin estos campos no se publica nada
    "origen",
    "metodo",
    "dataset_version",
    "fecha_disponible",
    "fuente_datos",
]

ORIGENES_VALIDOS = {"OBSERVADO", "IMPUTADO", "ESTIMADO", "PROYECTADO", "SIMULADO"}

def ejecutar_quality_gates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ejecuta las validaciones de esquema y dominio sobre el panel municipal.
    Lanza ValueError si alguna puerta de calidad falla.
    """
    errores: List[str] = []

    # 1. Columnas obligatorias
    faltantes = [c for c in REQUIRED_MUNICIPAL_COLUMNS if c not in df.columns]
    if faltantes:
        errores.append(f"Columnas obligatorias ausentes: {faltantes}")

    if not errores:
        # 2. No-nulidad de las columnas obligatorias (un nulo silencioso invalida el panel)
        nulos = {c: int(df[c].isna().sum()) for c in REQUIRED_MUNICIPAL_COLUMNS if df[c].isna().any()}
        if nulos:
            errores.append(f"Valores nulos en columnas obligatorias: {nulos}")

        # 2b. Procedencia: cada fila debe declarar un origen permitido (combinable con '+')
        origenes_invalidos = sorted({
            str(o) for o in df["origen"].dropna().unique()
            if any(parte.strip() not in ORIGENES_VALIDOS for parte in str(o).split("+"))
        })
        if origenes_invalidos:
            errores.append(f"Valores de 'origen' no permitidos: {origenes_invalidos}")

        # 2c. Fechas de disponibilidad parseables (control de punto-en-tiempo)
        if pd.to_datetime(df["fecha_disponible"], errors="coerce").isna().any():
            errores.append("'fecha_disponible' contiene fechas no parseables")

        # 2d. Unicidad de clave primaria e INE
        if df["id_municipio"].duplicated().any():
            errores.append("Duplicados detectados en 'id_municipio'")
        if df["codigo_ine"].duplicated().any():
            errores.append("Duplicados detectados en 'codigo_ine'")

        # 3. Criterio Datathon: Municipios < 50.000 habitantes
        if (df["poblacion_actual"] >= 50000).any() or (df["poblacion_actual"] <= 0).any():
            errores.append("Violación de dominio en 'poblacion_actual' (debe estar entre 1 y 49.999 hab.)")

        # 4. Sub-índices en rango [0, 100]
        for col in ["score_sanidad", "score_educacion", "score_conectividad", "score_vivienda", "indice_capacidad_global"]:
            if (df[col] < 0).any() or (df[col] > 100).any():
                errores.append(f"Sub-índice '{col}' fuera del rango [0, 100]")

        # 5. Precios positivos
        if (df["precio_m2_compra"] <= 0).any():
            errores.append("Precios por m² no positivos detectados")

        # 6. Ratios de saturación en dominio físico
        if (df["hab_por_medico"] <= 0).any():
            errores.append("'hab_por_medico' debe ser positivo")
        if (df["ocupacion_escolar_pct"] < 0).any():
            errores.append("'ocupacion_escolar_pct' no puede ser negativo")
        if (df["num_medicos_ap"] <= 0).any():
            errores.append("'num_medicos_ap' debe ser positivo (evita ratios infinitos)")
        if (df["num_plazas_escolares"] <= 0).any():
            errores.append("'num_plazas_escolares' debe ser positivo (evita ratios infinitos)")

    if errores:
        raise ValueError("PUERTA DE CALIDAD RECHAZADA: " + " | ".join(errores))

    return {
        "status": "PASSED",
        "filas_validadas": int(len(df)),
        "columnas_validadas": len(REQUIRED_MUNICIPAL_COLUMNS),
        "dataset_version": (
            str(df["dataset_version"].iloc[0])
            if "dataset_version" in df.columns and len(df) > 0
            else "SIN_VERSION_DECLARADA"
        ),
    }
