"""
optimizador_presupuestario.py - Asignación Presupuestaria por Comarca (Heurística Voraz Acotada MVP).
Fase F1 / M4: Distribuye una partida presupuestaria pública entre los municipios de una comarca
maximizando la ganancia neta en capacidad de acogida efectiva y desbloqueando cuellos de botella
sanitarios y educativos, evitando inversiones inútiles en municipios limitados por la vivienda.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from src.lodcore.config.thresholds import (
    COSTE_AMPLIACION_CONSULTA_MEDICA_EUR,
    COSTE_UNITARIO_PLAZA_ESCOLAR_EUR,
    MAX_HABITANTES_POR_MEDICO_AP,
    MAX_OCUPACION_ESCOLAR_PCT
)
from src.lodcore.data.provenance import TipoOrigen
from src.lodcore.models.m4_capacidad.reglas_no_compensatorias import aplicar_salvaguarda_no_compensatoria

@dataclass
class AsignacionMunicipio:
    codigo_ine: str
    municipio: str
    comarca: str
    medicos_asignados: int
    plazas_escolares_asignadas: int
    coste_eur: float
    ganancia_personas: int
    nuevo_ica: float
    semaforo: str
    motivo_asignacion: str

def optimizar_presupuesto_comarca(
    comarca: str,
    presupuesto_total_eur: float,
    data_path: str = "data/validated/municipios_panel_validado.csv"
) -> Dict[str, Any]:
    """
    Asigna un presupuesto cerrado entre los municipios de una comarca determinada
    utilizando una heurística voraz acotada (Greedy) con control de cuello de botella:
    - Poda de ineficiencia: no invierte en sanidad o educación si el municipio no puede acoger más
      personas debido a cuello de botella insalvable en vivienda.
    - Prioriza inversiones con mayor ratio de acogida efectiva por euro invertido (ganancia_personas / coste).
    - Recalcula los indicadores y el ICA con salvaguardas no compensatorias oficiales.
    """
    df = pd.read_csv(data_path)
    df["codigo_ine"] = df["codigo_ine"].astype(str).str.zfill(5)

    if comarca.lower() != "todas":
        muni_comarca = df[df["comarca"].str.lower() == comarca.lower()].copy()
    else:
        muni_comarca = df.copy()

    if muni_comarca.empty:
        comarcas_validas = sorted(df["comarca"].unique().tolist())
        raise ValueError(f"Comarca '{comarca}' no encontrada. Opciones válidas: {comarcas_validas}")

    presupuesto_restante = float(presupuesto_total_eur)
    asignaciones: List[AsignacionMunicipio] = []
    incremento_total_personas = 0

    candidatos_evaluados = []

    for _, row in muni_comarca.iterrows():
        pob = float(row["poblacion_actual"])
        medicos = float(row["num_medicos_ap"])
        plazas = float(row["num_plazas_escolares"])
        alumnos = float(row["num_alumnos_matriculados"])
        factor_lim = str(row.get("factor_limitante", "VIVIENDA"))
        ica_base = float(row.get("indice_capacidad_global", 50.0))

        abs_san_base = max(0.0, medicos * MAX_HABITANTES_POR_MEDICO_AP - pob)
        abs_edu_base = max(0.0, (plazas * (MAX_OCUPACION_ESCOLAR_PCT / 100.0) - alumnos) / 0.17)
        abs_viv_base = float(row.get("personas_absorb_vivienda", 0))
        acogida_efectiva_base = min(abs_san_base, abs_edu_base, abs_viv_base)

        # Techo de acogida impuesto por la vivienda
        margen_vivienda = max(0.0, abs_viv_base - acogida_efectiva_base)

        # Si el factor limitante es la vivienda y no hay holgura residencial, una inversión
        # en médicos o aulas no incrementa la acogida efectiva poblacional.
        if factor_lim == "VIVIENDA" and margen_vivienda <= 0:
            candidatos_evaluados.append({
                "row": row,
                "factible": False,
                "motivo": "Municipio limitado por vivienda; CAPEX sanitario/educativo no genera acogida",
                "medicos_propuestos": 0,
                "plazas_propuestas": 0,
                "coste_propuesto": 0.0,
                "ganancia_estimada": 0,
                "eficiencia": 0.0
            })
            continue

        # Calcular dotaciones requeridas para expandir acogida hasta el techo de vivienda o resolver déficit
        medicos_necesarios = max(0, math.ceil(pob / MAX_HABITANTES_POR_MEDICO_AP) - int(medicos))
        plazas_necesarias = max(0, math.ceil(alumnos / (MAX_OCUPACION_ESCOLAR_PCT / 100.0)) - int(plazas))

        # Determinar propuesta de dotación equilibrada
        m_prop = medicos_necesarios
        p_prop = plazas_necesarias

        # Si no había déficit crítico pero el cuello de botella son servicios, proponer ampliación
        if m_prop == 0 and p_prop == 0:
            if factor_lim == "SANIDAD":
                m_prop = 1
            elif factor_lim == "EDUCACION":
                p_prop = 50

        coste_prop = (
            m_prop * COSTE_AMPLIACION_CONSULTA_MEDICA_EUR +
            p_prop * COSTE_UNITARIO_PLAZA_ESCOLAR_EUR
        )

        if coste_prop <= 0:
            continue

        # Simular nueva capacidad efectiva
        nueva_abs_san = max(0.0, (medicos + m_prop) * MAX_HABITANTES_POR_MEDICO_AP - pob)
        nueva_abs_edu = max(0.0, ((plazas + p_prop) * (MAX_OCUPACION_ESCOLAR_PCT / 100.0) - alumnos) / 0.17)
        nueva_acogida_efectiva = min(nueva_abs_san, nueva_abs_edu, abs_viv_base)
        ganancia_neta = max(0, int(nueva_acogida_efectiva - acogida_efectiva_base))

        # Poda de eficiencia: descartar si ganancia es insignificante para el coste
        coste_por_persona = coste_prop / max(1, ganancia_neta)
        eficiencia = ganancia_neta / coste_prop

        # Si cuesta más de 10.000 € por persona acogida o ganancia es 0, no es eficiente
        if ganancia_neta <= 0 or coste_por_persona > 15000.0:
            factible = False
            motivo = f"Poda de eficiencia: ganancia marginal insuficiente ({ganancia_neta} hab, coste {coste_prop:.0f} €)"
        else:
            factible = True
            motivo = f"Resolución de cuello de botella en {factor_lim} y absorción de hasta {ganancia_neta} residentes"

        candidatos_evaluados.append({
            "row": row,
            "factible": factible,
            "motivo": motivo,
            "medicos_propuestos": m_prop,
            "plazas_propuestas": p_prop,
            "coste_propuesto": coste_prop,
            "ganancia_estimada": ganancia_neta,
            "eficiencia": eficiencia
        })

    # Ordenar candidatos factibles de mayor a menor eficiencia (personas acogibles por euro)
    candidatos_factibles = [c for c in candidatos_evaluados if c["factible"]]
    candidatos_factibles.sort(key=lambda c: c["eficiencia"], reverse=True)

    for item in candidatos_factibles:
        coste = item["coste_propuesto"]
        if presupuesto_restante >= coste:
            presupuesto_restante -= coste
            row = item["row"]
            m_asig = item["medicos_propuestos"]
            p_asig = item["plazas_propuestas"]
            ganancia = item["ganancia_estimada"]
            incremento_total_personas += ganancia

            # Recalcular nuevo estado e ICA con salvaguarda no compensatoria oficial
            pob = float(row["poblacion_actual"])
            medicos_tot = float(row["num_medicos_ap"]) + m_asig
            plazas_tot = float(row["num_plazas_escolares"]) + p_asig
            alumnos_act = float(row["num_alumnos_matriculados"])

            hab_med_nuevo = pob / medicos_tot
            ocup_esc_nueva = (alumnos_act / plazas_tot) * 100.0

            # Delta determinista proporcional al incremento relativo de dotación
            score_san_base = float(row.get("score_sanidad", 50.0))
            score_edu_base = float(row.get("score_educacion", 50.0))
            score_san_nuevo = min(100.0, score_san_base + (m_asig / max(1.0, medicos_tot)) * 25.0)
            score_edu_nuevo = min(100.0, score_edu_base + (p_asig / max(1.0, plazas_tot)) * 25.0)

            # ICA base recalculado deterministamente
            ica_raw = (
                0.30 * float(row.get("score_vivienda", 50.0)) +
                0.30 * ((score_san_nuevo + score_edu_nuevo) / 2.0) +
                0.25 * float(row.get("score_conectividad", 50.0)) +
                0.15 * float(row.get("score_holgura", 50.0))
            )

            eval_dict = {
                "hab_por_medico": hab_med_nuevo,
                "ocupacion_escolar_pct": ocup_esc_nueva,
                "indice_capacidad_global": ica_raw
            }
            salvaguarda = aplicar_salvaguarda_no_compensatoria(eval_dict)

            asignaciones.append(AsignacionMunicipio(
                codigo_ine=str(row["codigo_ine"]),
                municipio=str(row["nombre"]),
                comarca=str(row["comarca"]),
                medicos_asignados=m_asig,
                plazas_escolares_asignadas=p_asig,
                coste_eur=round(coste, 2),
                ganancia_personas=ganancia,
                nuevo_ica=round(salvaguarda["indice_capacidad_ajustado"], 2),
                semaforo=salvaguarda["semaforo_ica"],
                motivo_asignacion=item["motivo"]
            ))

    presupuesto_utilizado = float(presupuesto_total_eur) - presupuesto_restante

    # Contrato JSON unificado y estricto
    return {
        "comarca": comarca,
        "presupuesto_total_eur": float(presupuesto_total_eur),
        "presupuesto_utilizado_eur": round(presupuesto_utilizado, 2),
        "remanente_eur": round(presupuesto_restante, 2),
        "incremento_acogida_comarcal": int(incremento_total_personas),
        "municipios_evaluados": len(muni_comarca),
        "asignaciones": [
            {
                "codigo_ine": a.codigo_ine,
                "municipio": a.municipio,
                "comarca": a.comarca,
                "medicos_asignados": a.medicos_asignados,
                "plazas_escolares_asignadas": a.plazas_escolares_asignadas,
                "coste_eur": a.coste_eur,
                "ganancia_personas": a.ganancia_personas,
                "nuevo_ica": a.nuevo_ica,
                "semaforo": a.semaforo,
                "motivo_asignacion": a.motivo_asignacion
            }
            for a in asignaciones
        ],
        "metodologia": "Heurística voraz acotada por comarca con poda de ineficiencia residencial (MVP determinista)",
        "origen": TipoOrigen.SIMULADO.value,
        "metodo": "HEURISTICA_VORAZ_PRESUPUESTO_COMARCAL",
        "dataset_version": "v1.0.0-F2",
        "roadmap_fase_2": "Spillover territorial continuo mediante matriz de accesibilidad intermunicipal."
    }
