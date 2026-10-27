"""
recommend.py - Motor de recomendación personalizada mediante similitud del coseno ponderada (MCDM)
y filtrado de salvaguarda no compensatoria (Modelo M5).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.lodcore.models.m4_capacidad.reglas_no_compensatorias import aplicar_salvaguarda_no_compensatoria
from src.lodcore.data.provenance import TipoOrigen

def calcular_similitud_coseno_ponderada(
    vec_muni: np.ndarray,
    vec_ideal: np.ndarray,
    pesos: np.ndarray
) -> float:
    w_sqrt = np.sqrt(np.maximum(pesos, 0.0))
    a = vec_muni * w_sqrt
    b = vec_ideal * w_sqrt
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)

def recomendar_municipios_mcdm(
    df: pd.DataFrame,
    peso_sanidad: float = 0.25,
    peso_educacion: float = 0.25,
    peso_conectividad: float = 0.25,
    peso_vivienda: float = 0.25,
    presupuesto_max_m2: float = 2500.0,
    tiempo_max_madrid_min: int = 90,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Calcula el ranking personalizado por similitud del coseno ponderada sobre los sub-índices 0-100,
    penalizando aquellos municipios que excedan presupuesto, tiempo de desplazamiento o activen
    la regla dura no compensatoria.
    """
    pesos = np.array([peso_sanidad, peso_educacion, peso_conectividad, peso_vivienda], dtype=np.float64)
    if np.sum(pesos) <= 0:
        pesos = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    pesos = pesos / np.sum(pesos)

    vec_ideal = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    candidatos: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        m_dict = row.to_dict()
        m_dict = aplicar_salvaguarda_no_compensatoria(m_dict)

        vec_m = np.array([
            float(m_dict["score_sanidad"]),
            float(m_dict["score_educacion"]),
            float(m_dict["score_conectividad"]),
            float(m_dict["score_vivienda"])
        ], dtype=np.float64)

        cos_sim = calcular_similitud_coseno_ponderada(vec_m, vec_ideal, pesos)
        # Escalar magnitud y dirección coseno
        magnitud = float(np.dot(vec_m, pesos)) / 100.0
        afinidad_raw = (0.55 * cos_sim + 0.45 * magnitud) * 100.0

        # Penalizaciones por restricciones del ciudadano
        precio = float(m_dict["precio_m2_compra"])
        tiempo = int(m_dict["tiempo_medio_madrid_minutos"])
        if precio > presupuesto_max_m2:
            afinidad_raw -= min(30.0, ((precio - presupuesto_max_m2) / 50.0))
        if tiempo > tiempo_max_madrid_min:
            afinidad_raw -= min(25.0, (tiempo - tiempo_max_madrid_min) * 0.6)
        if m_dict["alerta_no_compensatoria"]:
            afinidad_raw = min(afinidad_raw, 39.0)

        afinidad_final = round(max(10.0, min(99.5, afinidad_raw)), 2)

        candidatos.append({
            "id_municipio": int(m_dict["id_municipio"]),
            "codigo_ine": int(m_dict["codigo_ine"]),
            "nombre": str(m_dict["nombre"]),
            "comarca": str(m_dict["comarca"]),
            "afinidad_mcdm_pct": afinidad_final,
            "similitud_coseno": round(cos_sim, 4),
            "indice_ica_ajustado": m_dict["indice_capacidad_ajustado"],
            "semaforo_ica": m_dict["semaforo_ica"],
            "alerta_no_compensatoria": m_dict["alerta_no_compensatoria"],
            "precio_m2_compra": round(precio, 2),
            "tiempo_medio_madrid_minutos": tiempo,
            "personas_acogida_efectiva": int(m_dict["personas_acogida_efectiva"]),
            "factor_limitante": str(m_dict["factor_limitante"]),
            "arquetipo_nombre": str(m_dict.get("arquetipo_nombre", "")),
            "origen": TipoOrigen.ESTIMADO.value,
            "metodo": "MCDM_WEIGHTED_COSINE_SIMILARITY",
            "dataset_version": "v1.0.0-F2"
        })

    candidatos.sort(key=lambda x: (x["afinidad_mcdm_pct"], x["indice_ica_ajustado"]), reverse=True)
    return candidatos[:top_k]
