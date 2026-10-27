"""
ranking.py - Motor de ranking del perfil ciudadano (C2).

Convierte las preferencias de un ciudadano en una lista ordenada de municipios:

  1. Calcula los pesos del perfil (delegando en config/perfiles.py).
  2. Aplica los filtros duros (precio y tiempo); el colapso NO excluye, se marca.
  3. Calcula la afinidad como suma ponderada de los 4 sub-indices.
  4. Aplica el techo de la regla no compensatoria a la afinidad (39 puntos).
  5. Devuelve el desglose por dimension, la asequibilidad y el riesgo futuro.

Reglas que este modulo NO puede romper:
  - No modifica el ICA ni la regla no compensatoria.
  - No muta las filas de entrada.
  - No guarda datos personales: todo se calcula en memoria.

La formula (suma ponderada) es la unica fuente de verdad para el ranking. La
version SQL (oracle/ddl/04-perfiles_y_ranking.sql) debe replicarla exactamente;
tests/unit/test_perfiles_sql_sync.py vigila que los numeros coincidan.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.lodcore.config.perfiles import (
    DIMENSIONES,
    PreferenciasCiudadano,
    evaluar_asequibilidad,
    filtrar_municipios,
    pesos_ciudadano,
    presupuesto_efectivo_m2,
    variantes_pesos,
)
from src.lodcore.models.m4_capacidad.reglas_no_compensatorias import (
    aplicar_salvaguarda_no_compensatoria,
)
from src.lodcore.explain.redaccion_natural import verificar_ausencia_causalidad

# Techo de afinidad para un municipio con alerta no compensatoria.
# Debe ser UMBRAL_ICA_BAJA - 1 (coherente con reglas_no_compensatorias).
TECHO_AFINIDAD_ALERTA: float = 39.0

# Etiquetas legibles de cada dimension (para textos e interfaz).
ETIQUETAS_DIMENSION: Dict[str, str] = {
    "vivienda": "vivienda",
    "educacion": "educación",
    "conectividad": "conectividad",
    "sanidad": "sanidad",
}

# Etiquetas del factor limitante que llega del panel (en mayusculas).
ETIQUETAS_FACTOR: Dict[str, str] = {
    "SANIDAD": "sanidad",
    "EDUCACION": "educación",
    "EDUCACIÓN": "educación",
    "VIVIENDA": "vivienda",
}


def _scores(fila: Dict[str, Any]) -> Dict[str, float]:
    """Extrae los 4 sub-indices de una fila del panel."""
    return {d: float(fila[f"score_{d}"]) for d in DIMENSIONES}


def afinidad_ponderada(scores: Dict[str, float], pesos: Dict[str, float]) -> float:
    """Suma ponderada de los 4 sub-indices. Escala 0-100."""
    return float(sum(pesos[d] * scores[d] for d in DIMENSIONES))


def _riesgo_futuro(fila: Dict[str, Any]) -> Dict[str, Any]:
    """Riesgo demografico ya calculado en el panel (M3). No se recalcula aqui."""
    return {
        "poblacion_2031": fila.get("poblacion_2031"),
        "anio_estimado_saturacion": fila.get("anio_estimado_saturacion"),
        "alerta_planificacion": fila.get("alerta_planificacion"),
    }


def _fortalezas_y_carencias(scores: Dict[str, float]) -> Dict[str, List[str]]:
    """Dos dimensiones mas fuertes y dos mas debiles, con su puntuacion."""
    ordenadas = sorted(DIMENSIONES, key=lambda d: scores[d], reverse=True)
    return {
        "fortalezas": [f"{ETIQUETAS_DIMENSION[d]} ({scores[d]:.0f}/100)" for d in ordenadas[:2]],
        "carencias": [f"{ETIQUETAS_DIMENSION[d]} ({scores[d]:.0f}/100)" for d in ordenadas[-2:]],
    }


def _motivo_recomendacion(
    scores: Dict[str, float],
    pesos: Dict[str, float],
    alerta: bool,
    motivo_colapso: str | None,
    factor_limitante: str | None = None,
    gap_mercado_pct: float | None = None,
) -> str:
    """Texto en espanol sin atribuir causalidad."""
    ordenadas = sorted(DIMENSIONES, key=lambda d: pesos[d] * scores[d], reverse=True)
    mejor, segunda = ordenadas[0], ordenadas[1]
    peor = min(DIMENSIONES, key=lambda d: scores[d])
    base = (
        f"Encaja contigo por {ETIQUETAS_DIMENSION[mejor]} "
        f"({scores[mejor]:.0f}/100, el factor que más valoras) y "
        f"{ETIQUETAS_DIMENSION[segunda]} ({scores[segunda]:.0f}/100). "
        f"Su punto más débil es {ETIQUETAS_DIMENSION[peor]} ({scores[peor]:.0f}/100)."
    )

    factor = ETIQUETAS_FACTOR.get(str(factor_limitante).upper()) if factor_limitante else None
    if factor:
        base += f" El servicio que se rompe primero es la {factor}."

    if gap_mercado_pct is not None:
        if gap_mercado_pct <= -5.0:
            base += (
                f" El modelo valora el m² un {abs(gap_mercado_pct):.0f} % por encima "
                "del precio de referencia."
            )
        elif gap_mercado_pct >= 5.0:
            base += (
                f" El modelo valora el m² un {gap_mercado_pct:.0f} % por debajo "
                "del precio de referencia."
            )
        else:
            base += " El precio de referencia está en línea con el valor del modelo."

    if alerta:
        base += f" Atención: {motivo_colapso}"

    verificar_ausencia_causalidad(base)
    return base


def _medias_panel(filas: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Media de los 4 sub-indices en el panel, para comparar cada municipio."""
    valores: Dict[str, List[float]] = {d: [] for d in DIMENSIONES}
    for fila in filas:
        try:
            scores = _scores(fila)
        except (KeyError, TypeError, ValueError):
            continue
        for d in DIMENSIONES:
            valores[d].append(scores[d])
    return {d: (float(np.mean(v)) if v else 0.0) for d, v in valores.items()}


def rankear_municipios(
    filas: Iterable[Dict[str, Any]],
    prefs: PreferenciasCiudadano,
    top_k: int | None = None,
) -> Dict[str, Any]:
    """Ordena los municipios para un perfil ciudadano.

    Devuelve un diccionario con:
      - pesos: los pesos aplicados
      - ranking: lista ordenada (o top_k si se indica)
      - excluidos: municipios descartados con su motivo
      - n_aceptados / n_excluidos
    """
    filas_lista = list(filas)
    aceptados, excluidos = filtrar_municipios(filas_lista, prefs)
    pesos = pesos_ciudadano(prefs)
    medias = _medias_panel(filas_lista)

    ranking: List[Dict[str, Any]] = []
    for fila in aceptados:
        evaluado = aplicar_salvaguarda_no_compensatoria(dict(fila))
        scores = _scores(evaluado)
        desglose = {d: round(pesos[d] * scores[d], 2) for d in DIMENSIONES}
        afinidad_raw = sum(desglose.values())
        afinidad = (
            min(afinidad_raw, TECHO_AFINIDAD_ALERTA)
            if evaluado["alerta_no_compensatoria"]
            else afinidad_raw
        )
        desglose_pct = {
            d: (round(desglose[d] / afinidad_raw * 100, 1) if afinidad_raw > 0 else 0.0)
            for d in DIMENSIONES
        }
        gap_val = (
            float(evaluado["gap_mercado_pct"])
            if evaluado.get("gap_mercado_pct") is not None
            else None
        )

        asequibilidad = evaluar_asequibilidad(prefs, evaluado)
        texto = _fortalezas_y_carencias(scores)

        ranking.append(
            {
                "id_municipio": int(evaluado["id_municipio"]),
                "codigo_ine": str(evaluado.get("codigo_ine", "")),
                "nombre": str(evaluado["nombre"]),
                "comarca": str(evaluado.get("comarca", "")),
                "afinidad_pct": round(float(afinidad), 2),
                "desglose": desglose,
                "desglose_pct": desglose_pct,
                "comparacion_media": {
                    d: round(scores[d] - medias[d], 1) for d in DIMENSIONES
                },
                "semaforo_ica": evaluado["semaforo_ica"],
                "indice_ica_ajustado": evaluado["indice_capacidad_ajustado"],
                "alerta_no_compensatoria": bool(evaluado["alerta_no_compensatoria"]),
                "motivo_colapso": evaluado["motivo_colapso"],
                "factor_limitante": str(evaluado.get("factor_limitante", "")),
                "precio_m2": asequibilidad["precio_m2_evaluado"],
                "cuota_mensual_eur": asequibilidad["cuota_mensual_eur"],
                "entrada_eur": asequibilidad["entrada_eur"],
                "ingresos_necesarios_eur": asequibilidad["ingresos_necesarios_eur"],
                "precio_max_sostenible_m2": asequibilidad["precio_max_sostenible_m2"],
                "esfuerzo_financiero_pct": asequibilidad["esfuerzo_financiero_pct"],
                "etiqueta_esfuerzo": asequibilidad["etiqueta_esfuerzo"],
                "precio_avm_eur_m2": (
                    round(float(evaluado["precio_predicho_avm_m2"]), 2)
                    if evaluado.get("precio_predicho_avm_m2") is not None
                    else None
                ),
                "gap_mercado_pct": round(gap_val, 2) if gap_val is not None else None,
                "diagnostico_mercado": evaluado.get("diagnostico_mercado"),
                "fortalezas": texto["fortalezas"],
                "carencias": texto["carencias"],
                "riesgo_futuro": _riesgo_futuro(evaluado),
                "motivo_recomendacion": _motivo_recomendacion(
                    scores,
                    pesos,
                    bool(evaluado["alerta_no_compensatoria"]),
                    evaluado["motivo_colapso"],
                    factor_limitante=evaluado.get("factor_limitante"),
                    gap_mercado_pct=gap_val,
                ),
            }
        )

    # Determinista: afinidad, luego ICA ajustado, luego nombre.
    ranking.sort(key=lambda x: (-x["afinidad_pct"], -x["indice_ica_ajustado"], x["nombre"]))
    for i, item in enumerate(ranking, 1):
        item["posicion"] = i

    if top_k is not None:
        ranking = ranking[:top_k]

    return {
        "pesos": {d: round(pesos[d], 4) for d in DIMENSIONES},
        "presupuesto_efectivo_m2": round(presupuesto_efectivo_m2(prefs), 2),
        "ranking": ranking,
        "excluidos": excluidos,
        "n_aceptados": len(aceptados),
        "n_excluidos": len(excluidos),
    }


# ---------------------------------------------------------------------------
# Sensibilidad
# ---------------------------------------------------------------------------

def _rangos(valores: Sequence[float]) -> np.ndarray:
    """Rangos con empates promediados (1..n)."""
    arr = np.asarray(valores, dtype=float)
    orden = arr.argsort()
    rangos = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[orden[j + 1]] == arr[orden[i]]:
            j += 1
        rango_medio = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rangos[orden[k]] = rango_medio
        i = j + 1
    return rangos


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Correlacion de Spearman entre dos vectores alineados."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra = _rangos(a)
    rb = _rangos(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denom < 1e-12:
        return 1.0
    return float((ra * rb).sum() / denom)


def analisis_sensibilidad(
    filas: Iterable[Dict[str, Any]],
    prefs: PreferenciasCiudadano,
    delta: float = 0.20,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Mide si el ranking depende de los pesos.

    Perturba cada peso +-delta, recalcula la afinidad del mismo conjunto de
    municipios y compara el orden con el ranking base.
    """
    filas_lista = list(filas)
    aceptados, _ = filtrar_municipios(filas_lista, prefs)
    pesos_base = pesos_ciudadano(prefs)

    def _afinidades(pesos: Dict[str, float]) -> List[Tuple[str, float]]:
        pares: List[Tuple[str, float]] = []
        for fila in aceptados:
            evaluado = aplicar_salvaguarda_no_compensatoria(dict(fila))
            scores = _scores(evaluado)
            af = afinidad_ponderada(scores, pesos)
            if evaluado["alerta_no_compensatoria"]:
                af = min(af, TECHO_AFINIDAD_ALERTA)
            pares.append((str(evaluado["nombre"]), float(af)))
        pares.sort(key=lambda x: (-x[1], x[0]))
        return pares

    base = _afinidades(pesos_base)
    nombres_base = [n for n, _ in base]
    valores_base = [v for _, v in base]

    spearman_vals: List[float] = []
    solapamientos: List[float] = []
    detalle: List[Dict[str, Any]] = []

    for pesos_var in variantes_pesos(pesos_base, delta=delta):
        var = _afinidades(pesos_var)
        por_nombre = {n: v for n, v in var}
        valores_var = [por_nombre[n] for n in nombres_base]
        rho = _spearman(valores_base, valores_var)
        top_base = set(nombres_base[:top_k])
        top_var = set(n for n, _ in var[:top_k])
        solape = len(top_base & top_var) / float(top_k) if top_k else 1.0
        spearman_vals.append(rho)
        solapamientos.append(solape)
        detalle.append({"spearman": round(rho, 4), "top5_solape": round(solape, 3)})

    return {
        "n_variantes": len(spearman_vals),
        "spearman_min": round(min(spearman_vals), 4) if spearman_vals else 1.0,
        "spearman_media": round(float(np.mean(spearman_vals)), 4) if spearman_vals else 1.0,
        "top_k_solape_min": round(min(solapamientos), 3) if solapamientos else 1.0,
        "top_k_solape_media": round(float(np.mean(solapamientos)), 3) if solapamientos else 1.0,
        "detalle": detalle,
    }
