"""
redaccion_natural.py - Generación de explicaciones en español de España sin vocabulario causal.
Regla transversal: las atribuciones describen el modelo ("el modelo asocia"), nunca causalidad física.
"""

from typing import Dict, Any, List
import re

TERMINOS_CAUSALES_PROHIBIDOS = [
    r"\bcausa\b",
    r"\bcausan\b",
    r"\bcausado\b",
    r"\bprovoca\b",
    r"\bprovocan\b",
    r"\bproduce\b",
    r"\bproducen\b",
    r"\bgenera\b",
    r"\bgeneran\b",
    r"\borigina\b"
]

ETIQUETAS_LEGIBLES = {
    "score_sanidad": "el sub-índice de cobertura sanitaria de Atención Primaria",
    "score_educacion": "el sub-índice de holgura en plazas escolares",
    "score_conectividad": "el sub-índice de conectividad y transporte público (CRTM)",
    "score_vivienda": "el sub-índice de disponibilidad de suelo y vivienda",
    "hab_por_medico": "la ratio de habitantes por facultativo médico",
    "ocupacion_escolar_pct": "el porcentaje de ocupación de centros educativos",
    "stock_vivienda_vacia_est": "el parque estimado de vivienda desocupada",
    "suelo_urbano_disponible_m2": "la superficie de suelo urbano vacante (m²)"
}

def verificar_ausencia_causalidad(texto: str) -> bool:
    """
    Audita un texto garantizando que no contenga ningún verbo causal prohibido.
    Lanza ValueError si detecta vocabulario causal.
    """
    for patron in TERMINOS_CAUSALES_PROHIBIDOS:
        if re.search(patron, texto, flags=re.IGNORECASE):
            raise ValueError(f"VIOLACIÓN DE EXPLICABILIDAD: Se detectó término causal prohibido ('{patron}') en: {texto}")
    return True

def redactar_explicacion_shap(
    municipio: str,
    score_predicho: float,
    ic_95_inf: float,
    ic_95_sup: float,
    base_value: float,
    contribuciones: List[Dict[str, Any]],
    alerta_no_compensatoria: bool = False,
    motivo_colapso: str = "",
    score_ajustado: float | None = None
) -> str:
    """
    Genera un párrafo explicativo en español de España riguroso y libre de afirmaciones causales.

    `score_predicho` es la salida bruta del modelo (a la que suman las contribuciones);
    `score_ajustado` es el valor tras aplicar la regla dura, si difiere.
    """
    ordenadas_pos = sorted([c for c in contribuciones if c["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)
    ordenadas_neg = sorted([c for c in contribuciones if c["shap_value"] < 0], key=lambda x: x["shap_value"])

    if (
        alerta_no_compensatoria
        and score_ajustado is not None
        and score_ajustado < score_predicho - 0.01
    ):
        frase_principal = (
            f"En la evaluación de {municipio}, el modelo estima un índice bruto de capacidad de "
            f"{score_predicho:.1f} puntos (intervalo de confianza al 95 %: "
            f"[{ic_95_inf:.1f}, {ic_95_sup:.1f}]) respecto al valor medio del background de "
            f"referencia ({base_value:.1f} puntos); la regla dura de no-compensatoriedad lo "
            f"ajusta a {score_ajustado:.1f} puntos."
        )
    else:
        frase_principal = (
            f"En la evaluación de {municipio}, el modelo estima un índice de capacidad de "
            f"{score_predicho:.1f} puntos (intervalo de confianza al 95 %: "
            f"[{ic_95_inf:.1f}, {ic_95_sup:.1f}]) respecto al valor medio del background de "
            f"referencia ({base_value:.1f} puntos)."
        )

    partes = [frase_principal]

    if ordenadas_pos:
        top_p = ordenadas_pos[0]
        nombre_p = ETIQUETAS_LEGIBLES.get(top_p["feature"], top_p["feature"])
        partes.append(
            f"El modelo asocia una aportación positiva de +{top_p['shap_value']:.2f} puntos con {nombre_p} "
            f"(valor registrado: {top_p['feature_value']:.2f})."
        )

    if ordenadas_neg:
        top_n = ordenadas_neg[0]
        nombre_n = ETIQUETAS_LEGIBLES.get(top_n["feature"], top_n["feature"])
        partes.append(
            f"En sentido inverso, el modelo asocia una penalización marginal de {top_n['shap_value']:.2f} puntos con {nombre_n} "
            f"(valor registrado: {top_n['feature_value']:.2f})."
        )

    if alerta_no_compensatoria and motivo_colapso:
        partes.append(
            f"Adicionalmente, se activa la regla dura de límite no compensatorio ({motivo_colapso}), "
            f"fijando el semáforo de acogida en nivel BAJA con independencia del resto de factores."
        )

    texto_final = " ".join(partes)
    verificar_ausencia_causalidad(texto_final)
    return texto_final
