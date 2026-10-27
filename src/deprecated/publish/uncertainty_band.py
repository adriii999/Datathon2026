"""
uncertainty_band.py - Empaquetado obligatorio de predicciones con banda de incertidumbre 95% y procedencia.
Regla transversal de LoDCORE Madrid: ninguna predicción se publica sin `origen`, `metodo` y banda 95%.
"""

from typing import Dict, Any, Sequence
import numpy as np
from src.lodcore.data.provenance import TipoOrigen, validar_procedencia_registro

def calcular_banda_95_ensemble(
    predicciones_arboles: np.ndarray,
    origen: TipoOrigen = TipoOrigen.ESTIMADO,
    metodo: str = "ENSEMBLE_QUANTILE_95",
    dataset_version: str = "v1.0.0-F2"
) -> Dict[str, Any]:
    """
    A partir de la matriz de predicciones de cada árbol del ensemble (shape: [n_trees, n_samples] o [n_trees]),
    calcula media/punto e intervalo de confianza al 95% [p2.5, p97.5].
    """
    arr = np.asarray(predicciones_arboles, dtype=np.float64)
    media = float(np.mean(arr))
    std = float(np.std(arr))
    # Percentiles empíricos de los árboles combinados con el margen normal al 95 %.
    # (No es un intervalo t-Student: se usa 1,96 y los percentiles empíricos como suelo.)
    ic_inferior = float(min(np.percentile(arr, 2.5), media - 1.96 * std))
    ic_superior = float(max(np.percentile(arr, 97.5), media + 1.96 * std))

    paquete = {
        "prediccion": round(media, 2),
        "ic_95_inferior": round(ic_inferior, 2),
        "ic_95_superior": round(ic_superior, 2),
        "incertidumbre_relativa_pct": round(((ic_superior - ic_inferior) / max(abs(media), 1e-6)) * 100.0, 2),
        "nivel_confianza": 0.95,
        "origen": origen.value if isinstance(origen, TipoOrigen) else str(origen),
        "metodo": metodo,
        "dataset_version": dataset_version
    }
    validar_procedencia_registro(paquete)
    return paquete

def empaquetar_prediccion_con_banda(
    valor_puntual: float,
    error_estandar: float,
    origen: TipoOrigen = TipoOrigen.ESTIMADO,
    metodo: str = "PARAMETRIC_CI_95",
    dataset_version: str = "v1.0.0-F2"
) -> Dict[str, Any]:
    """
    Empaqueta una predicción escalar junto a su error estándar para garantizar banda al 95%.
    """
    margen = 1.96 * abs(float(error_estandar))
    paquete = {
        "prediccion": round(float(valor_puntual), 2),
        "ic_95_inferior": round(float(valor_puntual) - margen, 2),
        "ic_95_superior": round(float(valor_puntual) + margen, 2),
        "incertidumbre_relativa_pct": round((2 * margen / max(abs(float(valor_puntual)), 1e-6)) * 100.0, 2),
        "nivel_confianza": 0.95,
        "origen": origen.value if isinstance(origen, TipoOrigen) else str(origen),
        "metodo": metodo,
        "dataset_version": dataset_version
    }
    validar_procedencia_registro(paquete)
    return paquete
