"""
predict.py - Inferencia con banda de incertidumbre al 95 % y cálculo de esfuerzo familiar (Modelo M1).
"""

from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from src.lodcore.models.m1_avm.train import FEATURES_M1, M1AVMModel

def predecir_avm_municipio(
    fila_muni: Dict[str, Any],
    model_dir: str = "src/lodcore/models/m1_avm"
) -> Dict[str, Any]:
    """
    Predice el precio justo por m² junto con su banda de incertidumbre al 95 %,
    el gap frente al precio observado y la tasa de esfuerzo hipotecario familiar.
    """
    model_path = Path(model_dir) / "model.joblib"
    model: M1AVMModel = joblib.load(model_path)

    x_vec = np.array([[float(fila_muni[f]) for f in FEATURES_M1]], dtype=np.float32)
    banda = model.uncertainty_95(x_vec)[0]

    precio_obs = float(fila_muni.get("precio_m2_compra", banda["prediccion"]))
    gap_pct = ((precio_obs - banda["prediccion"]) / max(banda["prediccion"], 1.0)) * 100.0

    # Esfuerzo financiero familiar estimado (vivienda tipo 90 m², hipoteca a 25 años al 3.2% sobre renta media 34.500€)
    renta_anual_hogar = float(fila_muni.get("renta_media_hogar", 34500.0))
    cuota_anual_est = (precio_obs * 90.0 * 0.80) * 0.058
    esfuerzo_familiar_pct = round((cuota_anual_est / max(renta_anual_hogar, 1.0)) * 100.0, 2)

    return {
        "id_municipio": int(fila_muni.get("id_municipio", 0)),
        "nombre": str(fila_muni.get("nombre", "")),
        "precio_observado_m2": round(precio_obs, 2),
        "precio_justo_avm_m2": banda["prediccion"],
        "ic_95_inferior": banda["ic_95_inferior"],
        "ic_95_superior": banda["ic_95_superior"],
        "gap_mercado_pct": round(gap_pct, 2),
        "esfuerzo_familiar_pct": esfuerzo_familiar_pct,
        "origen": banda["origen"],
        "metodo": banda["metodo"],
        "dataset_version": banda["dataset_version"]
    }
