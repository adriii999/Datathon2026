"""
predict.py - Inferencia auditada del Modelo M4 (Capacidad de Acogida) integrando:
  1. Predicción y banda de incertidumbre al 95 %
  2. Reglas duras de no-compensatoriedad
  3. Atribuciones aditivas normalizadas sobre background fijo con SHA-256
     (aproximación de 2 órdenes; no son valores Shapley exactos)
  4. Explicación en lenguaje natural (español de España) libre de causalidad
"""

import json
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np

from src.lodcore.models.m4_capacidad.train import FEATURES_M4, M4CapacidadModel
from src.lodcore.models.m4_capacidad.reglas_no_compensatorias import aplicar_salvaguarda_no_compensatoria
from src.lodcore.explain.shap_to_json import explicar_instancia_a_json
from src.lodcore.explain.redaccion_natural import redactar_explicacion_shap

def predecir_capacidad_municipio(
    fila_muni: Dict[str, Any],
    model_dir: str = "src/lodcore/models/m4_capacidad"
) -> Dict[str, Any]:
    dir_p = Path(model_dir)
    model: M4CapacidadModel = joblib.load(dir_p / "model.joblib")
    bg_data = json.loads((dir_p / "shap_background.json").read_text(encoding="utf-8"))

    x_vec = np.array([[float(fila_muni[f]) for f in FEATURES_M4]], dtype=np.float32)
    banda = model.uncertainty_95(x_vec)[0]

    # Copia de trabajo con la predicción del modelo para pasar por la capa no compensatoria
    evaluado = dict(fila_muni)
    evaluado["indice_capacidad_global"] = banda["prediccion"]
    evaluado = aplicar_salvaguarda_no_compensatoria(evaluado)

    # Atribuciones aditivas sobre el background persistido
    shap_info = explicar_instancia_a_json(model, x_vec[0], bg_data)

    # Redacción en español sin causalidad
    explicacion_es = redactar_explicacion_shap(
        municipio=str(fila_muni.get("nombre", "Municipio")),
        # El bruto es al que suman las contribuciones; el ajustado refleja el techo duro.
        score_predicho=banda["prediccion"],
        ic_95_inf=banda["ic_95_inferior"],
        ic_95_sup=banda["ic_95_superior"],
        base_value=shap_info["base_value"],
        contribuciones=shap_info["contributions"],
        alerta_no_compensatoria=evaluado["alerta_no_compensatoria"],
        motivo_colapso=evaluado.get("motivo_colapso") or "",
        score_ajustado=evaluado["indice_capacidad_ajustado"]
    )

    return {
        "id_municipio": int(fila_muni.get("id_municipio", 0)),
        "nombre": str(fila_muni.get("nombre", "")),
        "score_raw_modelo": banda["prediccion"],
        "indice_capacidad_ajustado": evaluado["indice_capacidad_ajustado"],
        "ic_95_inferior": banda["ic_95_inferior"],
        "ic_95_superior": banda["ic_95_superior"],
        "semaforo_ica": evaluado["semaforo_ica"],
        "alerta_no_compensatoria": evaluado["alerta_no_compensatoria"],
        "motivo_colapso": evaluado["motivo_colapso"],
        "shap": shap_info,
        "explicacion_natural_es": explicacion_es,
        "origen": banda["origen"],
        "metodo": "GBR_ATRIBUCIONES_ADITIVAS_NON_COMPENSATORY",
        "dataset_version": banda["dataset_version"]
    }
