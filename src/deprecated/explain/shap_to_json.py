"""
shap_to_json.py - Cálculo de atribuciones aditivas normalizadas respecto a un background
fijo y auditable. Es una aproximación de Shapley de 2 órdenes (media de la contribución
marginal hacia delante y hacia atrás), NO valores Shapley exactos.
Garantiza aditividad antes del redondeo de publicación: sum(contribuciones) + base_value == prediction, con un residuo de redondeo menor que 1e-4 por variable. Registra el hash SHA-256 del background.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

def calcular_hash_background(background_dict: Dict[str, Any]) -> str:
    serializado = json.dumps(background_dict, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serializado).hexdigest()

def guardar_shap_background(
    features: List[str],
    X_bg: np.ndarray,
    base_prediction: float,
    feature_importances: np.ndarray,
    stds: np.ndarray,
    ruta: str
) -> Dict[str, Any]:
    payload_core = {
        "features": list(features),
        "feature_means": [round(float(x), 6) for x in np.mean(X_bg, axis=0)],
        "feature_stds": [round(float(max(s, 1e-6)), 6) for s in stds],
        "feature_importances": [round(float(w), 6) for w in feature_importances],
        "base_value": round(float(base_prediction), 6),
        "n_samples_background": int(X_bg.shape[0]),
        "metodo_atribucion": "EFECTOS_MARGINALES_SIMETRICOS_NORMALIZADOS",
        "dataset_version": "v1.0.0-F2"
    }
    bg_hash = calcular_hash_background(payload_core)
    payload_core["background_sha256"] = bg_hash

    p = Path(ruta)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload_core, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload_core

def explicar_instancia_a_json(
    model: Any,
    x_instance: np.ndarray,
    background_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calcula contribuciones marginales aditivas de cada variable respecto al background
    fijado, usando intervención condicional sobre las medias y normalización de eficiencia.
    Es una aproximación de 2 órdenes, no Shapley exacto (no enumera todos los subconjuntos).
    """
    features = background_data["features"]
    means = np.asarray(background_data["feature_means"], dtype=np.float64)
    base_val = float(background_data["base_value"])

    x_vec = np.asarray(x_instance, dtype=np.float64).reshape(1, -1)
    pred_full = float(model.predict(x_vec)[0])

    # Efecto marginal univariado e interactivo sobre la media del background
    efectos_crudos = []
    for i in range(len(features)):
        # 1) Qué cambia al poner la variable i en su valor observado partiendo del background medio
        x_from_bg = means.copy().reshape(1, -1)
        x_from_bg[0, i] = x_vec[0, i]
        delta_forward = float(model.predict(x_from_bg)[0]) - base_val

        # 2) Qué cambia al reemplazar la variable i en la instancia actual por la media del background
        x_leave_one = x_vec.copy()
        x_leave_one[0, i] = means[i]
        delta_backward = pred_full - float(model.predict(x_leave_one)[0])

        efectos_crudos.append(0.5 * (delta_forward + delta_backward))

    efectos_arr = np.asarray(efectos_crudos, dtype=np.float64)
    diff_total = pred_full - base_val
    suma_efectos = float(np.sum(efectos_arr))

    # Propiedad de eficiencia: sum(phi_i) == f(x) - E[f(X)]. La normalización
    # redistribuye las interacciones no capturadas por el enfoque de 2 órdenes.
    if abs(suma_efectos) > 1e-9:
        shap_vals = efectos_arr * (diff_total / suma_efectos)
    else:
        # Sin efectos netos no hay proporción que repartir; asignamos el residuo
        # a la variable de mayor efecto bruto para no romper la aditividad.
        shap_vals = np.zeros_like(efectos_arr)
        if abs(diff_total) > 1e-9 and len(efectos_arr):
            idx = int(np.argmax(np.abs(efectos_arr)))
            shap_vals[idx] = diff_total

    contribuciones = []
    for i, (feat, val, phi) in enumerate(zip(features, x_vec[0], shap_vals)):
        contribuciones.append({
            "feature": feat,
            "feature_value": round(float(val), 4),
            "background_mean": round(float(means[i]), 4),
            # Se mantiene la clave `shap_value` por compatibilidad con la API/UI;
            # el método real se declara en `metodo_atribucion`.
            "shap_value": round(float(phi), 4)
        })

    return {
        "base_value": round(base_val, 4),
        "prediction": round(pred_full, 4),
        "background_sha256": background_data.get("background_sha256", ""),
        "metodo_atribucion": "EFECTOS_MARGINALES_SIMETRICOS_NORMALIZADOS",
        "contributions": contribuciones
    }
