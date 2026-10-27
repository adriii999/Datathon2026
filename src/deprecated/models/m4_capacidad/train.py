"""
train.py - Entrenamiento del Modelo M4 (Core: Capacidad de Acogida + atribuciones
aditivas + No-Compensatoriedad), persistencia auditable de `shap_background.json`
y exportación ONNX para Oracle Database.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.lodcore.config import cargar_params
from src.lodcore.models.base import BaseLoDCOREModel, comparar_equivalencia_onnx
from src.lodcore.explain.shap_to_json import guardar_shap_background
from src.lodcore.publish.uncertainty_band import empaquetar_prediccion_con_banda
from src.lodcore.data.provenance import TipoOrigen

FEATURES_M4 = [
    "score_sanidad",
    "score_educacion",
    "score_conectividad",
    "score_vivienda",
    "hab_por_medico",
    "ocupacion_escolar_pct",
    "stock_vivienda_vacia_est",
    "suelo_urbano_disponible_m2"
]

class M4CapacidadModel(BaseLoDCOREModel):
    model_name = "M4_CAPACIDAD_ACOGIDA"
    features = FEATURES_M4

    def __init__(self, n_estimators: int = 120, learning_rate: float = 0.05, max_depth: int = 3, random_state: int = 2026) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> Dict[str, Any]:
        X_f32 = np.asarray(X, dtype=np.float32)
        y_f32 = np.asarray(y, dtype=np.float32)

        if groups is not None and len(np.unique(groups)) >= 4:
            splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
            splits = list(splitter.split(X_f32, y_f32, groups=groups))
            cv_scheme = "Spatial_GroupKFold_Comarca"
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            splits = list(splitter.split(X_f32, y_f32))
            cv_scheme = "KFold_5"

        fold_maes = []
        fold_rmses = []
        oof_preds = np.zeros_like(y_f32, dtype=np.float64)

        for tr_idx, val_idx in splits:
            gbr_cv = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            gbr_cv.fit(X_f32[tr_idx], y_f32[tr_idx])
            p_val = gbr_cv.predict(X_f32[val_idx])
            oof_preds[val_idx] = p_val
            fold_maes.append(float(mean_absolute_error(y_f32[val_idx], p_val)))
            fold_rmses.append(float(root_mean_squared_error(y_f32[val_idx], p_val)))

        self.estimator = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.estimator.fit(X_f32, y_f32)
        full_preds = self.estimator.predict(X_f32)
        self.residual_std = max(1.5, float(np.std(y_f32 - oof_preds)))

        importances = {
            feat: round(float(imp), 4)
            for feat, imp in zip(self.features, self.estimator.feature_importances_)
        }

        return {
            "model_name": self.model_name,
            "algorithm": "GradientBoostingRegressor",
            "cv_scheme": cv_scheme,
            "n_samples": int(len(y_f32)),
            "n_features": len(self.features),
            "r2_in_sample": round(float(r2_score(y_f32, full_preds)), 4),
            "r2_cv_oof": round(float(r2_score(y_f32, oof_preds)), 4),
            "mae_cv_mean": round(float(np.mean(fold_maes)), 2),
            "mae_cv_std": round(float(np.std(fold_maes)), 2),
            "rmse_cv_mean": round(float(np.mean(fold_rmses)), 2),
            "rmse_cv_std": round(float(np.std(fold_rmses)), 2),
            "residual_std": round(self.residual_std, 2),
            "feature_importances": importances,
            "dataset_version": "v1.0.0-F2",
            "origen": TipoOrigen.ESTIMADO.value,
            "metodo": "GBR_ATRIBUCIONES_ADITIVAS_NON_COMPENSATORY_ONNX",
            "advertencia": (
                "El target (ICA) es una combinación determinista de los sub-índices: "
                "r2_in_sample=1.0 no es evidencia de generalización. La métrica de "
                "referencia es r2_cv_oof con GroupKFold por comarca."
            ),
        }

    def uncertainty_95(self, X: np.ndarray) -> List[Dict[str, Any]]:
        preds = self.predict(X)
        return [
            empaquetar_prediccion_con_banda(
                float(p),
                self.residual_std,
                origen=TipoOrigen.ESTIMADO,
                metodo="GBR_RESIDUAL_CI95",
                dataset_version="v1.0.0-F2"
            )
            for p in preds
        ]


def run_training_m4(
    data_path: str = "data/validated/municipios_panel_validado.csv",
    out_dir: str = "src/lodcore/models/m4_capacidad",
    oracle_onnx_dir: str = "oracle/onnx"
) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    X = df[FEATURES_M4].to_numpy(dtype=np.float32)
    y = df["indice_capacidad_global"].to_numpy(dtype=np.float32)
    groups = df["comarca"].to_numpy()

    _p = cargar_params().get("models", {}).get("m4_capacidad", {})
    model = M4CapacidadModel(
        n_estimators=int(_p.get("n_estimators", 120)),
        learning_rate=float(_p.get("learning_rate", 0.05)),
        max_depth=int(_p.get("max_depth", 3)),
        random_state=int(_p.get("random_state", 2026)),
    )
    metrics = model.train(X, y, groups=groups)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    onnx_file = model.to_onnx(str(out_p / "model.onnx"), n_features=len(FEATURES_M4))
    joblib.dump(model, out_p / "model.joblib")

    # Persistir shap_background.json fijado y auditable con SHA-256
    mean_vec = np.mean(X, axis=0, keepdims=True)
    base_pred = float(model.predict(mean_vec)[0])
    stds = np.std(X, axis=0)
    bg_payload = guardar_shap_background(
        features=FEATURES_M4,
        X_bg=X,
        base_prediction=base_pred,
        feature_importances=model.estimator.feature_importances_,
        stds=stds,
        ruta=str(out_p / "shap_background.json")
    )
    metrics["shap_background_sha256"] = bg_payload["background_sha256"]

    # Verificar equivalencia real: scikit-learn nativo vs binario ONNX
    preds_py = model.predict(X)
    preds_onnx = model.predict_onnx(onnx_file, X)
    metrics.update(comparar_equivalencia_onnx(preds_py, preds_onnx))

    (out_p / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    oracle_p = Path(oracle_onnx_dir)
    oracle_p.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_file, oracle_p / "m4_capacidad.onnx")

    return metrics

if __name__ == "__main__":
    m = run_training_m4()
    print(f"[M4 Capacidad] Entrenado y exportado a ONNX. MAE CV: {m['mae_cv_mean']} | Hash atribuciones: {m['shap_background_sha256'][:12]}...")
