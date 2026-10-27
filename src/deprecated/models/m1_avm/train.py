"""
train.py - Entrenamiento, validación cruzada espacial (GroupKFold por comarca) y exportación ONNX
del Modelo M1: Automated Valuation Model (AVM) y Esfuerzo Financiero Familiar.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.lodcore.config import cargar_params
from src.lodcore.models.base import BaseLoDCOREModel, comparar_equivalencia_onnx
from src.lodcore.publish.uncertainty_band import calcular_banda_95_ensemble
from src.lodcore.data.provenance import TipoOrigen

FEATURES_M1 = [
    "tiempo_medio_madrid_minutos",
    "densidad",
    "cobertura_fibra_pct",
    "paradas_x1000",
    "vacias_x1000",
    "medicos_x1000",
    "tasa_holgura_escolar"
]

class M1AVMModel(BaseLoDCOREModel):
    model_name = "M1_AVM_HEDONICO"
    features = FEATURES_M1

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, random_state: int = 2026) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> Dict[str, Any]:
        X_f32 = np.asarray(X, dtype=np.float32)
        y_f32 = np.asarray(y, dtype=np.float32)

        # Validación espacial por comarca (GroupKFold) o 5-Fold si no hay grupos
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

        min_samples_leaf = int(
            cargar_params().get("models", {}).get("m1_avm", {}).get("min_samples_leaf", 2)
        )
        for train_idx, val_idx in splits:
            rf_cv = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state
            )
            rf_cv.fit(X_f32[train_idx], y_f32[train_idx])
            preds_val = rf_cv.predict(X_f32[val_idx])
            oof_preds[val_idx] = preds_val
            fold_maes.append(float(mean_absolute_error(y_f32[val_idx], preds_val)))
            fold_rmses.append(float(root_mean_squared_error(y_f32[val_idx], preds_val)))

        # Ajuste final sobre el conjunto completo validado
        self.estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state
        )
        self.estimator.fit(X_f32, y_f32)
        full_preds = self.estimator.predict(X_f32)
        self.residual_std = float(np.std(y_f32 - oof_preds))

        importances = {
            feat: round(float(imp), 4)
            for feat, imp in zip(self.features, self.estimator.feature_importances_)
        }

        metrics = {
            "model_name": self.model_name,
            "algorithm": "RandomForestRegressor",
            "cv_scheme": cv_scheme,
            "n_samples": int(len(y_f32)),
            "n_features": len(self.features),
            "r2_in_sample": round(float(r2_score(y_f32, full_preds)), 4),
            "r2_cv_oof": round(float(r2_score(y_f32, oof_preds)), 4),
            "mae_cv_mean": round(float(np.mean(fold_maes)), 2),
            "mae_cv_std": round(float(np.std(fold_maes)), 2),
            "rmse_cv_mean": round(float(np.mean(fold_rmses)), 2),
            "rmse_cv_std": round(float(np.std(fold_rmses)), 2),
            "feature_importances": importances,
            "dataset_version": "v1.0.0-F2",
            "origen": TipoOrigen.ESTIMADO.value,
            "metodo": "RANDOM_FOREST_SPATIAL_CV_ONNX",
            "advertencia": (
                "Con n=20 y 8 comarcas el R² fuera de muestra es inestable y puede ser "
                "negativo. M1 es un módulo ilustrativo, no un AVM validado."
            ),
        }
        return metrics

    def uncertainty_95(self, X: np.ndarray) -> List[Dict[str, Any]]:
        if self.estimator is None:
            raise RuntimeError("El modelo M1 no está entrenado.")
        X_f32 = np.asarray(X, dtype=np.float32)
        # Predicciones individuales de los árboles para construir el intervalo al 95%
        tree_preds = np.column_stack([tree.predict(X_f32) for tree in self.estimator.estimators_])
        resultados = []
        for row_trees in tree_preds:
            resultados.append(
                calcular_banda_95_ensemble(
                    row_trees,
                    origen=TipoOrigen.ESTIMADO,
                    metodo="RF_TREE_QUANTILE_95",
                    dataset_version="v1.0.0-F2"
                )
            )
        return resultados


def run_training_m1(
    data_path: str = "data/validated/municipios_panel_validado.csv",
    out_dir: str = "src/lodcore/models/m1_avm",
    oracle_onnx_dir: str = "oracle/onnx"
) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    X = df[FEATURES_M1].to_numpy(dtype=np.float32)
    y = df["precio_m2_compra"].to_numpy(dtype=np.float32)
    groups = df["comarca"].to_numpy()

    _p = cargar_params().get("models", {}).get("m1_avm", {})
    model = M1AVMModel(
        n_estimators=int(_p.get("n_estimators", 100)),
        max_depth=int(_p.get("max_depth", 5)),
        random_state=int(_p.get("random_state", 2026)),
    )
    metrics = model.train(X, y, groups=groups)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    onnx_file = model.to_onnx(str(out_p / "model.onnx"), n_features=len(FEATURES_M1))
    joblib.dump(model, out_p / "model.joblib")

    # Verificar equivalencia real: scikit-learn nativo vs binario ONNX
    preds_py = model.predict(X)
    preds_onnx = model.predict_onnx(onnx_file, X)
    metrics.update(comparar_equivalencia_onnx(preds_py, preds_onnx))

    (out_p / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Copiar a oracle/onnx/ para carga directa con DBMS_VECTOR.LOAD_ONNX_MODEL
    oracle_p = Path(oracle_onnx_dir)
    oracle_p.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_file, oracle_p / "m1_avm.onnx")

    return metrics

if __name__ == "__main__":
    m = run_training_m1()
    print(f"[M1 AVM] Entrenado y exportado a ONNX. MAE CV: {m['mae_cv_mean']} €/m² | Max diff ONNX: {m['onnx_max_abs_diff']:.2e}")
