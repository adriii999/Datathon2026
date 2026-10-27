"""
base.py - Interfaz abstracta común para los 5 modelos de LoDCORE Madrid.
Define el contrato estricto: train, predict, uncertainty_95 y to_onnx.
Incluye alineación de precisión float32 en nodos de árboles y un contrato de
equivalencia numérica verificable: error relativo < 1e-6 y absoluto < 1e-3.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# El binario ONNX devuelve tensores float32. A magnitudes de precio (≈2.000 €/m²)
# la resolución de float32 es ≈1,2e-4, por lo que exigir un error ABSOLUTO < 1e-6
# es físicamente imposible. El contrato verificable es el error RELATIVO < 1e-6,
# con una cota absoluta de cordura de 1e-3 para detectar divergencias reales.
ONNX_REL_TOL = 1e-6
ONNX_ABS_TOL = 1e-3

_ONNX_SESSIONS: Dict[str, ort.InferenceSession] = {}

def comparar_equivalencia_onnx(
    pred_python: np.ndarray,
    pred_onnx: np.ndarray,
    rel_tol: float = ONNX_REL_TOL,
    abs_tol: float = ONNX_ABS_TOL
) -> Dict[str, Any]:
    """
    Compara la predicción nativa de scikit-learn con la del binario ONNX.

    Devuelve max_abs_diff, max_rel_diff y el veredicto del contrato:
    error relativo < 1e-6 y error absoluto < 1e-3.
    """
    py = np.asarray(pred_python, dtype=np.float64).ravel()
    ox = np.asarray(pred_onnx, dtype=np.float64).ravel()
    if py.shape != ox.shape:
        raise ValueError(f"Formas incompatibles: Python {py.shape} vs ONNX {ox.shape}")
    abs_diff = np.abs(py - ox)
    rel_diff = abs_diff / np.maximum(np.abs(py), 1e-12)
    max_abs = float(np.max(abs_diff)) if len(abs_diff) else 0.0
    max_rel = float(np.max(rel_diff)) if len(rel_diff) else 0.0
    return {
        "onnx_max_abs_diff": max_abs,
        "onnx_max_rel_diff": max_rel,
        "onnx_rel_tolerance": rel_tol,
        "onnx_abs_tolerance": abs_tol,
        "onnx_equivalence_passed": bool(max_rel < rel_tol and max_abs < abs_tol),
    }

def align_estimator_float32(estimator: Any) -> None:
    """
    Alinea en memoria los umbrales y valores hoja de ensembles de árboles de scikit-learn
    a precisión IEEE 754 float32 (idéntica a los tensores del grafo ONNX), eliminando la
    deriva de redondeo float64 -> float32. La equivalencia se mide con
    `comparar_equivalencia_onnx` (relativa < 1e-6, absoluta < 1e-3).
    """
    if hasattr(estimator, "estimators_"):
        estimators = np.asarray(estimator.estimators_).ravel()
        for sub in estimators:
            if hasattr(sub, "tree_"):
                sub.tree_.threshold[:] = sub.tree_.threshold.astype(np.float32).astype(np.float64)
                sub.tree_.value[:] = sub.tree_.value.astype(np.float32).astype(np.float64)
    if hasattr(estimator, "init_") and hasattr(estimator.init_, "constant_"):
        estimator.init_.constant_[:] = estimator.init_.constant_.astype(np.float32).astype(np.float64)


class BaseLoDCOREModel(ABC):
    model_name: str = "base_model"
    features: List[str] = []

    def __init__(self) -> None:
        self.estimator: Any = None
        self.residual_std: float = 1.0

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Entrena el modelo con validación cruzada espacial/anidada y devuelve métricas."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicción nativa del estimador de scikit-learn (float32 de entrada, suma float64).

        No debe delegar en ONNX: si lo hiciera, el test de equivalencia compararía
        ONNX contra ONNX y nunca detectaría una divergencia real con el modelo Python.
        Para inferencia ONNX existe `predict_onnx()`.
        """
        if self.estimator is None:
            raise RuntimeError(f"El modelo '{self.model_name}' no ha sido entrenado ni cargado.")
        X_f32 = np.asarray(X, dtype=np.float32)
        return np.asarray(self.estimator.predict(X_f32), dtype=np.float64).ravel()

    @abstractmethod
    def uncertainty_95(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """Devuelve predicciones acompañadas de su banda de incertidumbre al 95 % y procedencia."""
        raise NotImplementedError

    def to_onnx(self, output_path: str, n_features: int) -> str:
        """
        Exporta el estimador entrenado a formato estándar ONNX (opset 15) compatible con
        Oracle Database 23ai (DBMS_VECTOR.LOAD_ONNX_MODEL) y ONNX Runtime.
        """
        if self.estimator is None:
            raise RuntimeError(f"No se puede exportar '{self.model_name}' a ONNX sin entrenar.")
        align_estimator_float32(self.estimator)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = to_onnx(self.estimator, initial_types=initial_type, target_opset=15)
        onnx_bytes = onnx_model.SerializeToString()
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(onnx_bytes)
        return str(dest)

    @staticmethod
    def predict_onnx(onnx_path: str, X: np.ndarray) -> np.ndarray:
        """Ejecuta inferencia sobre el binario ONNX usando CPUExecutionProvider (sesión cacheada)."""
        ruta = str(Path(onnx_path).resolve())
        sess = _ONNX_SESSIONS.get(ruta)
        if sess is None:
            sess = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
            _ONNX_SESSIONS[ruta] = sess
        input_name = sess.get_inputs()[0].name
        X_f32 = np.asarray(X, dtype=np.float32)
        outputs = sess.run(None, {input_name: X_f32})
        return np.asarray(outputs[0], dtype=np.float64).ravel()
