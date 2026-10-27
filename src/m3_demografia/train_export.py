"""M3 · Demografía — entrenamiento, export ONNX y validación.

Salida:
    oracle/models/m3_demografia.onnx
    oracle/models/m3_demografia.json      (ficha del modelo)
    doc/modelos/m3_demografia.json        (copia para el jurado)

Contrato del modelo:
    ENTRADA  (16 floats, FEATURES_M3, en ese orden)
    SALIDA   1 float = log( poblacion(anio_destino) / poblacion(anio_base) )
             El consumidor aplica EXP() para obtener el factor.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import onnxruntime as ort
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from src.common.dataset import (
    ANIO_BASE,
    DIR_DOC,
    DIR_MODELOS,
    XLSX,
    cargar_municipios,
    cargar_panel,
    sha256_archivo,
)
from src.m3_demografia.features import FEATURES_M3, construir_dataset_m3, X_y

MODELO = "m3_demografia"
ONNX_PATH = DIR_MODELOS / f"{MODELO}.onnx"
TARGET_OPSET = 17


def _export_onnx(model, n_features: int) -> bytes:
    """Export con initial_types explícitos para que la ValueInfo declare
    la talla real de la entrada (None, n) y no una inferida de una muestra."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onx = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=TARGET_OPSET,
    )
    # Parche de metadata: declara la talla real de salida (N,1) en la ValueInfo
    # para que la carga en Oracle no encuentre dims vacías.
    import onnx

    salida = onx.graph.output[0]
    tipo = salida.type.tensor_type
    del tipo.shape.dim[:]
    origen = tipo.shape.dim.add()
    origen.dim_param = "N"
    destino = tipo.shape.dim.add()
    destino.dim_value = 1
    aux = ONNX_PATH.parent / f"{MODELO}.tmp.onnx"
    aux.write_bytes(onx.SerializeToString())
    data = aux.read_bytes()
    aux.unlink()
    return data


def entrenar() -> dict:
    panel = cargar_panel()
    municipios = cargar_municipios()
    data = construir_dataset_m3(panel, municipios)

    X, y = X_y(data)
    grupos = data["cod_ine"].values

    # Hold-out por MUNICIPIO (el municipio nunca puede estar en train y test)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026).split(X, y, grupos))
    Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=2026,
    )
    model.fit(Xtr, ytr)

    # Métricas: en log y en % (error del factor)
    def errores(X_, y_):
        pred = model.predict(X_)
        err_log = pred - y_.values
        factor_real = np.exp(y_.values)
        factor_sim = np.exp(pred)
        err_pct = (factor_sim - factor_real) / factor_real * 100
        return err_log, err_pct

    err_te_log, err_te_pct = errores(Xte, yte)
    err_tr_log, err_tr_pct = errores(Xtr, ytr)

    # Export + validación ONNX <-> sklearn a 1e-5 relativo en TODOS los datos
    X_all = X.values.astype(np.float32)
    onx_bytes = _export_onnx(model, len(FEATURES_M3))

    dir_modeles = ONNX_PATH.parent
    dir_modeles.mkdir(parents=True, exist_ok=True)
    ONNX_PATH.write_bytes(onx_bytes)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {input_name: X_all.astype(np.float32)})[0].ravel()
    pred_skl = model.predict(X_all.astype(np.float32))
    max_diff = float(np.max(np.abs(pred_onnx - pred_skl)))
    escala = float(np.max(np.abs(pred_skl)))
    check_onnx = max_diff / (escala + 1e-12)
    assert check_onnx < 1e-5, f"ONNX difiere del modelo sklearn: rel={check_onnx:.2e}"

    # Texto simple de validación: error mediano del factor por horizonte
    data_te = data.iloc[te]
    factor_real = np.exp(yte.values)
    factor_sim = np.exp(model.predict(Xte))
    err_pct_h = {
        int(h): float(np.median(np.abs((factor_sim[data_te["horizonte"].values == h] - factor_real[data_te["horizonte"].values == h]) / factor_real[data_te["horizonte"].values == h])) * 100)
        for h in sorted(data_te["horizonte"].unique())
    }

    ficha = {
        "modelo": MODELO,
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "version_sklearn": sklearn.__version__,
        "dataset": {
            "fichero": XLSX.name,
            "sha256": sha256_archivo(XLSX),
            "filas_entrenamiento": int(len(Xtr)),
            "filas_test": int(len(Xte)),
            "split": "GroupShuffleSplit por municipio (test 20 %, semilla 2026)",
        },
        "tarea": "Factor de crecimiento poblacional a horizonte h",
        "contrato_salida": "log(factor); el consumidor aplica EXP()",
        "features": FEATURES_M3,
        "orden_features": "posicional",
        "parametros": model.get_params(),
        "metricas": {
            "mae_log_test": float(np.abs(err_te_log).mean()),
            "error_absoluto_mediano_pct_test": float(np.median(np.abs(err_te_pct))),
            "mae_log_train": float(np.abs(err_tr_log).mean()),
            "error_absoluto_mediano_pct_train": float(np.median(np.abs(err_tr_pct))),
            "error_absoluto_mediano_pct_por_horizonte": err_pct_h,
        },
        "validacion_onnx": {
            "max_abs_diff": max_diff,
            "rep_ok": bool(np.allclose(pred_onnx, pred_skl, rtol=1e-5, atol=1e-7)),
        },
        "nota_horizonte": (
            "Solo hay labels observadas hasta 2025. Pronósticos a horizontes > 5 "
            "desde 2025 son extrapolación (el modelo no los vio concretamente); "
            "hasta 10 años se asumen válidos por el diseño de la task."
        ),
    }

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    (DIR_MODELOS / f"{MODELO}.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dir_fichas_doc = DIR_DOC / "modelos"
    dir_fichas_doc.mkdir(parents=True, exist_ok=True)
    (dir_fichas_doc / f"{MODELO}.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[M3] muestras={len(X)} (train={len(Xtr)}, test={len(Xte)})")
    print(f"[M3] MAE log test = {ficha['metricas']['mae_log_test']:.4f} "
          f"| error mediano del factor (test) = {ficha['metricas']['error_absoluto_mediano_pct_test']:.2f} %")
    print(f"[M3] ONNX generado y validado: {ONNX_PATH}")
    return ficha


if __name__ == "__main__":
    entrenar()
