"""M4 · Capacidad / Vitalidad — entrenamiento, export ONNX y validación.

Salida:
    oracle/models/m4_capacidad.onnx
    oracle/models/m4_capacidad.json      (ficha del modelo)
    doc/modelos/m4_capacidad.json        (copia para el jurado)

Contrato del modelo:
    ENTRADA  (14 floats, FEATURES_M4, en ese orden)
    SALIDA   5 floats = (dim_dem, dim_viv, dim_ser, dim_acc, dim_emp) en [0, 100]
    El índice y el factor limitante NO salen del modelo: se computan en SQL
    con los pesos fijos 0,20/0,15/0,25/0,25/0,15 y el argmin de las dims.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import onnxruntime as ort
import sklearn
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.common.dataset import (
    DIR_DOC,
    DIR_MODELOS,
    XLSX,
    cargar_municipios,
    cargar_panel,
    sha256_archivo,
)
from src.m4_capacidad.features import (
    FEATURES_M4,
    TARGETS_M4,
    construir_dataset_m4,
    indice,
    X_y,
)

MODELO = "m4_capacidad"
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
    # Parche: skl2onnx deja la ValueInfo del output multi-output como (N,1)
    # aunque el grafo emite (N,5). Se corrige la metadata para que el grafo
    # declare su talla real (clave para cargarse en Oracle).
    import onnx

    salida = onx.graph.output[0]
    tipo = salida.type.tensor_type
    del tipo.shape.dim[:]
    origen = tipo.shape.dim.add()
    origen.dim_param = "N"
    destino = tipo.shape.dim.add()
    destino.dim_value = len(TARGETS_M4)
    aux = ONNX_PATH.parent / f"{MODELO}.tmp.onnx"
    aux.write_bytes(onx.SerializeToString())
    data = aux.read_bytes()
    aux.unlink()
    return data


def entrenar() -> dict:
    panel = cargar_panel()
    municipios = cargar_municipios()
    data = construir_dataset_m4(panel, municipios)
    imputacion = data.attrs.get("imputacion_tiempo", {})
    n_anios = data.groupby("anio").size()

    df = data[data["anio"] == 2025].set_index("cod_ine")
    df["indice"] = indice(df)
    print("[M4] saneo del corte 2025 (índice calculado con los pesos de la app)")
    print(df["indice"].describe().to_string())

    X, y = X_y(data)
    # Split por filas (estados): lo que interesa puntuar son estados INTERPOLADOS
    tr, te = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=2026, shuffle=True
    )
    Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=2026,
    )
    model.fit(Xtr, ytr)

    # ── Métricas por dimensión ──
    pred_te = model.predict(Xte)
    metricas = {}
    for j, dim in enumerate(TARGETS_M4):
        real, sim = yte[dim].values, pred_te[:, j]
        rho = spearmanr(real, sim).statistic if np.std(sim) > 0 else 0.0
        metricas[dim] = {
            "mae": float(mean_absolute_error(real, sim)),
            "spearman": float(rho),
        }

    # ── Export + validación ONNX <-> sklearn ──
    X_all = X.values.astype(np.float32)
    onx_bytes = _export_onnx(model, len(FEATURES_M4))
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    ONNX_PATH.write_bytes(onx_bytes)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {input_name: X_all.astype(np.float32)})[0]
    pred_skl = model.predict(X_all.astype(np.float32))
    max_diff = float(np.max(np.abs(pred_onnx - pred_skl)))
    assert np.allclose(pred_onnx, pred_skl, rtol=1e-5, atol=1e-6), (
        f"ONNX difiere del modelo sklearn: max_diff={max_diff:.2e}"
    )

    ficha = {
        "modelo": MODELO,
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "version_sklearn": sklearn.__version__,
        "dataset": {
            "fichero": XLSX.name,
            "sha256": sha256_archivo(XLSX),
            "filas_entrenamiento": int(len(Xtr)),
            "filas_test": int(len(Xte)),
            "estados": "155 municipios x años 2010-2025",
            "split": "filas aleatorias 80/20 (semilla 2026)",
        },
        "tarea": "Puntuar un estado municipal: 5 dimensiones (0-100)",
        "contrato_salida": "(dim_dem, dim_viv, dim_ser, dim_acc, dim_emp); "
        "índice y factor limitante se calculan en SQL",
        "features": FEATURES_M4,
        "targets": TARGETS_M4,
        "pesos_indice": {"dim_dem": 0.2, "dim_viv": 0.15, "dim_ser": 0.25, "dim_acc": 0.25, "dim_emp": 0.15},
        "parametros": model.get_params(),
        "metricas_test": metricas,
        "validacion_onnx": {"max_abs_diff": max_diff},
        "imputacion_tiempo_centro": imputacion,
        "declaraciones": [
            "Targets = percent-rank dentro de la cohorte de cada año replicando PV_SIMULA",
            "Stock de vivienda por año: unidades_urbanas_residencial (única serie de stock en el xlsx)",
            "escuelas, tiempo_centro y distancia son estáticos por municipio",
        ],
    }

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    (DIR_MODELOS / f"{MODELO}.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dir_doc = DIR_DOC / "modelos"
    dir_doc.mkdir(parents=True, exist_ok=True)
    (dir_doc / f"{MODELO}.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[M4] estados={len(X)} (train={len(Xtr)}, test={len(Xte)})")
    print(f"[M4] estados por año={n_anios.min()}..{n_anios.max()} (años: {n_anios.index.min()}..{n_anios.index.max()})")
    print(f"[M4] corte 2025={len(df)} municipios (índice 2025: {df['indice'].mean():.1f} de media)")
    for dim, rr in metricas.items():
        print(f"[M4]   {dim}: MAE={rr['mae']:.3f} Spearman={rr['spearman']:.3f}")
    print(f"[M4] ONNX generado y validado: {ONNX_PATH}")
    return ficha


if __name__ == "__main__":
    entrenar()
