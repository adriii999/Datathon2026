"""
train.py - Entrenamiento del Modelo M5 (Clustering no supervisado de tipologías municipales K-Means)
y exportación de `clustering.json` y `metrics.json`.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.lodcore.config import cargar_params
from src.lodcore.data.provenance import TipoOrigen

FEATURES_M5_CLUSTER = [
    "score_sanidad",
    "score_educacion",
    "score_conectividad",
    "score_vivienda",
    "tiempo_medio_madrid_minutos",
    "precio_m2_compra",
    "densidad"
]

# Los nombres son una interpretación humana de los centroides, no una propiedad
# que K-Means descubra. Se mantiene el mapeo para k=4 (el valor del MVP) y se
# exportan los centroides en `clustering.json` para poder auditarlos. Si se cambia
# k, se usan nombres genéricos para no etiquetar mal los clusters.
NOMBRES_ARQUETIPOS = {
    0: "Nodos Metropolitanos y Corredores Dinámicos",
    1: "Cabeceras Comarcales Equilibradas",
    2: "Pueblos Conectados de Calidad Ambiental y Teletrabajo",
    3: "Reserva de Oportunidad Residencial y Suelo Asequible"
}

def run_training_m5(
    data_path: str = "data/validated/municipios_panel_validado.csv",
    out_dir: str = "src/lodcore/models/m5_recomendador"
) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    X_raw = df[FEATURES_M5_CLUSTER].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    _p = cargar_params().get("models", {}).get("m5_recomendador", {})
    k = int(_p.get("n_clusters", 4))
    random_state = int(_p.get("random_state", 2026))
    if k < 2:
        raise ValueError(f"n_clusters debe ser >= 2 (recibido {k})")
    if k >= len(df):
        raise ValueError(f"n_clusters ({k}) debe ser menor que el número de municipios ({len(df)})")

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = kmeans.fit_predict(X_std)
    sil_score = float(silhouette_score(X_std, labels))

    asignaciones = []
    # `labels` está alineado posicionalmente con las filas; usar la etiqueta del
    # índice (idx) fallaría si el CSV no trae un RangeIndex 0..n-1.
    for pos, (_, row) in enumerate(df.iterrows()):
        cid = int(labels[pos])
        tipologia = (
            NOMBRES_ARQUETIPOS.get(cid, f"Tipología {cid}")
            if k == len(NOMBRES_ARQUETIPOS)
            else f"Tipología {cid}"
        )
        asignaciones.append({
            "id_municipio": int(row["id_municipio"]),
            "codigo_ine": int(row["codigo_ine"]),
            "nombre": str(row["nombre"]),
            "cluster_id": cid,
            "tipologia_m5": tipologia,
            "arquetipo_actual": str(row.get("arquetipo_nombre", ""))
        })

    metrics = {
        "model_name": "M5_CLUSTERING_MCDM_COSINE",
        "algorithm": f"KMeans_k{k}_WeightedCosine",
        "n_clusters": k,
        "n_clusters_efectivos": int(len(np.unique(labels))),
        "silhouette_score": round(sil_score, 4),
        "inertia": round(float(kmeans.inertia_), 2),
        "n_municipios": int(len(df)),
        "dataset_version": "v1.0.0-F2",
        "origen": TipoOrigen.ESTIMADO.value,
        "metodo": "KMEANS_STD_SILHOUETTE"
    }

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "clustering.json").write_text(
        json.dumps(
            {
                "clusters": asignaciones,
                "centroids_features": FEATURES_M5_CLUSTER,
                # Centroides en la escala estandarizada usada por K-Means: permiten
                # auditar a qué arquetipo se asignó cada cluster_id.
                "centroids": {
                    str(cid): [round(float(v), 4) for v in centro]
                    for cid, centro in enumerate(kmeans.cluster_centers_)
                },
                "cluster_sizes": {
                    str(cid): int(count)
                    for cid, count in zip(*np.unique(labels, return_counts=True))
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8"
    )
    (out_p / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics

if __name__ == "__main__":
    m = run_training_m5()
    print(f"[M5 Recomendador] Clustering completado. Silhouette score: {m['silhouette_score']}")
