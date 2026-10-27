"""
train.py - Ejecución y persistencia de métricas para el Modelo M2 (PCA + Isócronas 30 min).
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.lodcore.models.m2_indicadores.pca_subindices import calcular_subindices_pca
from src.lodcore.models.m2_indicadores.isocronas import evaluar_isocronas_30min
from src.lodcore.data.provenance import TipoOrigen

def run_training_m2(
    data_path: str = "data/validated/municipios_panel_validado.csv",
    out_dir: str = "src/lodcore/models/m2_indicadores"
) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    df_pca, varianza = calcular_subindices_pca(df)
    df_full = evaluar_isocronas_30min(df_pca)

    metrics = {
        "model_name": "M2_PCA_HEURISTICA_30MIN",
        "n_municipios": int(len(df_full)),
        # Varianza explicada por el PC1 de cada bloque (escala 0-1), y su media.
        "explained_variance_ratio_pc1": round(float(sum(varianza.values()) / len(varianza)), 4),
        "explained_variance_por_bloque": varianza,
        "municipios_30min_privado": int(df_full["cumple_isocrona_30min_privado"].sum()),
        "municipios_30min_publico": int(df_full["cumple_isocrona_30min_publico"].sum()),
        "dataset_version": "v1.0.0-F2",
        "origen": TipoOrigen.ESTIMADO.value,
        "metodo": "PCA_MINMAX_0_100_HEURISTICA_TIEMPO_PARADAS"
    }

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics

if __name__ == "__main__":
    m = run_training_m2()
    print(f"[M2 Indicadores] Completado. Varianza media explicada por PC1: {m['explained_variance_ratio_pc1']}")
