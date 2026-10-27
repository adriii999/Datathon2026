"""
train.py - Ejecución de proyecciones demográficas y persistencia de métricas de M3.
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.lodcore.models.m3_demografia.project import proyectar_demografia_municipio
from src.lodcore.data.provenance import TipoOrigen

def run_training_m3(
    data_path: str = "data/validated/municipios_panel_validado.csv",
    out_dir: str = "src/lodcore/models/m3_demografia"
) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    proyecciones = [proyectar_demografia_municipio(row.to_dict()) for _, row in df.iterrows()]

    pob_2025_total = sum(p["poblacion_base_2025"] for p in proyecciones)
    pob_2030_total = sum(p["proyeccion_5y_2030"]["prediccion"] for p in proyecciones)
    pob_2035_total = sum(p["proyeccion_10y_2035"]["prediccion"] for p in proyecciones)
    en_riesgo_10pct = sum(1 for p in proyecciones if p["supera_capacidad_en_estres_10pct"])

    metrics = {
        "model_name": "M3_DEMOGRAFIA_EXTRAPOLACION_GEOMETRICA",
        "n_municipios": len(proyecciones),
        "poblacion_agregada_2025": int(pob_2025_total),
        "poblacion_proyectada_2030": round(float(pob_2030_total), 0),
        "poblacion_proyectada_2035": round(float(pob_2035_total), 0),
        "municipios_saturados_bajo_estres_10pct": int(en_riesgo_10pct),
        "confidence_level": 0.95,
        "dataset_version": "v1.0.0-F2",
        "origen": TipoOrigen.PROYECTADO.value,
        "metodo": "EXTRAPOLACION_GEOMETRICA_CALIBRADA_CI95",
        "advertencia": (
            "Proyección determinista calibrada con la proyección padronal "
            "intermedia de 2031. No es un modelo SARIMAX ni Prophet."
        ),
    }

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_p / "projections_2030_2035.json").write_text(
        json.dumps(proyecciones, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics

if __name__ == "__main__":
    m = run_training_m3()
    print(f"[M3 Demografía] Proyección 2030: {m['poblacion_proyectada_2030']:.0f} hab | Saturados en estrés +10%: {m['municipios_saturados_bajo_estres_10pct']}")
