"""
pca_subindices.py - Reducción de dimensionalidad sin multicolinealidad mediante PCA y MinMaxScaler (0-100)
para los 4 sub-índices de capacidad e infraestructura (Modelo M2):
  1. Capacidad Sanitaria y Educativa
  2. Conectividad y Movilidad
  3. Sostenibilidad Ambiental y Territorial
  4. Músculo Presupuestario y Habitacional
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BLOQUES_M2 = {
    "subindice_sanidad_educacion": ["medicos_x1000", "centros_salud_x10k", "tasa_holgura_escolar", "centros_edu_x1000_alumnos"],
    "subindice_conectividad": ["cobertura_fibra_pct", "paradas_x1000", "score_conectividad"],
    "subindice_sostenibilidad": ["superficie_km2", "suelo_m2_x_hab"],
    "subindice_musculo_habitacional": ["vacias_x1000", "score_vivienda", "rentabilidad_bruta_alquiler_pct"]
}

def calcular_subindices_pca(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_out = df.copy()
    varianza_explicada: Dict[str, float] = {}

    for nombre_sub, cols in BLOQUES_M2.items():
        X_raw = df[cols].to_numpy(dtype=np.float64)
        X_std = StandardScaler().fit_transform(X_raw)
        pca = PCA(n_components=1, random_state=2026)
        pc1 = pca.fit_transform(X_std)

        # Garantizar orientación positiva con la primera variable de capacidad del bloque
        if np.corrcoef(pc1.ravel(), X_raw[:, 0])[0, 1] < 0:
            pc1 = -pc1

        scaled_0_100 = MinMaxScaler(feature_range=(0, 100)).fit_transform(pc1).ravel()
        df_out[nombre_sub] = np.round(scaled_0_100, 2)
        varianza_explicada[nombre_sub] = round(float(pca.explained_variance_ratio_[0]), 4)

    return df_out, varianza_explicada
