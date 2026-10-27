"""
isocronas.py - Heurística del criterio 'Municipio de 30 minutos'.

No calcula rutas reales sobre GTFS/OSM: aproxima el tiempo al polo de empleo
a partir del tiempo a Madrid y de la densidad de paradas. Las rutas GTFS/OSM
son una mejora pendiente, no una funcionalidad ya implementada.
"""

from typing import Dict, Any
import pandas as pd
from src.lodcore.config.thresholds import TIEMPO_MAXIMO_POLO_EMPLEO_MIN

def evaluar_isocronas_30min(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    # Tiempo en vehículo privado hacia el polo metropolitano / hospital más próximo (aprox. 48% del tiempo a Puerta del Sol)
    df_out["tiempo_polo_empleo_privado_min"] = (df_out["tiempo_medio_madrid_minutos"] * 0.48).round(1)
    # Tiempo en transporte público interurbano: factor 1.18 si hay alta densidad de paradas (>8 por 1.000 hab.) y 1.38 si no.
    factor_tp = df_out["paradas_x1000"].apply(lambda p: 1.18 if p > 8.0 else 1.38)
    df_out["tiempo_polo_empleo_publico_min"] = (df_out["tiempo_polo_empleo_privado_min"] * factor_tp).round(1)

    df_out["cumple_isocrona_30min_privado"] = df_out["tiempo_polo_empleo_privado_min"] <= TIEMPO_MAXIMO_POLO_EMPLEO_MIN
    df_out["cumple_isocrona_30min_publico"] = df_out["tiempo_polo_empleo_publico_min"] <= TIEMPO_MAXIMO_POLO_EMPLEO_MIN
    return df_out
