"""M3 · Demografía — construcción de features para la predicción del factor
de crecimiento poblacional a horizonte h.

Concepto: cada muestra es un "momento de decisión" (municipio, año base t,
horizonte h). Todas las features provienen SOLO de información conocida en t
(serie de población, estructura demográfica, paro, migración, accesibilidad).
El target es el crecimiento logarítmico observado hasta t+h.

Semántica de la salida del modelo:  target = log( poblacion(t+h) / poblacion(t) ).
En SQL el consumidor aplica EXP(PREDICTION(...)) para obtener el factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features finales, en el ORDEN EXACTO en que consume el ONNX.
FEATURES_M3 = [
    "log_pob",
    "cagr_1",
    "cagr_3",
    "cagr_5",
    "cagr_10",
    "vol_5",
    "edad_media",
    "pct_menor_15",
    "pct_65mas",
    "reemplazamiento",
    "paro_100",
    "var_paro",
    "migracion_3y",
    "distancia_sol_km",
    "tiempo_centro_min",
    "horizonte",
]

TARGET_M3 = "log_factor"


def _crecimientos(serie: pd.Series) -> pd.Series:
    return serie.pct_change()


def construir_dataset_m3(panel: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por (municipio, año base t, horizonte h).

    Requisitos de ventana: cagr_10 exige serie completa desde 2010, por lo que
    t empieza en 2020; h llega hasta 2025 (año del último dato observado).
    """
    anios_obs = range(2010, 2026)
    serie_pop = (
        panel.pivot(index="cod_ine", columns="anio", values="poblacion_total")
        .sort_index()
        .sort_index(axis=1)
        .loc[:, anios_obs]
    )
    assert not serie_pop.isna().any().any() and serie_pop.shape[0] > 0

    panel_idx = panel.set_index(["cod_ine", "anio"])

    # Estática municipal (se une a cada muestra)
    estatica = municipios[["cod_ine", "distancia_centro_km", "tiempo_centro_min"]].rename(
        columns={"distancia_centro_km": "distancia_sol_km"}
    )

    filas = []
    for cod, pob in serie_pop.iterrows():
        crec = _crecimientos(pob)          # crecimiento año a año
        logp = np.log(pob)
        est = estatica.loc[estatica["cod_ine"] == cod].iloc[0]
        for t in range(2020, 2026):        # primer año con las 4 cagr completas
            if t + 1 > 2025:
                break                     # no hay horizonte de 1 año observable
            h_max = 2025 - t
            feats_base = {
                "log_pob": logp[t],
                "cagr_1": crec[t] / 1,
                "cagr_3": ((pob[t] / pob[t - 3]) ** (1 / 3) - 1),
                "cagr_5": ((pob[t] / pob[t - 5]) ** (1 / 5) - 1),
                "cagr_10": ((pob[t] / pob[t - 10]) ** (1 / 10) - 1),
                "vol_5": crec.loc[t - 4 : t].std(ddof=1),
                "edad_media": panel_idx.loc[(cod, t), "edad_media"],
                "pct_menor_15": panel_idx.loc[(cod, t), "pct_poblacion_menor_15"],
                "pct_65mas": panel_idx.loc[(cod, t), "pct_poblacion_65_mas"],
                "reemplazamiento": panel_idx.loc[(cod, t), "proporcion_reemplazamiento"],
                "paro_100": panel_idx.loc[(cod, t), "paro_por_100_hab"],
                "var_paro": panel_idx.loc[(cod, t), "variacion_relativa_paro"],
                "migracion_3y": panel.loc[
                    (panel["cod_ine"] == cod) & (panel["anio"].between(t - 2, t)),
                    "tasa_migracion",
                ].mean(),
                "distancia_sol_km": est["distancia_sol_km"],
                "tiempo_centro_min": est["tiempo_centro_min"],
            }
            for h in range(1, h_max + 1):
                fila = dict(feats_base)
                fila["horizonte"] = h
                fila["cod_ine"] = cod
                fila["anio_base"] = t
                fila["anio_destino"] = t + h
                fila[TARGET_M3] = logp[t + h] - logp[t]
                filas.append(fila)

    df = pd.DataFrame(filas)
    # El target de un horizonte concreto no depende de features del futuro; si
    # alguna feature faltara, esa muestra sale (y el aviso queda en la ficha).
    df = df.dropna(subset=FEATURES_M3 + [TARGET_M3]).reset_index(drop=True)
    return df


def X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES_M3].astype(float), df[TARGET_M3].astype(float)
