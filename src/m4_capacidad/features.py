"""M4 · Capacidad / Vitalidad — construcción de estados y targets.

Cada muestra es un ESTADO municipal en el año t (municipio, anio), con:
  - FEATURES: ratios de saturación "crudos" del estado (sin percent-rank),
    en unidades interpretables. Así el modelo puede puntuar estados
    SIMULADOS que no existen en la realidad (trabajo del what-if).
  - TARGETS: las 5 dimensiones del índice, definidas como en la app APEX
    (percent-rank dentro de la cohorte de los 155 municipios DE ESE AÑO,
    de la misma forma que lo hace PV_SIMULA sobre el corte actual).

Declaraciones (ficha del modelo):
  * El stock de vivienda por año usa unidades_urbanas_residencial (única
    magnitud de stock urbano vivienda con serie histórica en el xlsx; en la
    app el stock administrativo es viviendas_total y es estático).
  * escuelas (centros en ALTA), tiempo al centro, distancia y modo son
    estáticos por municipio: su por-1000 solo módula por población.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features finales, en el ORDEN EXACTO en que consume el ONNX.
FEATURES_M4 = [
    "log_pob",
    "salud_10k",
    "escuelas_1000",
    "paradas_1000",
    "telefono_1000",
    "viv_1000",
    "crec_10y_pct",
    "pct_menor_15",
    "pct_65mas",
    "reemplazamiento",
    "paro_100",
    "var_paro",
    "tiempo_centro_min",
    "distancia_sol_km",
]

# Targets (salidas del modelo multi-output, en ese orden)
TARGETS_M4 = ["dim_dem", "dim_viv", "dim_ser", "dim_acc", "dim_emp"]

PESOS = {"dim_dem": 0.20, "dim_viv": 0.15, "dim_ser": 0.25, "dim_acc": 0.25, "dim_emp": 0.15}


def _percent_rank(s: pd.Series, invertir: bool = False) -> pd.Series:
    """Réplica de Oracle PERCENT_RANK(): (rank-1)/(n-1)."""
    r = s.rank(method="average").sub(1).div(max(len(s) - 1, 1))
    return (1 - r) if invertir else r


def _estatica(municipios: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Estática municipal + imputación de tiempo_centro_min.

    99 de los 155 municipios no tienen modo de transporte registrado en el
    xlsx ('modo_mas_rapido' nulo) y por tanto tiempo_centro_min nulo. Se
    imputan con una aproximación lineal tiempo = a + b·distancia ajustada
    SOLO con los municipios que sí tienen dato. La regla es declarada y
    replicable en SQL (coeficientes van en la ficha del modelo).
    """
    df = municipios.rename(columns={"distancia_centro_km": "distancia_sol_km"})
    mask = df["tiempo_centro_min"].notna()
    a, b = 0.0, 0.0
    if mask.sum() >= 2 and (~mask).sum() > 0:
        x = df.loc[mask, "distancia_sol_km"].values
        y = df.loc[mask, "tiempo_centro_min"].values
        b, a = np.polyfit(x, y, deg=1)
        a, b = float(a), float(b)
        df.loc[~mask, "tiempo_centro_min"] = np.maximum(
            a + b * df.loc[~mask, "distancia_sol_km"].values, 0.0
        )
    imputacion = {
        "n_imputados": int((~mask).sum()),
        "a": a,
        "b": b,
        "regla": "tiempo_centro_min = max(0, a + b·distancia_sol_km) si falta",
        "municipios_con_dato": int(mask.sum()),
    }
    return df[["cod_ine", "distancia_sol_km", "tiempo_centro_min", "centros_educativos_alta"]], imputacion


def construir_dataset_m4(panel: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    serie = (
        panel.pivot(index="cod_ine", columns="anio", values="poblacion_total")
        .sort_index()
        .sort_index(axis=1)
    )

    estados = []
    for (cod, anio), g in panel.groupby(["cod_ine", "anio"], sort=True):
        f = g.iloc[0]
        pob = float(f["poblacion_total"])
        if pob <= 0:
            continue
        estados.append(
            {
                "cod_ine": cod,
                "anio": int(anio),
                "nombre": f["municipio"],
                "poblacion": pob,
                "log_pob": float(np.log(pob)),
                "salud_10k": (f["centros_de_salud"] + f["consultorios_locales"]) / pob * 10_000,
                "paradas_1000": (f["paradas_autobus"] + f["estaciones_metro"] + f["estaciones_cercanias"]) / pob * 1_000,
                "telefono_1000": (f["lineas_telefonicas_baf"] + f["lineas_telefonicas_atf"]) / pob * 1_000,
                "viv_1000": f["unidades_urbanas_residencial"] / pob * 1_000,
                "pct_menor_15": f["pct_poblacion_menor_15"],
                "pct_65mas": f["pct_poblacion_65_mas"],
                "reemplazamiento": f["proporcion_reemplazamiento"],
                "paro_100": f["paro_por_100_hab"],
                "var_paro": f["variacion_relativa_paro"],
            }
        )

    estatica, imputacion_tiempo = _estatica(municipios)
    df = pd.DataFrame(estados).merge(estatica, on="cod_ine", how="left")
    df.attrs["imputacion_tiempo"] = imputacion_tiempo
    df["escuelas_1000"] = df["centros_educativos_alta"] / df["poblacion"] * 1_000

    # Saneo declarado (convención de la capa real): indicador sin dato = 0.
    # Sin esto los aannos 2010-2011 desaparecerían del panel.
    df[["salud_10k", "paradas_1000", "telefono_1000", "var_paro", "paro_100"]] = df[
        ["salud_10k", "paradas_1000", "telefono_1000", "var_paro", "paro_100"]
    ].fillna(0.0)

    # ── crec_10y_pct: CAGR anualizado en la ventana disponible hasta t (≥ 4 años) ──
    def crec(cod: str, t: int):
        s = serie.loc[cod]
        start = max(s.index.min(), t - 10)
        ventana = [a for a in s.index if start <= a <= t]
        if len(ventana) < 4:
            return None
        p0, p1 = s[ventana[0]], s[ventana[-1]]
        w = ventana[-1] - ventana[0]
        return (p1 / p0) ** (1 / w) - 1 if p0 > 0 else None

    df["crec_10y_pct"] = [crec(c, t) for c, t in zip(df["cod_ine"], df["anio"])]

    df = df.dropna(subset=FEATURES_M4).reset_index(drop=True)

    # ── TARGETS: 5 dims con percent-rank DENTRO del corte de cada año ──
    partes = []
    for _, g in df.groupby("anio", sort=True):
        g = g.copy()
        r = lambda col, inv=False: _percent_rank(g[col], invertir=inv)
        g["dim_dem"] = (r("crec_10y_pct") + r("pct_menor_15") + r("pct_65mas", True) + r("reemplazamiento")) / 4 * 100
        g["dim_viv"] = r("viv_1000") * 100
        g["dim_ser"] = (r("salud_10k") + r("escuelas_1000") + r("paradas_1000") + r("telefono_1000")) / 4 * 100
        g["dim_acc"] = (r("tiempo_centro_min", True) + r("paradas_1000") + r("distancia_sol_km", True)) / 3 * 100
        g["dim_emp"] = (r("paro_100", True) + r("var_paro", True)) / 2 * 100
        partes.append(g)

    return (
        pd.concat(partes)
        .sort_values(["cod_ine", "anio"])
        .reset_index(drop=True)
    )


def indice(df: pd.DataFrame) -> pd.Series:
    """Índice ponderado EXACTO de la app (se calcula fuera del modelo, en SQL)."""
    out = None
    for k, w in PESOS.items():
        out = df[k].mul(w) if out is None else out.add(df[k].mul(w))
    return out


def X_y(df: pd.DataFrame):
    return df[FEATURES_M4].astype(float), df[TARGETS_M4].astype(float)
