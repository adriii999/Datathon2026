"""
validacion_ica.py - E6: validación del índice y backtest del modelo demográfico.

Responde a una sola pregunta: ¿sirve el índice para algo?

  A) ICA vs crecimiento real (2020-2025, 2015-2020, 2010-2015)
  B) Indicadores reales vs crecimiento (valor catastral, migración, densidad...)
  C) Regla no compensatoria: ¿los municipios colapsados crecen menos?
  D) Backtest de M3: predecir 2020 y 2025, comparar con la realidad
     y batir dos líneas base (persistencia y crecimiento comarcal)
  E) Coincidencia entre el top-5 por ICA y el top-5 por crecimiento

Salida: `data/output/validacion_ica.json` con todos los números y su fecha.
No modifica el panel, no entrena modelos y no decide nada: solo mide.

Uso:
    python -m src.lodcore.validation.validacion_ica
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.lodcore.models.m4_capacidad.reglas_no_compensatorias import (
    aplicar_salvaguarda_no_compensatoria,
)

PANEL_PATH = Path("data/validated/municipios_panel_validado.csv")
PADRON_PATH = Path("data/raw/ine/ine_padron_2881_16y_snapshot_20261027.json")
OUT_PATH = Path("data/output/validacion_ica.json")

# Ventanas de crecimiento analizadas
VENTANAS: Tuple[Tuple[int, int], ...] = ((2010, 2015), (2015, 2020), (2020, 2025))

# Parámetros del backtest: replican src/lodcore/models/m3_demografia/project.py
CAGR_MIN = -0.02
CAGR_MAX = 0.06

# Semilla del bootstrap (reproducibilidad)
SEED = 2026
N_BOOTSTRAP = 2000

# Indicadores reales que se contrastan contra el crecimiento
INDICADORES_REALES = (
    "valor_catastral_residencial_eur_m2",
    "migracion_neta_externa",
    "densidad",
    "tiempo_medio_madrid_minutos",
    "poblacion_actual",
)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def cargar_poblacion(ruta: Path = PADRON_PATH) -> Dict[str, Dict[int, float]]:
    """Devuelve {codigo_ine: {anio: poblacion}} desde el snapshot del INE."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el snapshot del padrón: {ruta}")
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    series = payload if isinstance(payload, list) else payload.get("Data", [])

    poblacion: Dict[str, Dict[int, float]] = {}
    for s in series:
        meta = {m.get("T3_Variable"): m for m in s.get("MetaData", [])}
        municipio, sexo = meta.get("Municipios"), meta.get("Sexo")
        if not municipio or not sexo or str(sexo.get("Codigo")) != "0":
            continue
        ine = str(municipio.get("Codigo", "")).strip().zfill(5)
        if not ine.startswith("28"):
            continue
        for d in s.get("Data") or []:
            if d.get("Valor") is not None:
                poblacion.setdefault(ine, {})[int(d["Anyo"])] = float(d["Valor"])
    return poblacion


def tasa_crecimiento(
    poblacion: Dict[str, Dict[int, float]], codigo: str, a0: int, a1: int
) -> float:
    """Crecimiento porcentual entre dos años. NaN si falta algún dato."""
    serie = poblacion.get(codigo, {})
    if a0 not in serie or a1 not in serie or serie[a0] <= 0:
        return float("nan")
    return (serie[a1] / serie[a0] - 1.0) * 100.0


def construir_dataset(
    panel: pd.DataFrame, poblacion: Dict[str, Dict[int, float]]
) -> pd.DataFrame:
    """Añade crecimiento por ventana y la bandera de la regla no compensatoria."""
    df = panel.copy()
    df["codigo_ine"] = df["codigo_ine"].astype(str).str.zfill(5)
    for a0, a1 in VENTANAS:
        df[f"crec_{a0}_{a1}"] = df["codigo_ine"].map(
            lambda c: tasa_crecimiento(poblacion, c, a0, a1)
        )
    df["alerta_no_compensatoria"] = df.apply(
        lambda r: bool(
            aplicar_salvaguarda_no_compensatoria(r.to_dict())["alerta_no_compensatoria"]
        ),
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# A) ICA vs crecimiento
# ---------------------------------------------------------------------------

def _correlacion(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "spearman": None, "p_valor": None, "pearson": None}
    rho, p = stats.spearmanr(x[ok], y[ok])
    r, _ = stats.pearsonr(x[ok], y[ok])
    return {
        "n": int(ok.sum()),
        "spearman": round(float(rho), 3),
        "p_valor": round(float(p), 3),
        "pearson": round(float(r), 3),
    }


def _bootstrap_spearman(
    x: Iterable[float], y: Iterable[float], n: int = N_BOOTSTRAP, seed: int = SEED
) -> Dict[str, float]:
    """IC 95 % del coeficiente de Spearman por bootstrap percentil."""
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    mask = ~(np.isnan(xv) | np.isnan(yv))
    xv, yv = xv[mask], yv[mask]
    if len(xv) < 3:
        return {"ic95_inferior": None, "ic95_superior": None}
    rng = np.random.default_rng(seed)
    valores: List[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(xv), len(xv))
        if len(set(idx.tolist())) < 3:
            continue
        rho = stats.spearmanr(xv[idx], yv[idx]).statistic
        if not np.isnan(rho):
            valores.append(float(rho))
    if not valores:
        return {"ic95_inferior": None, "ic95_superior": None}
    lo, hi = np.percentile(valores, [2.5, 97.5])
    return {"ic95_inferior": round(float(lo), 3), "ic95_superior": round(float(hi), 3)}


def analizar_ica_vs_crecimiento(df: pd.DataFrame) -> Dict[str, Any]:
    resultado: Dict[str, Any] = {}
    for a0, a1 in VENTANAS:
        col = f"crec_{a0}_{a1}"
        resultado[f"{a0}_{a1}"] = _correlacion(df["indice_capacidad_global"], df[col])

    col = "crec_2020_2025"
    ic = _bootstrap_spearman(df["indice_capacidad_global"], df[col])
    resultado["2020_2025"]["bootstrap_ic95"] = ic
    return resultado


def analizar_indicadores_reales(df: pd.DataFrame) -> Dict[str, Any]:
    resultado: Dict[str, Any] = {}
    for col in INDICADORES_REALES:
        if col not in df.columns:
            continue
        resultado[col] = _correlacion(df[col], df["crec_2020_2025"])
    return resultado


def analizar_regla_no_compensatoria(df: pd.DataFrame) -> Dict[str, Any]:
    salida: Dict[str, Any] = {}
    for flag, grupo in df.groupby("alerta_no_compensatoria"):
        etiqueta = "con_alerta" if flag else "sin_alerta"
        salida[etiqueta] = {
            "n": int(len(grupo)),
            "crecimiento_medio_2020_2025": round(
                float(grupo["crec_2020_2025"].mean()), 2
            ),
        }
    return salida


# ---------------------------------------------------------------------------
# D) Backtest de M3
# ---------------------------------------------------------------------------

def _cagr(valor_inicial: float, valor_final: float, anios: int) -> float:
    if valor_inicial <= 0 or anios <= 0:
        return float("nan")
    tasa = (valor_final / valor_inicial) ** (1.0 / anios) - 1.0
    return min(CAGR_MAX, max(CAGR_MIN, tasa))


def _predecir_individual(
    poblacion: Dict[str, Dict[int, float]], codigo: str, a_fit0: int, a_fit1: int, a_pred: int
) -> float:
    serie = poblacion.get(codigo, {})
    if a_fit0 not in serie or a_fit1 not in serie:
        return float("nan")
    tasa = _cagr(serie[a_fit0], serie[a_fit1], a_fit1 - a_fit0)
    if np.isnan(tasa):
        return float("nan")
    return serie[a_fit1] * ((1.0 + tasa) ** (a_pred - a_fit1))


def _predecir_comarca(
    df: pd.DataFrame,
    poblacion: Dict[str, Dict[int, float]],
    codigo: str,
    comarca: str,
    a_fit0: int,
    a_fit1: int,
    a_pred: int,
) -> float:
    serie_cod = poblacion.get(codigo, {})
    if a_fit1 not in serie_cod:
        return float("nan")
    codigos = df.loc[df["comarca"] == comarca, "codigo_ine"]
    pob0 = sum(poblacion.get(c, {}).get(a_fit0, 0.0) for c in codigos)
    pob1 = sum(poblacion.get(c, {}).get(a_fit1, 0.0) for c in codigos)
    tasa = _cagr(pob0, pob1, a_fit1 - a_fit0)
    if np.isnan(tasa):
        return float("nan")
    return serie_cod[a_fit1] * ((1.0 + tasa) ** (a_pred - a_fit1))


def _metricas(pred: np.ndarray, real: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    real = np.asarray(real, dtype=float)
    mask = ~(np.isnan(pred) | np.isnan(real))
    pred, real = pred[mask], real[mask]
    if len(pred) == 0:
        return {"n": 0}
    error_abs = np.abs(pred - real)
    ape = error_abs / np.where(real == 0, np.nan, real)
    return {
        "n": int(len(pred)),
        "mae_hab": round(float(error_abs.mean()), 2),
        "error_mediano_pct": round(float(np.nanmedian(ape)) * 100.0, 2),
        "sesgo_hab": round(float((pred - real).mean()), 2),
    }


def backtest_m3(
    df: pd.DataFrame, poblacion: Dict[str, Dict[int, float]]
) -> Dict[str, Any]:
    """Dos ventanas: predecir 2020 con datos 2010-2015, y 2025 con 2015-2020."""
    salida: Dict[str, Any] = {}
    for a_fit0, a_fit1, a_pred in ((2010, 2015, 2020), (2015, 2020, 2025)):
        reales, m3, comarca, persistencia = [], [], [], []
        for _, r in df.iterrows():
            cod, com = r["codigo_ine"], r["comarca"]
            real = poblacion.get(cod, {}).get(a_pred, np.nan)
            p_m3 = _predecir_individual(poblacion, cod, a_fit0, a_fit1, a_pred)
            p_com = _predecir_comarca(df, poblacion, cod, com, a_fit0, a_fit1, a_pred)
            p_per = poblacion.get(cod, {}).get(a_fit1, np.nan)
            if any(pd.isna(v) for v in (real, p_m3, p_com, p_per)):
                continue
            reales.append(real)
            m3.append(p_m3)
            comarca.append(p_com)
            persistencia.append(p_per)

        salida[f"{a_fit1}->{a_pred}"] = {
            "m3_individual": _metricas(np.array(m3), np.array(reales)),
            "base_comarca": _metricas(np.array(comarca), np.array(reales)),
            "persistencia": _metricas(np.array(persistencia), np.array(reales)),
        }
    return salida


# ---------------------------------------------------------------------------
# E) Coincidencia de rankings
# ---------------------------------------------------------------------------

def coincidencia_top(df: pd.DataFrame, k: int = 5) -> Dict[str, Any]:
    top_ica = (
        df.sort_values("indice_capacidad_global", ascending=False).head(k)["nombre"].tolist()
    )
    top_crec = df.sort_values("crec_2020_2025", ascending=False).head(k)["nombre"].tolist()
    comunes = sorted(set(top_ica) & set(top_crec))
    return {
        "k": k,
        "top_ica": top_ica,
        "top_crecimiento": top_crec,
        "coinciden": comunes,
        "n_coinciden": len(comunes),
    }


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

def ejecutar_validacion(
    panel_path: Path = PANEL_PATH,
    padron_path: Path = PADRON_PATH,
    out_path: Path = OUT_PATH,
    escribir: bool = True,
) -> Dict[str, Any]:
    """Ejecuta las cinco pruebas y devuelve (y opcionalmente guarda) el informe."""
    panel = pd.read_csv(panel_path)
    poblacion = cargar_poblacion(padron_path)
    df = construir_dataset(panel, poblacion)

    resultados: Dict[str, Any] = {
        "meta": {
            "fecha": date.today().isoformat(),
            "n_municipios": int(len(df)),
            "panel": str(panel_path),
            "padron": str(padron_path),
            "ventanas": [f"{a0}-{a1}" for a0, a1 in VENTANAS],
            "seed_bootstrap": SEED,
            "aviso": (
                "El ICA es actual (2024-2026) y el crecimiento es histórico: "
                "las correlaciones son asociación, no predicción."
            ),
        },
        "A_ica_vs_crecimiento": analizar_ica_vs_crecimiento(df),
        "B_indicadores_reales_vs_crecimiento": analizar_indicadores_reales(df),
        "C_regla_no_compensatoria": analizar_regla_no_compensatoria(df),
        "D_backtest_m3": backtest_m3(df, poblacion),
        "E_coincidencia_top5": coincidencia_top(df),
    }

    if escribir:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return resultados


def _imprimir(res: Dict[str, Any]) -> None:
    print("=" * 70)
    print("VALIDACIÓN E6 · LoDCORE Madrid")
    print("=" * 70)
    print(f"Municipios: {res['meta']['n_municipios']} · fecha: {res['meta']['fecha']}")

    print("\nA) ICA vs crecimiento real")
    for ventana, d in res["A_ica_vs_crecimiento"].items():
        extra = ""
        if "bootstrap_ic95" in d:
            ic = d["bootstrap_ic95"]
            extra = f"  IC95% [{ic['ic95_inferior']}, {ic['ic95_superior']}]"
        print(
            f"  {ventana}: rho={d['spearman']:+.3f} (p={d['p_valor']}) "
            f"r={d['pearson']:+.3f} n={d['n']}{extra}"
        )

    print("\nB) Indicadores reales vs crecimiento 2020-2025")
    for col, d in res["B_indicadores_reales_vs_crecimiento"].items():
        print(f"  {col:38s} rho={d['spearman']:+.3f} (p={d['p_valor']})")

    print("\nC) Regla no compensatoria")
    for grupo, d in res["C_regla_no_compensatoria"].items():
        print(
            f"  {grupo:12s} n={d['n']:2d}  crecimiento medio = "
            f"{d['crecimiento_medio_2020_2025']:+.2f} %"
        )

    print("\nD) Backtest M3")
    for ventana, modelos in res["D_backtest_m3"].items():
        print(f"  {ventana}")
        for nombre, m in modelos.items():
            print(
                f"    {nombre:15s} MAE={m['mae_hab']:8.2f} hab | "
                f"error mediano={m['error_mediano_pct']:5.2f} % | sesgo={m['sesgo_hab']:+8.2f}"
            )

    print("\nE) Coincidencia top-5")
    e = res["E_coincidencia_top5"]
    print(f"  ICA: {e['top_ica']}")
    print(f"  Crecimiento: {e['top_crecimiento']}")
    print(f"  Coinciden: {e['coinciden']} ({e['n_coinciden']}/{e['k']})")

    print("\nResultado guardado en data/output/validacion_ica.json")


if __name__ == "__main__":
    _imprimir(ejecutar_validacion())
