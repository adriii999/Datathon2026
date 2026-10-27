"""
ckan_vivienda_migracion.py - Conector de las fuentes REALES de la Etapa 1 (vivienda y demografía).

Tres fuentes oficiales, todas por municipio y con cruce por código INE de 5 dígitos:

1. Valor catastral residencial por unidad urbana (DG Catastro, vía CKAN CAM, serie 2003-2026).
2. Migraciones / Variaciones Residenciales (INE EVR, vía CKAN CAM, serie 1988-2021).
   Solo se usan los flujos externos NO ambiguos: llegadas desde otras CCAA + extranjero
   y salidas hacia otras CCAA. El flujo intra-CAM ("Destino y procedencia la Comunidad
   de Madrid") agrega llegadas y salidas sin separarlas y NO se usa. Las salidas al
   extranjero no se publican por municipio.
3. Padrón 2881 con 6 años de historia (INE Tempus3, nult=6): crecimiento real 2020-2025.

Los snapshots ya descargados viven en data/raw/ y son inmutables. Este conector solo
los lee; la descarga inicial está en el historial de git del repo.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger("lodcore.ingest.vivienda_migracion")

SNAP_VALOR_CATASTRAL = "ckan_madrid_valor_catastral_snapshot_20261025.csv"
SNAP_MIGRACIONES = "ckan_madrid_migraciones_vr_snapshot_20261025.csv"
SNAP_PADRON_6Y = "ine_padron_2881_6y_snapshot_20261025.json"

# Ventanas de cálculo explícitas (no constantes escondidas)
MIGRACION_ANIOS = (2017, 2021)   # media anual de la última ventana quinquenal publicada
CRECIMIENTO_ANIOS = (2020, 2025)  # padrón: de 2020 a 2025

# Categorías del EVR que son flujos externos NO ambiguos
CAT_LLEGADA_OTRAS_CCAA = "Destino la Comunidad de Madrid y procedencia otras Comunidades"
CAT_LLEGADA_EXTRANJERO = "Destino la Comunidad de Madrid y procedencia el extranjero"
CAT_SALIDA_OTRAS_CCAA = "Procedencia la Comunidad de Madrid con destino otras Comunidades"


def _cam_a_ine(cod_val: Any) -> Optional[str]:
    """Convierte el código de territorio de la CAM al código INE de 5 dígitos (28xxx).

    El portal de la CAM codifica los municipios con su número INE + un dígito final
    de control (p.ej. Robledo: CAM 1258 -> INE 28125). Se elimina el último carácter
    y se prefija 28. Misma convención que src/lodcore/ingest/ckan_madrid.py,
    con soporte para códigos leídos como float (columnas con NaN en el CSV).
    """
    if cod_val is None:
        return None
    try:
        if pd.isna(cod_val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(cod_val, float):
        if not float(cod_val).is_integer():
            return None
        cod_val = int(cod_val)
    s = str(cod_val).strip()
    if not s.isdigit() or len(s) < 2:
        return None
    try:
        n = int(s[:-1])
    except ValueError:
        return None
    return f"28{n:03d}"


def _num_es(s: Any) -> float:
    """Parsea números en formato español ('32,00') o inglés ('32.00')."""
    if s is None:
        return float("nan")
    t = str(s).strip().replace(" ", "")
    if t in ("", "nan", "None"):
        return float("nan")
    t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return float("nan")


class IngestorViviendaMigracionReal:
    """Procesa los tres snapshots reales y los indexa por código INE de 5 dígitos."""

    def __init__(self, raw_ckan: Optional[Path] = None, raw_ine: Optional[Path] = None):
        self.raw_ckan = Path(raw_ckan) if raw_ckan else Path("data/raw/ckan")
        self.raw_ine = Path(raw_ine) if raw_ine else Path("data/raw/ine")

    # ------------------------------------------------------------------ CSVs
    def _csv(self, nombre: str) -> pd.DataFrame:
        ruta = self.raw_ckan / nombre
        if not ruta.exists():
            raise FileNotFoundError(
                f"Snapshot no encontrado: {ruta}. Los snapshots son inmutables; "
                "consulte el historial de git para su descarga original."
            )
        df = pd.read_csv(ruta, encoding="latin-1", sep=";")
        df.columns = [c.strip() for c in df.columns]
        # Normaliza espacios dobles en categorías ("el  extranjero" llega con doble espacio)
        for col in ("Procedencia y destino", "Tipo territorio", "Edad", "Sexo"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        return df

    # -------------------------------------------------- 1. Valor catastral
    def procesar_valor_catastral(self) -> Dict[str, Dict[str, Any]]:
        """Valor catastral residencial medio por unidad urbana (€/m²), último año publicado."""
        df = self._csv(SNAP_VALOR_CATASTRAL)
        munis = df[df["Tipo territorio"] == "Municipios"].copy()
        if munis.empty:
            raise ValueError("Sin registros municipales en el snapshot de valor catastral")
        anio = int(munis["Año"].max())

        out: Dict[str, Dict[str, Any]] = {}
        for _, r in munis[munis["Año"] == anio].iterrows():
            ine = _cam_a_ine(r["Código territorio"])
            val = _num_es(r["Valor"])
            if ine is None or pd.isna(val):
                continue
            out[ine] = {
                "codigo_ine": ine,
                "nombre_territorio": str(r.get("Territorio", "")),
                "valor_catastral_residencial_eur_m2": val,
                "anio_valor_catastral": anio,
                "origen": "OBSERVADO",
                "metodo": "CATASTRO_VALOR_CATASTRAL_RESIDENCIAL_CKAN_CAM",
                "fuente_datos": (
                    "DG Catastro (valor catastral residencial por unidad urbana) "
                    f"vía CKAN Comunidad de Madrid, año {anio}"
                ),
            }
        logger.info("Valor catastral %d: %d municipios", anio, len(out))
        return out

    # -------------------------------------------------- 2. Migraciones externas
    def procesar_migraciones_externas(self) -> Dict[str, Dict[str, Any]]:
        """Saldo migratorio EXTERNO por municipio (llegadas - salidas), media anual de la ventana.

        Solo incluye: llegadas desde otras CCAA y desde el extranjero, y salidas hacia
        otras CCAA. No incluye flujos intra-CAM (el dataset los agrega sin separar
        dirección) ni salidas al extranjero (no publicadas por municipio).
        """
        df = self._csv(SNAP_MIGRACIONES)
        munis = df[(df["Tipo territorio"] == "Municipios")
                   & (df["Edad"] == "Total") & (df["Sexo"] == "Total")].copy()
        a0, a1 = MIGRACION_ANIOS
        munis["Año"] = pd.to_numeric(munis["Año"], errors="coerce")
        munis = munis[munis["Año"].between(a0, a1)]
        if munis.empty:
            raise ValueError(
                f"Sin registros municipales {a0}-{a1} en el snapshot de migraciones"
            )
        munis["Valor"] = munis["Valor"].map(_num_es)

        def flujo(r: pd.Series) -> str:
            c = r["Procedencia y destino"]
            if c.startswith(CAT_LLEGADA_OTRAS_CCAA[:25]) and "otras Comunidades" in c and "Destino" in c:
                return "llegada"
            if "extranjero" in c and c.startswith("Destino"):
                return "llegada"
            if "otras Comunidades" in c and c.startswith("Procedencia"):
                return "salida"
            return "intra_ignorado"

        munis["flujo"] = munis.apply(flujo, axis=1)
        munis = munis[munis["flujo"] != "intra_ignorado"]
        munis["codigo_ine"] = munis["Código territorio"].map(_cam_a_ine)
        munis = munis.dropna(subset=["codigo_ine"])

        out: Dict[str, Dict[str, Any]] = {}
        for ine, g in munis.groupby("codigo_ine"):
            lleg = g[g["flujo"] == "llegada"]["Valor"].sum()
            sal = g[g["flujo"] == "salida"]["Valor"].sum()
            n_anios = int(g["Año"].nunique())
            # Media anual sobre la ventana completa (no sobre los años con dato)
            media_lleg = lleg / max(n_anios, 1)
            media_sal = sal / max(n_anios, 1)
            out[ine] = {
                "codigo_ine": ine,
                "llegadas_externas_media_anual": round(media_lleg, 1),
                "salidas_otras_ccaa_media_anual": round(media_sal, 1),
                "migracion_neta_externa": round(media_lleg - media_sal, 1),
                "anios_migracion": f"{a0}-{a1}",
                "origen": "OBSERVADO",
                "metodo": "INE_EVR_SALDO_EXTERNO_CKAN_CAM",
                "fuente_datos": (
                    "INE Estadística de Variaciones Residenciales (flujos externos) "
                    f"vía CKAN Comunidad de Madrid, media anual {a0}-{a1}"
                ),
            }
        logger.info("Migraciones externas %d-%d: %d municipios", a0, a1, len(out))
        return out

    # -------------------------------------------------- 3. Crecimiento padrón
    def procesar_crecimiento_padron(self) -> Dict[str, Dict[str, Any]]:
        """Crecimiento poblacional real entre dos años del padrón (INE 2881, nult=6)."""
        ruta = self.raw_ine / SNAP_PADRON_6Y
        if not ruta.exists():
            raise FileNotFoundError(f"Snapshot no encontrado: {ruta}")
        payload = json.loads(ruta.read_text(encoding="utf-8"))
        series = payload if isinstance(payload, list) else payload.get("Data", [])
        a0, a1 = CRECIMIENTO_ANIOS
        pob: Dict[str, Dict[int, float]] = {}

        for s in series:
            meta = {m.get("T3_Variable"): m for m in s.get("MetaData", [])}
            mun, sexo = meta.get("Municipios"), meta.get("Sexo")
            if not mun or not sexo or str(sexo.get("Codigo")) != "0":
                continue
            ine = str(mun.get("Codigo", "")).strip().zfill(5)
            if not re.fullmatch(r"28\d{3}", ine):
                continue
            for d in s.get("Data") or []:
                if d.get("Valor") is not None:
                    pob.setdefault(ine, {})[int(d["Anyo"])] = float(d["Valor"])

        out: Dict[str, Dict[str, Any]] = {}
        for ine, serie in pob.items():
            if a0 not in serie or a1 not in serie or serie[a0] <= 0:
                continue
            out[ine] = {
                "codigo_ine": ine,
                "pob_padron_2020": int(serie[a0]),
                "pob_padron_2025": int(serie[a1]),
                "crecimiento_pob_pct_2020_2025": round(
                    (serie[a1] / serie[a0] - 1.0) * 100.0, 2
                ),
                "origen": "OBSERVADO",
                "metodo": "INE_PADRON_2881_NULT6_CRECIMIENTO",
                "fuente_datos": f"INE Padrón continuo (tabla 2881, nult=6), {a0}-{a1}",
            }
        logger.info("Crecimiento padrón %d-%d: %d municipios", a0, a1, len(out))
        return out


if __name__ == "__main__":
    ing = IngestorViviendaMigracionReal()
    vc = ing.procesar_valor_catastral()
    mg = ing.procesar_migraciones_externas()
    pb = ing.procesar_crecimiento_padron()
    print(f"valor catastral: {len(vc)} | migraciones: {len(mg)} | padrón: {len(pb)}")
    ej = "28125"  # Robledo de Chavela
    for nombre, d in (("valor_catastral", vc), ("migraciones", mg), ("padron", pb)):
        print(nombre, "->", d.get(ej))
