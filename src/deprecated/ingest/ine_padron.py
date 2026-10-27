"""
ine_padron.py - Conector oficial a la API JSON del INE (Padrón Municipal continuo).
Fase F1.1 / F1.2 / F1.3 - Ingesta, snapshot inmutable en data/raw/ e identificación por código INE.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("lodcore.ingest.ine")

INE_TABLA_2881_URL = "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/2881?nult=1&tip=AM"

class IngestorINEPadron:
    """
    Ingesta de cifras oficiales de población municipal desde la API del INE (tabla 2881).
    Garantiza:
    - Snapshot inmutable fechado en data/raw/ine/
    - Extracción indexada por código INE de 5 dígitos (28xxx)
    - Trazabilidad y linaje formal
    """

    def __init__(self, raw_dir: Optional[Path] = None):
        if raw_dir is None:
            self.raw_dir = Path("data/raw/ine")
        else:
            self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def descargar_y_guardar_snapshot(self, timeout: int = 25, overwrite: bool = False) -> Tuple[Path, Dict[str, Any]]:
        """
        Descarga la tabla 2881 de la API del INE y la guarda como snapshot inmutable.
        Si el snapshot para la fecha actual ya existe y overwrite=False, lo reutiliza sin mutar disco ni red.
        """
        fecha_str = date.today().strftime("%Y%m%d")
        snapshot_path = self.raw_dir / f"ine_padron_2881_snapshot_{fecha_str}.json"

        if snapshot_path.exists() and not overwrite:
            logger.info("Snapshot INE existente reutilizado (inmutable): %s", snapshot_path)
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            return snapshot_path, payload

        logger.info("Conectando a la API INE (tabla 2881): %s", INE_TABLA_2881_URL)
        req = urllib.request.Request(
            INE_TABLA_2881_URL,
            headers={
                "User-Agent": "LoDCORE-Madrid-Datathon/2.0 (UC3M-CAM; contact: lodcore@madrid.gov)",
                "Accept": "application/json"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
            texto = raw_bytes.decode("utf-8", errors="replace")
            payload = json.loads(texto)

            snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Snapshot INE guardado con éxito en: %s", snapshot_path)
            return snapshot_path, payload

        except Exception as exc:
            logger.warning("Error descargando de la API del INE: %s. Buscando snapshot previo...", exc)
            snapshots_existentes = sorted(self.raw_dir.glob("ine_padron_2881_snapshot_*.json"), reverse=True)
            if snapshots_existentes:
                ultimo = snapshots_existentes[0]
                logger.info("Usando snapshot previo de respaldo: %s", ultimo)
                payload = json.loads(ultimo.read_text(encoding="utf-8"))
                return ultimo, payload
            raise RuntimeError(f"No fue posible conectar con INE ni se encontró snapshot en {self.raw_dir}") from exc

    def procesar_poblacion_madrid(self, payload: Any) -> Dict[str, Dict[str, Any]]:
        r"""
        Parsea las series del INE filtrando por:
        - Sexo = Total (código '0')
        - Provincia = Madrid (código municipal regex ^28\d{3}$)
        Devuelve un diccionario indexado estrictamente por código INE de 5 dígitos:
        {
            "28125": {
                "codigo_ine": "28125",
                "nombre_ine": "Robledo de Chavela",
                "poblacion_oficial": 4818,
                "anio_padron": 2025,
                "origen": "OBSERVADO",
                "metodo": "INE_TABLA_2881_API",
                "fuente_datos": "INE Padrón Municipal continuo (tabla 2881)"
            },
            ...
        }
        """
        series = payload if isinstance(payload, list) else payload.get("Data", [])
        resultado: Dict[str, Dict[str, Any]] = {}

        for serie in series:
            meta = {m.get("T3_Variable"): m for m in serie.get("MetaData", [])}
            mun = meta.get("Municipios")
            sexo = meta.get("Sexo")

            # Solo nos interesa el total de ambos sexos
            if not mun or not sexo or str(sexo.get("Codigo")) != "0":
                continue

            codigo = str(mun.get("Codigo", "")).strip()
            nombre = str(mun.get("Nombre", "")).strip()
            datos = serie.get("Data") or []

            # Filtrar solo municipios de Madrid (código 28xxx)
            if not re.fullmatch(r"28\d{3}", codigo) or not datos:
                continue

            # Tomar el dato del año más reciente
            dato_reciente = max(datos, key=lambda d: d.get("Anyo", 0))
            if dato_reciente.get("Valor") is None:
                continue

            pob_val = int(round(float(dato_reciente["Valor"])))
            anio_val = int(dato_reciente.get("Anyo", 2025))

            resultado[codigo] = {
                "codigo_ine": codigo,
                "nombre_ine": nombre,
                "poblacion_oficial": pob_val,
                "anio_padron": anio_val,
                "origen": "OBSERVADO",
                "metodo": "INE_TABLA_2881_API",
                "fuente_datos": f"INE Padrón Municipal continuo (tabla 2881, año {anio_val})"
            }

        logger.info("Procesados %d municipios oficiales de Madrid desde INE", len(resultado))
        return resultado
