"""
ckan_madrid.py - Conector oficial al Portal de Datos Abiertos de la Comunidad de Madrid (CKAN).
Fase F1.1 / F1.2 / F1.3 - Ingesta de Sanidad y Educación, snapshot inmutable y cruce por código INE.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("lodcore.ingest.ckan")

CKAN_BASE_URL = "https://datos.comunidad.madrid/api/3/action"
RESOURCE_ID_SANIDAD = "631ebd15-c19f-4984-9ad5-bb4b8554b5e0"   # Recursos y asistencia sanitaria por tipo
RESOURCE_ID_EDUCACION = "28d60557-1d73-4281-ab08-6cfd3b2f5f83" # Centros educativos de la CAM

# Tipos educativos que forman parte de la red de enseñanza escolar obligatoria y complementaria
TIPOS_EDUCACION_REGLADA = {
    "CP INF-PRI", "CP EGB", "CP INF-PRI-SEC", "IES", "SIES", "CPR INF-PRI",
    "CPR INF-PRI-SEC", "CPR SEC", "EEI", "EEI-CN", "CN", "CRA", "CEIP",
    "CENTRO PRIVADO DE EDUCACIÓN INFANTIL", "COLEGIO DE EDUCACIÓN INFANTIL Y PRIMARIA",
    "INSTITUTO DE EDUCACIÓN SECUNDARIA", "ESCUELA INFANTIL-CASA DE NIÑOS",
    "CENTRO PRIVADO DE EDUCACIÓN PRIMARIA Y SECUNDARIA", "CENTRO PRIVADO DE EDUCACIÓN SECUNDARIA",
    "COLEGIO RURAL AGRUPADO"
}

class IngestorCKANMadrid:
    """
    Ingesta de equipamientos de Sanidad y Educación desde el CKAN de la Comunidad de Madrid.
    Garantiza:
    - Paginación completa de endpoints datastore
    - Snapshot inmutable fechado en data/raw/ckan/ sin sobrescritura involuntaria
    - Filtrado semántico estricto de centros activos en situación de ALTA
    - Mapeo estricto a código INE de 5 dígitos (28xxx)
    - Trazabilidad y linaje formal
    """

    def __init__(self, raw_dir: Optional[Path] = None):
        if raw_dir is None:
            self.raw_dir = Path("data/raw/ckan")
        else:
            self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_datastore_all(self, resource_id: str, batch_size: int = 5000, timeout: int = 30) -> Dict[str, Any]:
        """
        Descarga la totalidad de registros de un recurso datastore de CKAN mediante paginación por offset.
        """
        records: List[Dict[str, Any]] = []
        offset = 0
        total = None

        while True:
            url = f"{CKAN_BASE_URL}/datastore_search?resource_id={resource_id}&limit={batch_size}&offset={offset}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "LoDCORE-Madrid-Datathon/2.0 (UC3M-CAM; contact: lodcore@madrid.gov)",
                    "Accept": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
            data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            result = data.get("result", {})
            batch = result.get("records", [])
            if not batch:
                break
            records.extend(batch)
            if total is None:
                total = result.get("total", len(records))
            offset += len(batch)
            logger.debug("Descargados %d / %s registros de %s", len(records), total, resource_id)
            if len(records) >= total:
                break

        return {
            "help": f"{CKAN_BASE_URL}/help_show?name=datastore_search",
            "success": True,
            "result": {
                "resource_id": resource_id,
                "total": len(records),
                "records": records
            }
        }

    def descargar_y_guardar_snapshot_sanidad(self, timeout: int = 25, overwrite: bool = False) -> Tuple[Path, Dict[str, Any]]:
        """Descarga centros sanitarios por tipo y persiste snapshot inmutable."""
        fecha_str = date.today().strftime("%Y%m%d")
        snapshot_path = self.raw_dir / f"ckan_madrid_sanidad_snapshot_{fecha_str}.json"

        if snapshot_path.exists() and not overwrite:
            logger.info("Snapshot CKAN Sanidad existente reutilizado: %s", snapshot_path)
            return snapshot_path, json.loads(snapshot_path.read_text(encoding="utf-8"))

        try:
            payload = self._fetch_datastore_all(RESOURCE_ID_SANIDAD, timeout=timeout)
            snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Snapshot CKAN Sanidad guardado en: %s (%d registros)", snapshot_path, len(payload["result"]["records"]))
            return snapshot_path, payload
        except Exception as exc:
            logger.warning("Error descargando CKAN Sanidad: %s. Buscando snapshot previo...", exc)
            previos = sorted(self.raw_dir.glob("ckan_madrid_sanidad_snapshot_*.json"), reverse=True)
            if previos:
                logger.info("Usando snapshot previo de Sanidad: %s", previos[0])
                return previos[0], json.loads(previos[0].read_text(encoding="utf-8"))
            raise RuntimeError("No se pudo conectar a CKAN Sanidad ni hay snapshot previo.") from exc

    def descargar_y_guardar_snapshot_educacion(self, timeout: int = 35, overwrite: bool = False) -> Tuple[Path, Dict[str, Any]]:
        """Descarga centros educativos mediante paginación completa y persiste snapshot inmutable."""
        fecha_str = date.today().strftime("%Y%m%d")
        snapshot_path = self.raw_dir / f"ckan_madrid_educacion_snapshot_{fecha_str}.json"

        if snapshot_path.exists() and not overwrite:
            logger.info("Snapshot CKAN Educación existente reutilizado: %s", snapshot_path)
            return snapshot_path, json.loads(snapshot_path.read_text(encoding="utf-8"))

        try:
            payload = self._fetch_datastore_all(RESOURCE_ID_EDUCACION, timeout=timeout)
            snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Snapshot CKAN Educación guardado en: %s (%d registros)", snapshot_path, len(payload["result"]["records"]))
            return snapshot_path, payload
        except Exception as exc:
            logger.warning("Error descargando CKAN Educación: %s. Buscando snapshot previo...", exc)
            previos = sorted(self.raw_dir.glob("ckan_madrid_educacion_snapshot_*.json"), reverse=True)
            if previos:
                logger.info("Usando snapshot previo de Educación: %s", previos[0])
                return previos[0], json.loads(previos[0].read_text(encoding="utf-8"))
            raise RuntimeError("No se pudo conectar a CKAN Educación ni hay snapshot previo.") from exc

    def procesar_sanidad(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Agrupa los centros sanitarios (centros de salud y consultorios) del año más reciente
        por código INE de 5 dígitos estricto.
        """
        records: List[Dict[str, Any]] = payload.get("result", {}).get("records", [])
        if not records:
            return {}

        k_anio = next((k for k in records[0].keys() if "o" in k.lower() and len(k) <= 4), "Año")
        k_cod = next((k for k in records[0].keys() if "digo" in k.lower() and "territorio" in k.lower()), "Código territorio")

        recs_muni = [r for r in records if r.get("Tipo territorio") in ("Municipios", "Municipio") and r.get(k_cod) is not None]
        if not recs_muni:
            return {}

        anio_max = max(int(r.get(k_anio, 0)) for r in recs_muni)

        sanidad_por_ine: Dict[str, Dict[str, Any]] = {}
        for r in recs_muni:
            if int(r.get(k_anio, 0)) != anio_max:
                continue

            cod_val = r.get(k_cod)
            if cod_val is None or not str(cod_val).isdigit():
                continue

            s = str(cod_val)
            if len(s) >= 2:
                muni_num = int(s[:-1])
                codigo_ine = f"28{muni_num:03d}"
            else:
                continue

            tipo = str(r.get("Tipo", "")).strip().lower()
            try:
                valor = int(r.get("Valor", 0))
            except (ValueError, TypeError):
                valor = 0

            if codigo_ine not in sanidad_por_ine:
                sanidad_por_ine[codigo_ine] = {
                    "codigo_ine": codigo_ine,
                    "nombre_territorio": r.get("Territorio", ""),
                    "num_centros_salud_ckan": 0,
                    "num_consultorios_ckan": 0,
                    "anio_referencia": anio_max,
                    "origen": "OBSERVADO",
                    "metodo": "CKAN_MADRID_CENTROS_SANITARIOS_POR_TIPO",
                    "fuente_datos": f"CKAN Comunidad de Madrid (Recursos sanitarios por tipo, tabla oficial, año {anio_max})"
                }

            if "salud" in tipo and "consultorio" not in tipo:
                sanidad_por_ine[codigo_ine]["num_centros_salud_ckan"] += valor
            elif "consultorio" in tipo and "salud" not in tipo:
                sanidad_por_ine[codigo_ine]["num_consultorios_ckan"] += valor
            elif "consultorios locales y centros de salud" in tipo:
                if sanidad_por_ine[codigo_ine]["num_centros_salud_ckan"] == 0:
                    sanidad_por_ine[codigo_ine]["num_centros_salud_ckan"] = valor

        return sanidad_por_ine

    def procesar_educacion(self, payload: Dict[str, Any], solo_reglada: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Agrupa los centros educativos por código INE de 5 dígitos.
        Aplica filtros de calidad y coherencia semántica:
        - Exclusión de centros en estado BAJA, INACTIVO, ANULACIÓN o EN TRÁMITE (solo SITUACIÓN == 'ALTA')
        - Opcionalmente filtra centros de enseñanza escolar reglada (Infantil, Primaria, Secundaria, Bachillerato, FP).
        """
        records: List[Dict[str, Any]] = payload.get("result", {}).get("records", [])
        if not records:
            return {}

        educacion_por_ine: Dict[str, Dict[str, Any]] = {}
        for r in records:
            situacion = str(r.get("SITUACIÓN", r.get("SITUACIN", ""))).strip().upper()
            if situacion != "ALTA":
                continue

            tipo_abrv = str(r.get("TIPO_ABRV", "")).strip()
            tipo_ext = str(r.get("TIPO_EXT", "")).strip()

            if solo_reglada:
                es_reglada = (
                    tipo_abrv in TIPOS_EDUCACION_REGLADA or
                    any(t in tipo_ext.upper() for t in ["INFANTIL", "PRIMARIA", "SECUNDARIA", "INSTITUTO", "COLEGIO"])
                )
                if not es_reglada:
                    continue

            cdmuni = r.get("CDMUNI")
            if cdmuni is None:
                continue

            try:
                cod_num = int(cdmuni)
            except (ValueError, TypeError):
                continue

            codigo_ine = f"28{cod_num:03d}"
            if codigo_ine not in educacion_por_ine:
                educacion_por_ine[codigo_ine] = {
                    "codigo_ine": codigo_ine,
                    "num_centros_educativos_ckan": 0,
                    "origen": "OBSERVADO",
                    "metodo": "CKAN_MADRID_CENTROS_EDUCATIVOS_ALTA",
                    "fuente_datos": "CKAN Comunidad de Madrid (Directorio oficial Centros Educativos en ALTA)"
                }

            educacion_por_ine[codigo_ine]["num_centros_educativos_ckan"] += 1

        return educacion_por_ine
