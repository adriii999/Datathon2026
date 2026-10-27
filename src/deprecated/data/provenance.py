"""
provenance.py - Trazabilidad de procedencia y linaje de datos en LoDCORE Madrid.
Regla transversal: toda variable o predicción debe portar `origen`, `metodo` y `dataset_version`.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict

class TipoOrigen(str, Enum):
    OBSERVADO = "OBSERVADO"
    IMPUTADO = "IMPUTADO"
    ESTIMADO = "ESTIMADO"
    PROYECTADO = "PROYECTADO"
    SIMULADO = "SIMULADO"

@dataclass(frozen=True)
class CeldaAuditada:
    valor: Any
    origen: TipoOrigen
    metodo: str
    fuente_oficial: str
    dataset_version: str = "v1.0.0-F2"

    def to_dict(self) -> Dict[str, Any]:
        if not self.origen or not self.metodo:
            raise ValueError("VIOLACIÓN DE PROCEDENCIA: Ninguna celda puede carecer de 'origen' o 'metodo'.")
        return {
            "valor": self.valor,
            "origen": self.origen.value,
            "metodo": self.metodo,
            "fuente_oficial": self.fuente_oficial,
            "dataset_version": self.dataset_version
        }

def validar_procedencia_registro(registro: Dict[str, Any]) -> bool:
    """
    Verifica que un registro o predicción publicada incluya campos de linaje obligatorios.
    Lanza ValueError si falta alguno.
    """
    campos_obligatorios = ("origen", "metodo", "dataset_version")
    faltantes = [c for c in campos_obligatorios if not registro.get(c)]
    if faltantes:
        raise ValueError(f"Registro rechazado por falta de procedencia: faltan {faltantes}")
    validos = {e.value for e in TipoOrigen}
    partes = [p.strip() for p in str(registro["origen"]).split("+")]
    invalidos = [p for p in partes if p not in validos]
    if invalidos:
        raise ValueError(
            f"Tipo de origen inválido: {registro['origen']} "
            f"(partes no permitidas: {invalidos})"
        )
    return True
