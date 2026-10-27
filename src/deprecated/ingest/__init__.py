"""
src/lodcore/ingest - Módulos de ingesta y snapshot inmutable de fuentes oficiales.
Fase F1 - Conectores INE (Padrón continuo) y CKAN Comunidad de Madrid (Sanidad / Educación).
"""

from .ine_padron import IngestorINEPadron
from .ckan_madrid import IngestorCKANMadrid

__all__ = ["IngestorINEPadron", "IngestorCKANMadrid"]
