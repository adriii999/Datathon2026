"""
recomendar.py - Endpoint lógico N3 del perfil Ciudadano.

Recibe las preferencias del ciudadano (presupuesto, ingresos, tiempo, hijos,
edad, transporte y tenencia), deriva los pesos con `config/perfiles.py` y
ordena los municipios con `m5_recomendador/ranking.py`.

Misma lógica que la página APEX (`oracle/ddl/04-perfiles_y_ranking.sql`):
formulario -> pesos -> ranking. Una sola verdad, dos caminos.
"""

from typing import Any, Dict

import pandas as pd
from pydantic import BaseModel, Field

from src.lodcore.config.perfiles import PreferenciasCiudadano
from src.lodcore.models.m5_recomendador.ranking import (
    analisis_sensibilidad,
    rankear_municipios,
)


class PreferenciasCiudadanoRequest(BaseModel):
    """Entrada del perfil ciudadano. Los rangos replican config/perfiles.py."""

    presupuesto_max_m2_eur: float = Field(
        default=2000.0, ge=300.0, le=6000.0,
        description="Precio máximo admisible (€/m² en compra; €/m²/mes en alquiler)",
    )
    ingresos_netos_hogar_eur: float = Field(
        default=2500.0, ge=500.0, le=20000.0,
        description="Ingresos netos mensuales del hogar (€/mes)",
    )
    tiempo_max_commute_min: int = Field(
        default=90, ge=10, le=120,
        description="Tiempo máximo de desplazamiento a Madrid (min)",
    )
    modo_transporte: str = Field(
        default="publico", pattern="^(publico|privado)$",
        description="Modo de transporte habitual",
    )
    num_hijos: int = Field(default=0, ge=0, le=6, description="Número de hijos")
    edad: int = Field(default=35, ge=18, le=90, description="Edad del solicitante")
    tenencia: str = Field(
        default="compra", pattern="^(compra|alquiler)$", description="Régimen de tenencia"
    )
    m2_deseados: float = Field(default=90.0, ge=40.0, le=250.0, description="Superficie deseada (m²)")
    top_k: int = Field(default=5, ge=1, le=20, description="Número de municipios a devolver")
    incluir_sensibilidad: bool = Field(
        default=False,
        description="Si es true, añade el análisis de sensibilidad de los pesos (±20 %)",
    )


def ejecutar_recomendacion_ciudadana(
    req: PreferenciasCiudadanoRequest,
    data_path: str = "data/validated/municipios_panel_validado.csv",
) -> Dict[str, Any]:
    """Construye el ranking del perfil ciudadano y lo devuelve con su explicación."""
    df = pd.read_csv(data_path)
    filas = df.to_dict(orient="records")

    prefs = PreferenciasCiudadano(
        presupuesto_max_m2_eur=req.presupuesto_max_m2_eur,
        ingresos_netos_hogar_eur=req.ingresos_netos_hogar_eur,
        tiempo_max_commute_min=req.tiempo_max_commute_min,
        modo_transporte=req.modo_transporte,
        num_hijos=req.num_hijos,
        edad=req.edad,
        tenencia=req.tenencia,
        m2_deseados=req.m2_deseados,
    )

    resultado = rankear_municipios(filas, prefs, top_k=req.top_k)

    sensibilidad = None
    if req.incluir_sensibilidad:
        sensibilidad = analisis_sensibilidad(
            filas, prefs, delta=0.20, top_k=min(req.top_k, 5)
        )

    dataset_version = (
        str(df["dataset_version"].iloc[0])
        if "dataset_version" in df.columns and len(df) > 0
        else "SIN_VERSION_DECLARADA"
    )

    return {
        "total_evaluados": int(len(df)),
        "n_aceptados": int(resultado["n_aceptados"]),
        "n_excluidos": int(resultado["n_excluidos"]),
        "top_k": req.top_k,
        "preferencias_aplicadas": req.model_dump(),
        "pesos_aplicados": resultado["pesos"],
        "presupuesto_efectivo_m2": resultado["presupuesto_efectivo_m2"],
        "ranking": resultado["ranking"],
        "excluidos": resultado["excluidos"],
        "sensibilidad": sensibilidad,
        "origen": "ESTIMADO",
        "metodo": "PERFIL_CIUDADANO_SUMA_PONDERADA",
        "dataset_version": dataset_version,
    }
