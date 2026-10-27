"""
app.py - Servidor mínimo FastAPI Nivel N3 para LoDCORE Madrid.
Siguiendo el diseño de arquitectura:
  - Nivel N1: Entrenamiento y exportación ONNX (Python/CI)
  - Nivel N2: Scoring masivo en Oracle Database (DBMS_VECTOR.LOAD_ONNX_MODEL + PREDICTION())
  - Nivel N3: Runtime dinámico mínimo en FastAPI para `/recomendar`, `/whatif` y `/optimizar-presupuesto`
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.lodcore import __version__, DATASET_VERSION
from src.lodcore.serving.fastapi.recomendar import (
    PreferenciasCiudadanoRequest,
    ejecutar_recomendacion_ciudadana
)
from src.lodcore.serving.fastapi.whatif import (
    SimuladorWhatIfRequest,
    MunicipioNoEncontrado,
    ejecutar_simulacion_whatif
)
from src.lodcore.models.m4_capacidad.optimizador_presupuestario import optimizar_presupuesto_comarca

class OptimizadorPresupuestoRequest(BaseModel):
    comarca: str = Field(default="Sierra Oeste", description="Comarca de la Comunidad de Madrid o 'todas'")
    presupuesto_total_eur: float = Field(default=2000000.0, gt=0, le=100000000.0, description="Presupuesto a distribuir (€)")

app = FastAPI(
    title="LoDCORE Madrid • API Nivel N3 (Ciudadano & Institución)",
    description=(
        "Servicio de inferencia interactiva para el II Datathon LoDCOREMadrid 2026. "
        "Expone `/recomendar` (M5 MCDM), `/whatif` (M4 contrafactual con SHAP) y "
        "`/optimizar-presupuesto` (asignación presupuestaria determinista por comarca)."
    ),
    version=__version__
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Diagnóstico"])
def health_check():
    return {
        "status": "ok",
        "service": "lodcore-madrid-n3-fastapi",
        "version": __version__,
        "dataset_version": DATASET_VERSION,
        "endpoints_activos": ["/recomendar", "/whatif", "/optimizar-presupuesto"]
    }

@app.post("/recomendar", tags=["Portal Ciudadano (M5)"])
def post_recomendar(req: PreferenciasCiudadanoRequest):
    """
    Calcula el ranking personalizado de municipios según las preferencias del ciudadano
    utilizando similitud del coseno ponderada (MCDM) y aplicando la salvaguarda no compensatoria.
    """
    try:
        return ejecutar_recomendacion_ciudadana(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/whatif", tags=["Panel Institucional (M4)"])
def post_whatif(req: SimuladorWhatIfRequest):
    """
    Ejecuta una simulación contrafactual sobre un municipio ante cambios demográficos o
    inversiones dotacionales, recalculando atribuciones, bandas al 95 % y el semáforo no compensatorio.
    """
    try:
        return ejecutar_simulacion_whatif(req)
    except MunicipioNoEncontrado as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en simulación: {exc}") from exc

@app.post("/optimizar-presupuesto", tags=["Panel Institucional (M4-OPT)"])
def post_optimizar_presupuesto(req: OptimizadorPresupuestoRequest):
    """
    Asigna un presupuesto cerrado entre los municipios de una comarca para maximizar la acogida
    neta comarcal resolviendo cuellos de botella con salvaguarda no compensatoria.
    """
    try:
        return optimizar_presupuesto_comarca(
            comarca=req.comarca,
            presupuesto_total_eur=req.presupuesto_total_eur
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en optimización: {exc}") from exc
