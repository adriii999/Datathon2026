"""
whatif.py - Endpoint lógico N3 para el perfil Institucional:
Simulación contrafactual paramétrica de choque demográfico e inversión pública
con recálculo en vivo de atribuciones aditivas, banda 95 % y reglas no compensatorias.
"""

from typing import Dict, Any
import pandas as pd
from pydantic import BaseModel, Field

from src.lodcore.config.thresholds import (
    COSTE_UNITARIO_PLAZA_ESCOLAR_EUR,
    COSTE_AMPLIACION_CONSULTA_MEDICA_EUR,
    PROPORCION_POBLACION_ESCOLAR,
    UMBRAL_ICA_BAJA
)
from src.lodcore.models.m4_capacidad.predict import predecir_capacidad_municipio
from src.lodcore.data.provenance import TipoOrigen

class MunicipioNoEncontrado(Exception):
    """Señal explícita para que la API devuelva 404 y no confundirlo con un KeyError."""


class SimuladorWhatIfRequest(BaseModel):
    id_municipio: int = Field(..., ge=1, description="Identificador único del municipio")
    nuevos_residentes: int = Field(default=500, ge=0, le=15000, description="Incremento demográfico simulado")
    nuevos_medicos_ap: int = Field(default=0, ge=0, le=50, description="Contratación simulada de facultativos de Atención Primaria")
    nuevas_plazas_escolares: int = Field(default=0, ge=0, le=5000, description="Ampliación simulada de plazas educativas públicas")
    inversion_conectividad_eur: float = Field(default=0.0, ge=0.0, le=50000000.0, description="Inversión municipal adicional en transporte y conectividad (€)")

def ejecutar_simulacion_whatif(
    req: SimuladorWhatIfRequest,
    data_path: str = "data/validated/municipios_panel_validado.csv",
    model_dir: str = "src/lodcore/models/m4_capacidad"
) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
    match = df.loc[df["id_municipio"] == req.id_municipio]
    if match.empty:
        raise MunicipioNoEncontrado(f"No existe el municipio con id_municipio={req.id_municipio}")

    fila_base = match.iloc[0].to_dict()
    eval_base = predecir_capacidad_municipio(fila_base, model_dir=model_dir)

    # Construir estado contrafactual
    fila_sim = dict(fila_base)
    pob_nueva = float(fila_base["poblacion_actual"]) + float(req.nuevos_residentes)
    medicos_nuevos = float(fila_base["num_medicos_ap"]) + float(req.nuevos_medicos_ap)
    plazas_nuevas = float(fila_base["num_plazas_escolares"]) + float(req.nuevas_plazas_escolares)
    if medicos_nuevos <= 0 or plazas_nuevas <= 0:
        raise ValueError(
            "El municipio no tiene médicos o plazas escolares registrados: "
            "el contrafactual no puede calcular ratios de saturación fiables."
        )
    # Proporción del contingente entrante en edad escolar obligatoria (3-16 años).
    # Debe coincidir con el denominador 0,17 usado en la capacidad educativa del panel.
    alumnos_nuevos = (
        float(fila_base["num_alumnos_matriculados"])
        + PROPORCION_POBLACION_ESCOLAR * float(req.nuevos_residentes)
    )

    hab_por_medico_sim = pob_nueva / medicos_nuevos
    ocupacion_escolar_sim = (alumnos_nuevos / plazas_nuevas) * 100.0

    fila_sim["poblacion_actual"] = int(round(pob_nueva))
    fila_sim["hab_por_medico"] = round(hab_por_medico_sim, 2)
    fila_sim["ocupacion_escolar_pct"] = round(ocupacion_escolar_sim, 2)

    # Actualizar sub-índices en función del cambio relativo de saturación e inversión
    delta_med = (float(fila_base["hab_por_medico"]) - hab_por_medico_sim) * 0.05
    fila_sim["score_sanidad"] = float(max(0.0, min(100.0, float(fila_base["score_sanidad"]) + delta_med)))

    delta_esc = (float(fila_base["ocupacion_escolar_pct"]) - ocupacion_escolar_sim) * 0.85
    fila_sim["score_educacion"] = float(max(0.0, min(100.0, float(fila_base["score_educacion"]) + delta_esc)))

    bono_transporte = min(20.0, (req.inversion_conectividad_eur / 250000.0) * 2.5)
    fila_sim["score_conectividad"] = float(max(0.0, min(100.0, float(fila_base["score_conectividad"]) + bono_transporte)))

    # Consumo de vivienda vacía por los nuevos hogares (2.4 hab/hogar)
    hogares_entrantes = float(req.nuevos_residentes) / 2.4
    fila_sim["stock_vivienda_vacia_est"] = float(max(0.0, float(fila_base["stock_vivienda_vacia_est"]) - hogares_entrantes))

    eval_sim = predecir_capacidad_municipio(fila_sim, model_dir=model_dir)

    # Presupuesto CAPEX total de las medidas simuladas
    capex_medicos = float(req.nuevos_medicos_ap) * COSTE_AMPLIACION_CONSULTA_MEDICA_EUR
    capex_escuelas = float(req.nuevas_plazas_escolares) * COSTE_UNITARIO_PLAZA_ESCOLAR_EUR
    capex_total = capex_medicos + capex_escuelas + float(req.inversion_conectividad_eur)

    # La banda del modelo es sobre el ICA bruto. Si la regla dura trunca el ICA,
    # la banda publicada debe respetar el mismo techo; si no, mostraría valores
    # por encima del colapso que la propia regla prohíbe.
    ic_inf_sim = float(eval_sim["ic_95_inferior"])
    ic_sup_sim = float(eval_sim["ic_95_superior"])
    if eval_sim["alerta_no_compensatoria"]:
        techo = UMBRAL_ICA_BAJA - 1.0
        ic_sup_sim = min(ic_sup_sim, techo)
        ic_inf_sim = min(ic_inf_sim, ic_sup_sim)

    return {
        "id_municipio": req.id_municipio,
        "nombre": str(fila_base["nombre"]),
        "parametros_simulacion": req.model_dump(),
        "capex_total_estimado_eur": round(capex_total, 2),
        "estado_actual": {
            "indice_ica_ajustado": eval_base["indice_capacidad_ajustado"],
            "semaforo_ica": eval_base["semaforo_ica"],
            "hab_por_medico": round(float(fila_base["hab_por_medico"]), 1),
            "ocupacion_escolar_pct": round(float(fila_base["ocupacion_escolar_pct"]), 1),
            "alerta_no_compensatoria": eval_base["alerta_no_compensatoria"]
        },
        "estado_simulado": {
            "indice_ica_ajustado": eval_sim["indice_capacidad_ajustado"],
            "ic_95_inferior": round(ic_inf_sim, 2),
            "ic_95_superior": round(ic_sup_sim, 2),
            "delta_ica": round(eval_sim["indice_capacidad_ajustado"] - eval_base["indice_capacidad_ajustado"], 2),
            "semaforo_ica": eval_sim["semaforo_ica"],
            "hab_por_medico": round(hab_por_medico_sim, 1),
            "ocupacion_escolar_pct": round(ocupacion_escolar_sim, 1),
            "alerta_no_compensatoria": eval_sim["alerta_no_compensatoria"],
            "motivo_colapso": eval_sim["motivo_colapso"],
            "shap_recalculado": eval_sim["shap"],
            "explicacion_natural_es": eval_sim["explicacion_natural_es"]
        },
        "origen": TipoOrigen.SIMULADO.value,
        "metodo": "CONTRAFACTUAL_GBR_ATRIBUCIONES_ADITIVAS_NON_COMPENSATORY",
        "dataset_version": "v1.0.0-F2"
    }
