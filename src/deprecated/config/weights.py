"""
weights.py - Ponderaciones de sub-índices y preferencias MCDM para LoDCORE Madrid.
"""

# Pesos de los 4 Sub-índices para el Índice Global ICA (Suma = 1.0)
# Debe coincidir con la fórmula que genera `indice_capacidad_global` en el panel F2:
#   0,30·vivienda + 0,30·(sanidad+educación)/2 + 0,25·conectividad + 0,15·holgura
WEIGHTS_SUBINDICES_ICA = {
    "sanidad_educacion": 0.30,
    "conectividad": 0.25,
    "vivienda": 0.30,
    "holgura": 0.15
}

# Perfiles de recomendación ciudadana (MCDM - Distancia Coseno Ponderada)
CITIZEN_PROFILE_WEIGHTS = {
    "familia_escolar": {
        "educacion": 0.40,
        "sanidad": 0.30,
        "vivienda": 0.20,
        "conectividad": 0.10
    },
    "profesional_teletrabajo": {
        "conectividad": 0.40,
        "vivienda": 0.30,
        "tiempo_madrid": 0.20,
        "sanidad": 0.10
    },
    "senior_jubilacion": {
        "sanidad": 0.50,
        "servicios": 0.25,
        "tranquilidad_densidad": 0.15,
        "vivienda": 0.10
    }
}
