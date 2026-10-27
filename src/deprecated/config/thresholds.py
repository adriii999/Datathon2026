"""
thresholds.py - Única fuente de verdad de umbrales operativos para LoDCORE Madrid.
Regla AGENTS.md: ningún umbral debe estar hardcodeado en los modelos o en la UI.
"""

# Umbrales de saturación sanitaria (SERMAS / OMS)
MAX_HABITANTES_POR_MEDICO_AP = 1500     # Ratio crítico de saturación asistencial
MEDICOS_AP_OBJETIVO_POR_1000 = 0.8       # 1 médico cada 1.250 hab. (óptimo SERMAS)

# Umbrales de saturación educativa (Consejería de Educación CAM)
MAX_OCUPACION_ESCOLAR_PCT = 95.0         # Por encima de 95%, el centro se considera colapsado
RATIO_ALUMNOS_POR_AULA = 25              # Ratio estándar por unidad de Infantil/Primaria
PROPORCION_POBLACION_ESCOLAR = 0.17      # 17% del contingente entrante en edad escolar (3-16)

# Umbrales del Semáforo ICA (Índice de Capacidad de Acogida)
UMBRAL_ICA_BAJA = 40.0                   # < 40.0: Capacidad Baja / Cuello de Botella Severo
UMBRAL_ICA_MEDIA = 60.0                  # 40.0 - 59.9: Capacidad Media / Requiere Inversión
                                         # >= 60.0: Capacidad Alta / Zona de Confort

# Criterio Municipio de 30 minutos (Movilidad y Transporte)
TIEMPO_MAXIMO_POLO_EMPLEO_MIN = 30       # Isócrona de acceso a hospitales o nodos metropolitanos

# Escenarios de estrés demográfico (Modelo M3)
ESCENARIO_ESTRES_MODERADO_PCT = 5.0      # +5% de población proyectada
ESCENARIO_ESTRES_SEVERO_PCT = 10.0       # +10% de población proyectada

# Módulos de costes de inversión dotacional (CAPEX estándar Comunidad de Madrid)
# Deben coincidir con `coste_estimado_ampliacion_servicios` del panel F2.
COSTE_UNITARIO_PLAZA_ESCOLAR_EUR = 12000.0      # Coste de ampliación de una plaza escolar
COSTE_AMPLIACION_CONSULTA_MEDICA_EUR = 450000.0 # Coste de ampliación de consulta AP + equipamiento

# Sensibilidad de los sub-índices del panel ante un cambio de saturación.
# Regla ÚNICA compartida por el simulador what-if y por la heurística de
# presupuesto: si vivieran en dos sitios, las dos herramientas darían números
# distintos para la misma inversión.
SENSIBILIDAD_SCORE_SANIDAD_POR_HAB_MEDICO = 0.05   # Puntos de score por hab/médico recuperado
SENSIBILIDAD_SCORE_EDUCACION_POR_PCT_OCUPACION = 0.85  # Puntos de score por % de ocupación recuperado
BONO_CONECTIVIDAD_POR_250K_EUR = 2.5                # Puntos de conectividad por cada 250.000 € invertidos
BONO_CONECTIVIDAD_MAXIMO = 20.0                    # Techo del bono de conectividad
HABITANTES_POR_HOGAR_ABSORCION = 2.4                # Consumo de vivienda vacía por hogar nuevo
