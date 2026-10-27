"""
perfiles.py - Modelo de preferencias y pesos del perfil ciudadano (C1).

Principio de diseño: no existe el peso "correcto". Los pesos base son iguales
(la opción menos arbitraria cuando no hay evidencia), los ajustes personales son
supuestos declarados y acotados, y se aplican por TRANSFERENCIA (el total suma
siempre 1,0). La incertidumbre sobre los pesos se mide con análisis de sensibilidad.

Regla intocable: estos pesos afectan SOLO al ranking. Nunca al ICA ni a la regla
no compensatoria.

Sin estado y sin E/S: todas las funciones son puras.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Dimensiones y pesos base
# ---------------------------------------------------------------------------

DIMENSIONES: Tuple[str, ...] = ("vivienda", "educacion", "conectividad", "sanidad")

PESOS_BASE_IGUALES: Dict[str, float] = {d: 0.25 for d in DIMENSIONES}

# Tope de masa que se puede transferir en total (20 %). Evita que un perfil
# extremo deje una dimensión sin peso efectivo.
MAX_TRANSFERENCIA_TOTAL: float = 0.20

# Límites de seguridad de cada peso individual.
PESO_MIN: float = 0.05
PESO_MAX: float = 0.60

# Supuestos declarados: (incremento, justificación, rango de sensibilidad).
SUPUESTOS_AJUSTES: Dict[str, Dict[str, object]] = {
    "hijo": {
        "delta": 0.05,
        "tope": 0.15,
        "justificacion": "La escolarización es la primera restricción de una familia",
        "rango_sensibilidad": 0.02,
    },
    "edad_joven": {
        "delta": 0.05,
        "justificacion": "Etapa de formación de hogar (edad <= 45)",
        "rango_sensibilidad": 0.02,
    },
    "edad_senior": {
        "delta": 0.10,
        "justificacion": "Uso sanitario creciente (edad >= 65)",
        "rango_sensibilidad": 0.03,
    },
    "edad_media": {
        "delta": 0.05,
        "justificacion": "Transición (46 <= edad < 65)",
        "rango_sensibilidad": 0.02,
    },
    "transporte_publico": {
        "delta": 0.05,
        "justificacion": "Dependencia real de la red de transporte",
        "rango_sensibilidad": 0.02,
    },
    "alquiler": {
        "delta": 0.05,
        "justificacion": "Sensibilidad al mercado de alquiler",
        "rango_sensibilidad": 0.02,
    },
}

# Parámetros financieros declarados (no constantes escondidas).
FINANCIACION_COMPRA: float = 0.80
TIPO_INTERES_ANUAL: float = 0.035
ANIOS_HIPOTECA: int = 30

# Asequibilidad (C3)
ENTRADA_PCT: float = 0.20          # % del precio que no se financia
ESFUERZO_MAXIMO_PCT: float = 30.0  # % máximo de ingresos destinado a vivienda

# Umbrales de esfuerzo financiero (% sobre ingresos netos).
UMBRAL_ASEQUIBLE: float = 30.0
UMBRAL_AJUSTADO: float = 35.0
UMBRAL_ALTO: float = 45.0


@dataclass(frozen=True)
class PreferenciasCiudadano:
    """Entrada del perfil ciudadano. Valida rangos al construirse."""

    presupuesto_max_m2_eur: float
    ingresos_netos_hogar_eur: float
    tiempo_max_commute_min: int
    modo_transporte: str = "publico"
    num_hijos: int = 0
    edad: int = 35
    tenencia: str = "compra"
    m2_deseados: float = 90.0

    def __post_init__(self) -> None:
        if not 300.0 <= self.presupuesto_max_m2_eur <= 6000.0:
            raise ValueError("presupuesto_max_m2_eur debe estar entre 300 y 6000")
        if not 500.0 <= self.ingresos_netos_hogar_eur <= 20000.0:
            raise ValueError("ingresos_netos_hogar_eur debe estar entre 500 y 20000")
        if not 10 <= self.tiempo_max_commute_min <= 120:
            raise ValueError("tiempo_max_commute_min debe estar entre 10 y 120")
        if self.modo_transporte not in ("publico", "privado"):
            raise ValueError("modo_transporte debe ser 'publico' o 'privado'")
        if not 0 <= self.num_hijos <= 6:
            raise ValueError("num_hijos debe estar entre 0 y 6")
        if not 18 <= self.edad <= 90:
            raise ValueError("edad debe estar entre 18 y 90")
        if self.tenencia not in ("compra", "alquiler"):
            raise ValueError("tenencia debe ser 'compra' o 'alquiler'")
        if not 40.0 <= self.m2_deseados <= 250.0:
            raise ValueError("m2_deseados debe estar entre 40 y 250")


def _transferir(base: Dict[str, float], incrementos: Dict[str, float]) -> Dict[str, float]:
    """Añade masa a unas dimensiones tomándola proporcionalmente de las demás.

    Garantiza suma 1,0 y ninguna dimensión negativa. Aplica el tope global de
    transferencia y los límites individuales.
    """
    pesos = dict(base)
    inc = {d: max(0.0, float(incrementos.get(d, 0.0))) for d in DIMENSIONES}
    total_inc = sum(inc.values())

    if total_inc <= 0.0:
        return _normalizar(pesos)

    # Tope global: se escala si se pide transferir más de lo permitido.
    if total_inc > MAX_TRANSFERENCIA_TOTAL:
        factor = MAX_TRANSFERENCIA_TOTAL / total_inc
        inc = {d: v * factor for d, v in inc.items()}
        total_inc = MAX_TRANSFERENCIA_TOTAL

    # Tope individual: ninguna dimensión receptora puede superar PESO_MAX.
    for d, v in inc.items():
        if pesos[d] + v > PESO_MAX:
            inc[d] = max(0.0, PESO_MAX - pesos[d])
    total_inc = sum(inc.values())
    if total_inc <= 0.0:
        return _normalizar(pesos)

    receptoras = [d for d in DIMENSIONES if inc[d] > 0.0]
    donantes = [d for d in DIMENSIONES if inc[d] == 0.0]
    masa_donantes = sum(pesos[d] for d in donantes)

    if masa_donantes <= 1e-9:
        # Caso límite: las cuatro dimensiones reciben ajuste, no hay de dónde
        # transferir. Se aplica el incremento y se renormaliza (variante A).
        return _normalizar({d: pesos[d] + inc[d] for d in DIMENSIONES})

    for d in receptoras:
        pesos[d] += inc[d]
    for d in donantes:
        pesos[d] -= total_inc * (pesos[d] / masa_donantes)

    return _normalizar(pesos)


def _normalizar(pesos: Dict[str, float]) -> Dict[str, float]:
    total = sum(pesos.values())
    if total <= 0.0:
        return dict(PESOS_BASE_IGUALES)
    return {d: pesos[d] / total for d in DIMENSIONES}


def incrementos_desde_preferencias(prefs: PreferenciasCiudadano) -> Dict[str, float]:
    """Traduce preferencias a incrementos de peso. Documenta el 'por qué' de cada uno."""
    inc: Dict[str, float] = {d: 0.0 for d in DIMENSIONES}

    if prefs.num_hijos > 0:
        inc["educacion"] += min(
            float(SUPUESTOS_AJUSTES["hijo"]["tope"]),
            float(SUPUESTOS_AJUSTES["hijo"]["delta"]) * prefs.num_hijos,
        )
    if prefs.edad <= 45:
        inc["educacion"] += float(SUPUESTOS_AJUSTES["edad_joven"]["delta"])
    elif prefs.edad >= 65:
        inc["sanidad"] += float(SUPUESTOS_AJUSTES["edad_senior"]["delta"])
    else:
        inc["sanidad"] += float(SUPUESTOS_AJUSTES["edad_media"]["delta"])

    if prefs.modo_transporte == "publico":
        inc["conectividad"] += float(SUPUESTOS_AJUSTES["transporte_publico"]["delta"])
    if prefs.tenencia == "alquiler":
        inc["vivienda"] += float(SUPUESTOS_AJUSTES["alquiler"]["delta"])

    return inc


def pesos_ciudadano(prefs: PreferenciasCiudadano) -> Dict[str, float]:
    """Pesos finales del perfil ciudadano (suman 1,0)."""
    return _transferir(PESOS_BASE_IGUALES, incrementos_desde_preferencias(prefs))


# ---------------------------------------------------------------------------
# Asequibilidad
# ---------------------------------------------------------------------------

def cuota_mensual_compra(
    precio_m2_eur: float,
    m2: float,
    financiacion: float = FINANCIACION_COMPRA,
    tipo_anual: float = TIPO_INTERES_ANUAL,
    anios: int = ANIOS_HIPOTECA,
) -> float:
    """Cuota mensual de una hipoteca francesa (80 % financiado por defecto)."""
    if precio_m2_eur <= 0 or m2 <= 0:
        return float("nan")
    prestamo = precio_m2_eur * m2 * financiacion
    r = tipo_anual / 12.0
    n = int(anios * 12)
    if r <= 0.0:
        return prestamo / n
    factor = r / (1.0 - (1.0 + r) ** (-n))
    return prestamo * factor


def cuota_mensual_alquiler(precio_m2_mes_eur: float, m2: float) -> float:
    """Cuota mensual de alquiler (el panel guarda €/m²/mes)."""
    if precio_m2_mes_eur <= 0 or m2 <= 0:
        return float("nan")
    return precio_m2_mes_eur * m2


def esfuerzo_financiero_pct(cuota_mensual: float, ingresos_netos_mes_eur: float) -> float:
    """Porcentaje de los ingresos netos que se va en la cuota de vivienda."""
    if ingresos_netos_mes_eur <= 0:
        raise ValueError("Los ingresos netos deben ser positivos")
    return (cuota_mensual / ingresos_netos_mes_eur) * 100.0


def factor_hipotecario(
    tipo_anual: float = TIPO_INTERES_ANUAL, anios: int = ANIOS_HIPOTECA
) -> float:
    """Cuota mensual por cada euro prestado (sistema francés)."""
    r = tipo_anual / 12.0
    n = int(anios * 12)
    if r <= 0.0:
        return 1.0 / n
    return r / (1.0 - (1.0 + r) ** (-n))


def entrada_eur(precio_m2_eur: float, m2: float, entrada_pct: float = ENTRADA_PCT) -> float:
    """Entrada no financiada (20 % por defecto)."""
    return precio_m2_eur * m2 * entrada_pct


def ingresos_necesarios_eur(
    cuota_mensual_eur: float, esfuerzo_maximo_pct: float = ESFUERZO_MAXIMO_PCT
) -> float:
    """Ingresos mensuales que harían falta para que la cuota no supere el esfuerzo máximo."""
    return cuota_mensual_eur / (esfuerzo_maximo_pct / 100.0)


def precio_max_sostenible_m2(
    ingresos_netos_mes_eur: float,
    m2: float,
    tenencia: str = "compra",
    esfuerzo_maximo_pct: float = ESFUERZO_MAXIMO_PCT,
    financiacion: float = FINANCIACION_COMPRA,
    tipo_anual: float = TIPO_INTERES_ANUAL,
    anios: int = ANIOS_HIPOTECA,
) -> float:
    """Precio máximo por m2 que el hogar puede pagar sin superar el esfuerzo máximo."""
    if ingresos_netos_mes_eur <= 0 or m2 <= 0:
        return float("nan")
    cuota_max = ingresos_netos_mes_eur * (esfuerzo_maximo_pct / 100.0)
    if tenencia == "alquiler":
        return cuota_max / m2
    prestamo_max = cuota_max / factor_hipotecario(tipo_anual, anios)
    precio_vivienda_max = prestamo_max / financiacion
    return precio_vivienda_max / m2


def presupuesto_efectivo_m2(prefs: PreferenciasCiudadano) -> float:
    """Presupuesto aplicado: el menor entre lo declarado y lo que el hogar puede pagar.

    Evita mostrar municipios que el ciudadano no podría pagar aunque los declare
    como admisibles.
    """
    sostenible = precio_max_sostenible_m2(
        prefs.ingresos_netos_hogar_eur, prefs.m2_deseados, prefs.tenencia
    )
    return min(prefs.presupuesto_max_m2_eur, sostenible)


def etiqueta_esfuerzo(esfuerzo_pct: float) -> str:
    if esfuerzo_pct <= UMBRAL_ASEQUIBLE:
        return "Asequible"
    if esfuerzo_pct <= UMBRAL_AJUSTADO:
        return "Ajustado"
    if esfuerzo_pct <= UMBRAL_ALTO:
        return "Alto"
    return "No asequible"


def evaluar_asequibilidad(prefs: PreferenciasCiudadano, fila: Dict[str, object]) -> Dict[str, object]:
    """Calcula cuota, entrada, esfuerzo y precio sostenible para una fila del panel."""
    if prefs.tenencia == "compra":
        precio_m2 = float(fila["precio_m2_compra"])
        cuota = cuota_mensual_compra(precio_m2, prefs.m2_deseados)
        entrada = entrada_eur(precio_m2, prefs.m2_deseados)
    else:
        precio_m2 = float(fila["precio_m2_alquiler"])
        cuota = cuota_mensual_alquiler(precio_m2, prefs.m2_deseados)
        entrada = 0.0

    esfuerzo = esfuerzo_financiero_pct(cuota, prefs.ingresos_netos_hogar_eur)
    sostenible = precio_max_sostenible_m2(
        prefs.ingresos_netos_hogar_eur, prefs.m2_deseados, prefs.tenencia
    )
    return {
        "precio_m2_evaluado": round(precio_m2, 2),
        "cuota_mensual_eur": round(cuota, 2),
        "entrada_eur": round(entrada, 2),
        "esfuerzo_financiero_pct": round(esfuerzo, 2),
        "etiqueta_esfuerzo": etiqueta_esfuerzo(esfuerzo),
        "ingresos_necesarios_eur": round(ingresos_necesarios_eur(cuota), 2),
        "precio_max_sostenible_m2": round(sostenible, 2),
    }


# ---------------------------------------------------------------------------
# Filtros duros
# ---------------------------------------------------------------------------

def filtrar_municipios(
    filas: Iterable[Dict[str, object]],
    prefs: PreferenciasCiudadano,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Aplica filtros duros. Devuelve (aceptados, excluidos con motivo).

    El colapso NO excluye: se marca aguas abajo. Aquí solo se filtran precio y tiempo.
    """
    aceptados: List[Dict[str, object]] = []
    excluidos: List[Dict[str, object]] = []
    presupuesto = presupuesto_efectivo_m2(prefs)

    for fila in filas:
        nombre = str(fila.get("nombre", "?"))
        if prefs.tenencia == "compra":
            precio = float(fila["precio_m2_compra"])
        else:
            precio = float(fila["precio_m2_alquiler"])

        tiempo = int(fila["tiempo_medio_madrid_minutos"])

        if precio > presupuesto:
            excluidos.append(
                {
                    "nombre": nombre,
                    "motivo": "PRECIO",
                    "detalle": f"{precio:.0f} €/m² > presupuesto efectivo {presupuesto:.0f}",
                }
            )
            continue
        if tiempo > prefs.tiempo_max_commute_min:
            excluidos.append(
                {
                    "nombre": nombre,
                    "motivo": "TIEMPO",
                    "detalle": f"{tiempo} min > máximo {prefs.tiempo_max_commute_min} min",
                }
            )
            continue

        aceptados.append(dict(fila))

    return aceptados, excluidos


# ---------------------------------------------------------------------------
# Sensibilidad
# ---------------------------------------------------------------------------

def variantes_pesos(
    pesos: Dict[str, float], delta: float = 0.20
) -> List[Dict[str, float]]:
    """Genera variantes del vector de pesos: +delta y -delta por dimensión.

    Sirve para medir si el ranking depende de los pesos (C2). No decide nada:
    solo produce los escenarios que hay que evaluar.
    """
    variantes: List[Dict[str, float]] = []
    for d in DIMENSIONES:
        for signo in (1.0, -1.0):
            v = dict(pesos)
            objetivo = max(PESO_MIN, min(PESO_MAX, v[d] * (1.0 + signo * delta)))
            # Reequilibrar el resto proporcionalmente para que la suma sea 1,0
            resto = [x for x in DIMENSIONES if x != d]
            masa_resto = sum(v[x] for x in resto)
            nueva_masa = 1.0 - objetivo
            for x in resto:
                v[x] = nueva_masa * (v[x] / masa_resto) if masa_resto > 0 else nueva_masa / len(resto)
            v[d] = objetivo
            variantes.append(_normalizar(v))
    return variantes


def resumen_perfil(prefs: PreferenciasCiudadano) -> Dict[str, object]:
    """Resumen legible para depuración y para la ficha de la interfaz."""
    pesos = pesos_ciudadano(prefs)
    return {
        "pesos": {d: round(p, 4) for d, p in pesos.items()},
        "suma_pesos": round(sum(pesos.values()), 10),
        "incrementos_solicitados": {d: round(v, 4) for d, v in incrementos_desde_preferencias(prefs).items()},
        "supuestos": {
            k: {"delta": v["delta"], "justificacion": v["justificacion"]}
            for k, v in SUPUESTOS_AJUSTES.items()
        },
        "parametros_financieros": {
            "financiacion": FINANCIACION_COMPRA,
            "tipo_interes_anual": TIPO_INTERES_ANUAL,
            "anios": ANIOS_HIPOTECA,
        },
    }
