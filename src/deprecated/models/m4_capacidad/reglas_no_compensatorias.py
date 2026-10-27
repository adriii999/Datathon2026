"""
reglas_no_compensatorias.py - Capa de salvaguarda metodológica de LoDCORE Madrid.

Hipótesis central: Carencia severa no compensable.
Si un municipio supera 1.500 hab/médico o una ocupación escolar del 95%,
su índice de capacidad de acogida (ICA) se trunca a alerta baja (< 40)
y no puede ser inflado artificialmente por factores inmobiliarios o fiscales.
"""

from src.lodcore.config.thresholds import (
    MAX_HABITANTES_POR_MEDICO_AP,
    MAX_OCUPACION_ESCOLAR_PCT,
    UMBRAL_ICA_BAJA,
    UMBRAL_ICA_MEDIA
)

def aplicar_salvaguarda_no_compensatoria(muni_dict: dict) -> dict:
    """
    Aplica las reglas duras de colapso a un municipio.
    
    Retorna el diccionario enriquecido con:
      - 'alerta_no_compensatoria': bool
      - 'motivo_colapso': str | None
      - 'indice_capacidad_ajustado': float
      - 'semaforo_ica': 'ALTA' | 'MEDIA' | 'BAJA'
    """
    faltantes = [c for c in ("hab_por_medico", "ocupacion_escolar_pct", "indice_capacidad_global") if c not in muni_dict]
    if faltantes:
        # Un 0 por defecto haría pasar por "sin saturación" un municipio sin datos.
        raise ValueError(f"Faltan campos obligatorios para la salvaguarda: {faltantes}")

    hab_por_medico = float(muni_dict["hab_por_medico"])
    ocupacion_escolar = float(muni_dict["ocupacion_escolar_pct"])
    score_raw = float(muni_dict["indice_capacidad_global"])

    colapso_sanitario = hab_por_medico > MAX_HABITANTES_POR_MEDICO_AP
    colapso_educativo = ocupacion_escolar > MAX_OCUPACION_ESCOLAR_PCT
    
    motivos = []
    if colapso_sanitario:
        motivos.append(f"Saturación sanitaria crítica ({hab_por_medico:.0f} hab/médico > {MAX_HABITANTES_POR_MEDICO_AP})")
    if colapso_educativo:
        motivos.append(f"Colapso de plazas escolares ({ocupacion_escolar:.1f}% > {MAX_OCUPACION_ESCOLAR_PCT}%)")
        
    if motivos:
        # Penalización no compensatoria: techo duro fijado en umbral bajo
        score_ajustado = min(score_raw, UMBRAL_ICA_BAJA - 1.0)
        semaforo = "BAJA"
        alerta = True
        motivo_str = "; ".join(motivos)
    else:
        score_ajustado = score_raw
        alerta = False
        motivo_str = None
        if score_ajustado >= UMBRAL_ICA_MEDIA:
            semaforo = "ALTA"
        elif score_ajustado >= UMBRAL_ICA_BAJA:
            semaforo = "MEDIA"
        else:
            semaforo = "BAJA"
            
    # Devolver una copia: la función no debe mutar el registro de entrada.
    resultado = dict(muni_dict)
    resultado['alerta_no_compensatoria'] = alerta
    resultado['motivo_colapso'] = motivo_str
    resultado['indice_capacidad_ajustado'] = round(score_ajustado, 2)
    resultado['semaforo_ica'] = semaforo

    return resultado
