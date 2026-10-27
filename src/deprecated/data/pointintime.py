"""
pointintime.py - Control estricto de punto-en-tiempo (fecha_disponible) para evitar fuga temporal.
Ningún modelo puede entrenar o inferir usando observaciones cuya fecha_disponible sea posterior
a la fecha_decision evaluada.
"""

from datetime import date, datetime
from typing import Union
import pandas as pd

def parse_fecha(f: Union[str, date, datetime]) -> date:
    if isinstance(f, datetime):
        return f.date()
    if isinstance(f, date):
        return f
    return datetime.fromisoformat(str(f).split("T")[0]).date()

def filtrar_point_in_time(df: pd.DataFrame, fecha_decision: Union[str, date], col_fecha: str = "fecha_disponible") -> pd.DataFrame:
    """
    Filtra un DataFrame garantizando que ninguna fila tenga `fecha_disponible > fecha_decision`.
    Lanza ValueError si falta la columna `fecha_disponible`.
    """
    if col_fecha not in df.columns:
        raise ValueError(f"FUGA TEMPORAL POTENCIAL: Falta columna obligatoria '{col_fecha}' en el dataset.")
    
    corte = parse_fecha(fecha_decision)
    fechas_serie = pd.to_datetime(df[col_fecha]).dt.date
    
    violaciones = (fechas_serie > corte).sum()
    if violaciones > 0:
        df_filtrado = df.loc[fechas_serie <= corte].copy()
        return df_filtrado
    return df.copy()

def verificar_no_leakage_temporal(df_train: pd.DataFrame, df_test: pd.DataFrame, col_fecha: str = "fecha_disponible") -> bool:
    """
    Verifica que max(df_train.fecha_disponible) <= min(df_test.fecha_disponible).
    """
    max_train = pd.to_datetime(df_train[col_fecha]).max()
    min_test = pd.to_datetime(df_test[col_fecha]).min()
    if max_train > min_test:
        raise AssertionError(f"Fuga temporal detectada: max(train)={max_train} > min(test)={min_test}")
    return True
