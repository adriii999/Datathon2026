# PROJECT_CONTEXT.md — LoDCORE Madrid

> Documento de contexto operativo. Sustituye al plan anterior de 16 fases.
> **Última actualización:** 26 de octubre de 2026.
> Estado: MVP con datos reales integrados; en construcción el perfil ciudadano.

---

## 1. Qué es el proyecto

LoDCORE Madrid calcula qué municipios de la Comunidad de Madrid con menos de
50.000 habitantes pueden acoger nueva población sin colapsar sanidad, educación
y vivienda, y cuánto cuesta ampliar su capacidad.

**Pregunta que responde:**
> ¿Qué municipios pueden acoger la nueva población y cómo repartir inversión
> para que ninguno quede por debajo del umbral?

---

## 2. Estado real

| Elemento | Valor |
|---|---|
| Municipios del panel | **20** (8 comarcas, 200.267 hab.) |
| Columnas del panel | **92** |
| Columnas observadas (reales) | **8** |
| Dataset version | `v1.1.0-F2` |
| Capa real 179 | `panel_179_real.csv` (179 × 16, todo oficial, 0 nulos) |
| **Capa real <50k** | **`panel_menores_50k_real.csv` (155 × 16, 997.059 hab., 0 nulos)** |
| Capa real 179 | `data/validated/panel_179_real.csv` (10 columnas oficiales; 156 completos) |
| Perfil ciudadano | **C1-C6 hechos**; C7 (página APEX completa) pendiente |
| Ranking Oracle | `04-perfiles_y_ranking.sql` ejecutado: pesos y ranking coinciden con Python |
| Tests | **73 en verde** |
| Modelos ONNX en Oracle | M1 y M4 |
| API | `/health`, `/recomendar`, `/whatif`, `/optimizar-presupuesto` |
| Página de algoritmos | `frontend/algoritmos.html` (10 secciones: módulos, perfiles, arquitectura, datos reales, validación y glosario), enlazada desde el portal |
| DDL Oracle | `01-tablas_procesadas.sql`, `02-vistas_apex.sql`, `03-modelos_onnx.sql`, `04-perfiles_y_ranking.sql` |

**Ejecutar:**
```powershell
python -m pytest tests -q
python -m src.lodcore.ingest.pipeline_datos_reales
python scripts/run_all_models_n1_n2_n3.py
uvicorn src.lodcore.serving.fastapi.app:app --port 8000
```

---

## 3. Qué es real y qué es calibrado

**Observado (fuente oficial):**

| Columna | Fuente |
|---|---|
| `poblacion_actual`, `anio_padron` | INE Padrón continuo (tabla 2881) |
| `num_centros_salud` | CKAN Comunidad de Madrid (sanidad, centros en ALTA) |
| `num_centros_educativos` | CKAN Comunidad de Madrid (educación reglada en ALTA) |
| `valor_catastral_residencial_eur_m2`, `anio_valor_catastral` | DG Catastro vía CKAN CAM (2026) |
| `migracion_neta_externa` | INE Variaciones Residenciales vía CKAN CAM (media 2017-2021) |
| `crecimiento_pob_pct_2020_2025` | INE Padrón 2881 (nult=6) |

**Calibrado con `seed=2026` y declarado en `fuente_datos`:** precios de vivienda,
viviendas vacías, suelo disponible, médicos de atención primaria, plazas
escolares, camas hospitalarias, transporte y fibra.

**Limitación estructural:** el número de médicos y de plazas escolares por
municipio **no existe en datos abiertos** (SERMAS y Consejería no lo publican
desagregado). Por eso la regla de 1.500 hab/médico se evalúa sobre una
calibración, y así debe declararse.

---

## 4. Reto y criterios del jurado

**Reto:** ayudar a las Administraciones Públicas a identificar qué municipios de
menos de 50.000 habitantes tienen mejores posibilidades para acoger la nueva
población.

**Presentación:** 15 minutos, 29 de octubre de 2026, UC3M Leganés.
**Jurado:** perfil institucional (UC3M + Comunidad de Madrid), no ingeniero.

| Bloque | Criterio | Puntos |
|---|---|---|
| 1 | Planteamiento teórico correcto | 25 |
| 1 | Grado de resolución del reto | 25 |
| 2 | Sencillez de obtención y comprensión | 25 |
| 2 | Interactividad y flexibilidad | 25 |

**Implicación:** el jurado premia claridad e interacción, no complejidad
algorítmica. Hablar de inversión, capacidad y coste; evitar jerga de ML.

---

## 5. Arquitectura

| Nivel | Dónde | Qué hace |
|---|---|---|
| **N1** | Python / CI | Entrena y exporta a ONNX |
| **N2** | Oracle Database 23ai | `DBMS_VECTOR.LOAD_ONNX_MODEL` + `PREDICTION()` |
| **N3** | FastAPI + APEX | Recomendación, what-if y optimización |

**Flujo de una consulta desde APEX:**

```
APEX (formulario de perfil y parámetros)
   → proceso PL/SQL
   → SELECT sobre municipios_procesados + PREDICTION(ONNX)
   → función de ranking por perfil
   → región de resultados + mapa + comparador + ficha
```

El modelo **no** accede a la base de datos: el modelo **vive dentro** de la base
de datos. Ese es el argumento Oracle del proyecto.

---

## 6. Los 5 modelos y su papel

| Modelo | Algoritmo real | Pregunta que responde | Estado |
|---|---|---|---|
| **M1 · AVM** | Random Forest (sklearn) | ¿Puedo permitírmelo? Precio €/m² y gap de mercado | Ilustrativo: R² fuera de muestra −0,34. Diagnóstico 27-10: Ridge, RF simple y KNN tampoco logran R² positivo; el límite es la muestra, no el algoritmo |
| **M2 · Indicadores** | PCA + StandardScaler + MinMax | ¿Qué tal servicios y conexión? 4 sub-índices 0-100 | Determinista |
| **M3 · Demografía** | Extrapolación geométrica calibrada (CAGR) | ¿Va a crecer o vaciarse? | Determinista; pendiente backtest |
| **M4 · Capacidad** | GradientBoostingRegressor (sklearn) | ¿Cuánta gente cabe y qué se rompe antes? | Core; reproduce la fórmula del ICA |
| **M5 · Recomendador** | KMeans + coseno ponderado (MCDM) | ¿Cuál encaja conmigo? | Determinista |

**No se usa:** XGBoost, LightGBM, SHAP (librería), statsmodels, Prophet, OSMnx,
NetworkX, inferencia causal. Si se mencionan, deben presentarse como roadmap.

**Cadena:**
```
M2 (sub-índices) → M4 (ICA + regla dura) ┐
M1 (asequibilidad)                        ├→ M5 (ranking por perfil) → ficha
M3 (riesgo futuro)                        ┘
```

---

## 7. Reglas de negocio (no cambian sin discusión)

### Índice de capacidad de acogida
```
Sanidad    = médicos × 1.500 − población
Educación  = (plazas × 0,95 − alumnos) / 0,17
Vivienda   = 0,75 × viviendas vacías + 0,00375 × suelo disponible

ICA = 0,30·vivienda + 0,30·(sanidad+educación)/2 + 0,25·conectividad + 0,15·holgura
```

### Regla no compensatoria
```
Si hab/médico > 1.500  o  ocupación escolar > 95 %  →  ICA < 40 (BAJA)
```
Un servicio crítico colapsado no se compensa con otros factores.

### Regla de los perfiles
> **Los pesos del perfil cambian el RANKING. Nunca el ICA ni la regla dura.**

Orden obligatorio de cálculo:
```
1. ICA objetivo (igual para todos)
2. Regla no compensatoria (techo 40 si colapsa)
3. Ranking según pesos del perfil (solo entre municipios ya evaluados)
```

### Construcción de pesos (decisión C1)
```
Pesos base = iguales (0,25 cada dimensión): no hay evidencia para preferir
Ajustes personales = supuestos declarados y acotados
Aplicación = transferencia (el total suma 1,0; tope del 20 % por perfil)
Sensibilidad = ±20 % por dimensión; se reporta Spearman y estabilidad del top-5
```
No se afirma que exista un peso correcto: se mide cuánto importa equivocarse.

### Umbrales
Viven en `src/lodcore/config/thresholds.py`. Una sola fuente de verdad.
La fecha de decisión vive en `src/lodcore/config` (`FECHA_DECISION_MVP`).

---

## 8. Perfiles de usuario

| Perfil | Estado | Entradas | Objetivo |
|---|---|---|---|
| **Ciudadano** | En construcción (C1-C7) | presupuesto, ingresos, tiempo, hijos, edad, transporte, tenencia | Ranking personalizado |
| **Institución** | Diseñado (I1-I6) | presupuesto, comarca, objetivo eficiencia/equidad | Dónde invertir sin dejar municipios atrás |
| **Empresario** | **Roadmap** | — | Requiere suelo industrial, empleo y CNAE; hoy no hay datos |

Los municipios colapsados **se muestran marcados**, no se ocultan.

Los datos personales **no se almacenan**: se calculan en la sesión.

---

## 9. Etapas del perfil ciudadano

| Etapa | Contenido | Estado |
|---|---|---|
| **C1** | Modelo de preferencias: entradas, pesos, filtros y fórmula de esfuerzo | ✅ **Hecho**: `docs/perfil_ciudadano.md` + `src/lodcore/config/perfiles.py` (verificado) |
| **C2** | Motor de ranking con pesos del perfil y filtros duros | ✅ **Hecho y verificado**: `m5_recomendador/ranking.py`, 12 tests y coincidencia con Oracle |
| **C3** | Asequibilidad: cuota, entrada, ingresos necesarios, precio sostenible y gap de M1 | ✅ **Hecho**: funciones en `perfiles.py`, campos en `ranking.py` y en la función SQL |
| **C4** | Explicación por municipio: desglose, comparación con la media, fortalezas, carencias y texto sin causalidad | ✅ **Hecho**: `desglose_pct`, `comparacion_media`, `posicion` y motivo ampliado |
| **C5** | Capa Oracle: `PERFILES_PARAMETROS`, `PERFILES_BASE`, `PERFILES_AJUSTES` y `f_ranking_ciudadano` | ✅ **Ejecutado en Oracle**: 8 parámetros, 4 pesos base, 6 reglas; pesos y ranking coinciden con Python |
| **C6** | API N3 equivalente a la función SQL | ✅ **Hecho**: `/recomendar` usa `ranking.py` + `perfiles.py`; mismo resultado que APEX |
| **C7** | Página APEX del ciudadano: formulario, resultados, mapa y comparador | 🟡 SQL listo (`codigo_ine` en el ranking); falta la página de detalle |

---

## 10. Etapas del perfil institución (diseño)

| Etapa | Contenido |
|---|---|
| **I1** | Tabla de objetivos: eficiencia vs equidad |
| **I2** | Optimizador con objetivo max-min (subir el mínimo comarcal) |
| **I3** | Efecto de arrastre en vecinos (supuesto configurable, **no causal**) |
| **I4** | Comparativa "mismo presupuesto, dos objetivos" |
| **I5** | Página APEX institucional |
| **I6** | Pruebas del optimizador |

---

## 11. Pruebas transversales

- Monotonía: bajar presupuesto no incorpora municipios más caros.
- Filtros duros: precio y tiempo se respetan siempre.
- Regla dura: un municipio colapsado no entra en el top aunque sus pesos le favorezcan.
- Determinismo: mismos parámetros, mismo orden.
- Equivalencia SQL ↔ Python del ranking: `tests/unit/test_perfiles_sql_sync.py`
  (9 comprobaciones de parametros, pesos base, reglas y topes). No necesita Oracle.
- Equivalencia ONNX ↔ Python (`< 1e-6` relativo).

---

## 12. Qué está hecho, qué falta y qué no se hará

**Hecho:** panel con datos reales, índice documentado, reglas duras, simulador,
optimizador por eficiencia, ONNX en Oracle, mapa ORDS, portal estático,
perfil ciudadano C1-C4 y C6 con ranking por perfiles, capa Oracle (04) y 73 tests.

**Falta:** C1-C7, I1-I6, backtest de M3, validación ICA vs crecimiento real,
regeneración del paquete de entrega.

**No se hará en este ciclo:** isócronas con GTFS, SARIMAX/Prophet, inferencia
causal/ATE, spillover medido, perfil empresario, panel de 179 municipios.

---

## 13. Roadmap técnico (para el pitch)

1. Isócronas reales sobre GTFS/OSM (hoy: heurística por tiempo y paradas).
2. SARIMAX con exógenas (hoy: extrapolación geométrica validable por backtest).
3. Inferencia causal y ATE (requiere historial de intervenciones no público).
4. Spillover estimado con matriz origen-destino (hoy: supuesto configurable).
5. Capa de datos oficiales para los 179 municipios menores de 50.000 hab.
6. Perfil empresario con CNAE, afiliación y uso industrial del suelo.

---

## 14. Regla de honestidad

Todo número publicado debe poder rastrearse a un fichero o a una función.
Si una variable es calibrada, se declara en `fuente_datos` y se dice en la
presentación. Es mejor un sistema pequeño y comprobable que uno grande y falso.

---

## 15. Ordenar, predecir y simular: tres cosas distintas

| Operación | Pregunta | ¿Usa datos del futuro? | Dónde está |
|---|---|---|---|
| **Evaluar / ordenar** | ¿Qué municipio encaja con lo que pido? | No | Ranking (C2) |
| **Predecir** | ¿Qué pasará en este municipio en 2030? | Sí | M3 (población y saturación) |
| **Simular** | ¿Qué pasa si invierto X o llegan N personas? | No | What-if y optimizador |

**Importante:** el formulario del ciudadano no alimenta una predicción: son sus
preferencias. El ranking evalúa una realidad ya presente en los datos.

**La predicción honesta que falta (E6):** comprobar si el ICA anticipa el
crecimiento real. Se hace con backtest: indicadores de 2015 -> predecir 2020 ->
comparar con el crecimiento observado. Eso permite decir "el índice acierta",
no "el índice predice".

**Lo que no se puede predecir:** "¿seré feliz viviendo ahí?". No existe ninguna
etiqueta pública de satisfacción por municipio y perfil. Sin `y` no hay modelo.

---

## 16. Ampliación a 179 municipios (E4, no C6)

Hoy el panel tiene 20 municipios porque son los únicos con las 92 columnas.

**El código no asume 20:** el ranking usa `len(aceptados)` y las funciones SQL
recorren las filas de la tabla. Añadir municipios no requiere cambiar el motor.

**Lo que sí habría que actualizar al ampliar:**

| Elemento | Cambio |
|---|---|
| `tests/unit/test_ingest.py` (72, 98) | Asserts que fijan 20 |
| `tests/integration/test_fuga_l1_l7_y_baselines.py` (99-100) | Asserts que fijan 20 |
| `tests/unit/test_ranking_ciudadano.py` (255) | Assert que fija 20 |
| `scripts/check_model_cards.py` (61) | Población agregada (200.267) |
| `frontend/index.js` | Datos embebidos |
| Oracle | `TRUNCATE` + recarga del CSV |

**Decisión:** no se amplía inventando las 84 columnas calibradas. Se añade una
**capa real** con datos oficiales como dataset separado para el mapa y el
diagnóstico:

- `data/validated/panel_179_real.csv` → los 179 municipios (179 × 16)
- `data/validated/panel_menores_50k_real.csv` → **155 municipios < 50.000 hab.** (el ámbito del reto)

Ambas con 0 nulos: los centros sin dato en CKAN se rellenan con 0. La capa de 20
mantiene el método completo (92 columnas, modelos y ranking).

**Estado:** ✅ dataset construido con `python -m src.lodcore.ingest.pipeline_179`
(179 filas, 16 columnas, 156 municipios con todas las fuentes; 23 pueblos
pequeños sin centros educativos en ALTA). Falta cargarlo en Oracle y pintarlo
en el mapa.

**Frase para el jurado:** "El método está validado en 20 municipios; la capa
oficial cubre 179; extender el modelo es sustituir variables calibradas por
SERMAS y Educación, no escribir más código."

---

## 17. Validación (E6)

**Qué mide:** si el índice se corresponde con el crecimiento real, y cuánto
acierta la proyección demográfica. **Qué no mide:** predicción. El ICA es actual
(2024-2026) y el crecimiento es histórico: son asociaciones, no pronósticos.

**Cómo se regenera:**
```
python -m src.lodcore.validation.validacion_ica
```
Salida: `data/output/validacion_ica.json`

### Resultados

| Prueba | Resultado |
|---|---|
| ICA vs crecimiento 2020-2025 | ρ = **+0,101** (p = 0,673); IC95 % [−0,333, +0,555] |
| ICA vs crecimiento 2015-2020 | ρ = −0,367 (p = 0,112) |
| ICA vs crecimiento 2010-2015 | ρ = −0,156 (p = 0,510) |
| Indicadores reales vs crecimiento | Ninguno significativo (valor catastral −0,295; migración +0,072) |
| Regla no compensatoria | Sin alerta +6,95 % · Con alerta +7,58 % |
| Backtest M3 2020→2025 | Error mediano **1,55 %** · persistencia 7,02 % · base comarca 2,87 % |
| Backtest M3 2015→2020 | Error mediano 4,29 % · persistencia 4,91 % · base comarca 3,83 % |
| Coincidencia top-5 ICA / crecimiento | **2 de 5** (Montejo y Robledo) |

### Hallazgo principal

Los municipios que más crecieron son los que menos capacidad tienen:

| Municipio | Crecimiento 2020-2025 | ICA |
|---|---|---|
| Villa del Prado | +16,2 % | 36,9 |
| Colmenar de Oreja | +11,9 % | 35,3 |
| Navalcarnero | +11,2 % | 24,5 |

**Lectura:** el ICA mide capacidad disponible, no atractivo. El crecimiento real
no sigue a la capacidad, lo que confirma el problema territorial que justifica
la herramienta.

**Pendiente:** elegir el enfoque del pitch (A negativo / B diferencial / C mixto).

