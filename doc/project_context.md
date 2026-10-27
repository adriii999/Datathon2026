# PROJECT_CONTEXT.md — Pueblos con vida · Simulador What-If

> Documento de contexto operativo. Sustituye por completo al borrador anterior
> (`doc/legacy/contexto_primer_borrador.md`), que describía 5 perfiles, 3 capas
> de datos y 73 tests. **Ese plan está retirado.** Este proyecto ahora tiene
> UNA funcionalidad, y todo esfuerzo se concentra en que funcione de verdad.

---

## 1. Qué es el proyecto

Un simulador What-If para los 155 municipios de la Comunidad de Madrid con
menos de 50.000 habitantes, dentro de la aplicación APEX **215 "Pueblos con
vida"** (página 10, Perfil Institucional).

**Una sola interacción:**

```
El usuario elige municipio + año + parámetros de choque
   → pulsa "Simular escenario"
   → se ejecutan los modelos M3 y M4 (ONNX, dentro de la base de datos)
   → el resultado son las métricas SIMULADAS de ese municipio para ese año
   → se pintan de forma visual (barras antes/después, semáforos, índice)
```

La salida de la simulación es **efímera**: no se guarda en ninguna tabla. Si
el usuario recarga, el escenario desaparece (volvió el mapa original).

**Pregunta que responde el simulador:**
> Si este municipio gana N habitantes y hacemos X inversión/en ampliaciones,
> ¿cómo queda su índice vitalidad para el año Y? ¿Y qué se rompe primero?

---

## 2. Datos: una única fuente

`data/dataset_155_municipios_2010-2025.xlsx` — construido por el equipo, casi
todo datos **oficiales** del INE y de la Comunidad de Madrid (ver sheet
`Fuentes`). Se guarda la versión anterior calibrada: ya no hay variables
inventadas, por lo que la regla de honestidad ahora es más sencilla.

| Sheet | Contenido |
|---|---|
| `LEEME` / `Diccionario` / `Fuentes` | Metadatos (109 indicadores) |
| `Panel_2010_2025` | Panel apilado: 155 municipios × años 2010-2025 = 2.480 filas × 32 columnas (demografía, paro, centros de salud, consultorios, transporte, paradas, valor catastral) |
| `Municipios` |иболее 155 filas: datos geográficos y de accesibilidad de cada municipio (distancia a Madrid, paradas por modo, escuelas en ALTA, estaciones de calidad del aire) |

---

## 3. Los modelos M1-M5: qué se sube a Oracle y qué no

Espectro de decisión (hay 2 modelos ONNX en `oracle/models`):

| Modelo | ¿Se sube? | Motivo |
|---|---|---|
| **M3 · Demografía** | ✅ **Sí, ONNX** | Es lo que da la **dimensión temporal** al simulador: predice la trayectoria poblacional (factor de crecimiento por horizonte h) del municipio para el año elegido. Sin el, el "año seleccionado" tendria que volver a ser una tabla precaculada estática |
| **M4 · Capacidad / Vitalidad** | ✅ **Sí, ONNX** | Es el **motor de las métricas simuladas**: aprende el mapeo estado municipal (ratios de saturación) → 5 dimensiones + índice. Al äüssenle el estado simulado del año Y devuelve las 5 dims (0-100) que pinta la interfaz |
| **M1 · AVM (precio vivienda)** | ❌ No | El negocio del simulador no es comprar piso; los precios quedan como valor estático del mapa. Además sus métricas es la más débil del proyecto (### ra: imposible ajustar bien el precio con esta muestra y unitario) |
| **M2 · Indicadores (PCA)** | ❌ No | Era un motor para los sub-índices calibrados del borrador antiguo. Las 5 dimensiones viven ahora (en el ínice) como rangos percentiles calcados en SQL; el PCA no modela nada nuevo sobre este estado |
| **M5 · Recomendador / ranking** | ❌ No | El ranking de los 155 municipios ya es una vista del mapa (columna `indice`), altamente interpretable y probada. **El simulador no ordena municipios; simula un municipio**. Una recommendation y un ranking son cosas distintas (ver §4 del legacy) |

Filosofía: **una dimension, un modelo, un fichero .onnx**. Resultado del
proyecto análog es simple: dos pipelines Python → dos .onnx → impregnados en
Oracle.

---

## 4. Arquitectura

| Nivel | Dónde | Qué |
|---|---|---|
| **N1** | Entrenamiento (Python) | `src/m3_demografia` y `src/m4_capacidad` entrenan y exportan `oracle/models/*.onnx` |
| **N2** | Oracle 23ai | Carga de ONNX (`DBMS_DATA_MINING` / `LOAD_ONNX_MODEL`) + `PREDICTION()` dentro de la función `PV_SIMULA` |
| **N3** | APEX (app 215) | Formulario → proceso AJAX "SIMULAR" → `PV_SIMULA` → JSON → render previo en la página con JS |

**Flujos ya verificados en la app de hoy (estado):**

```
Proceso AJAX "DATOS"    → ya funciona (tabla de municipios + serie)
Proceso AJAX "SIMULAR"  → hoy llama a `PV_SIMULA` solo para el año ACTUAL,
                          con recalcul de índice por percent-ranks puro en SQL
                          (es el esqueleto — hay que enchufarle los modelos)
```

### El estado simulado (lo ONNX-entity)

Ante cada combinación (año Y, P) el PL/SQL:

1. **M3** entrega el factor d'croissance del municipio hacia el año Y →
   `poblacion_base(Y) = poblacion_2025 × factor` (y la banda de confianza
   baja/alta de la misma forma).
2. Se añade el choque del usuario: `poblacion_sim(Y) = poblacion_base(Y) + N_hab`
   (+ viviendas, + centros salud, + escuelas, + paradas añadidas) → ratios.
3. **M4** toma los ratios del estado simulado → devuelve
   `(dim_dem, dim_viv, dim_ser, dim_acc, dim_emp)` ∈ [0,100].
4. El `índice_sim` es combinación ponderada fija (0,20/0,15/0,25/0,25/0,15);
   el `factor_limitante` es el `argmin₁` de las 5 dims (regla SQL que ya
   funciona en `PV_SIMULA` y se conserva).
5. Todo esto NO se persiste; sale como CLOB JSON. La function `PV_SIMULA`
   devuelve el mismo esquema "antes/despues" que ya muestra el front: los 10
   pares + `delta_indice`.

**Regla de los pesos:** los pesos del índice son decisión del diseño y no
cambia con el año ni con el perfil (no hay perfiles ya).
---

## 5. Estructura del repositorio (nueva)

```
Datathon2026/
├── data/
│   └── dataset_155_municipios_2010-2025.xlsx   ← único dataset de entrada
├── doc/
│   ├── project_context.md                      ← ESTE documento
│   ├── legacy/                                 ← borrador anterior (referencia)
│   └── modelos/                                ← (nuevo) fichas de M3 y M4
├── oracle/
│   ├── models/                                 ← SALIDA: un .onnx por modelo
│   │   ├── m3_demografia.onnx
│   │   └── m4_capacidad.onnx
│   └── sql/                                    ← carga ONNX + PV_SIMULA v2
├── scripts/
│   ├── run_m3_demografia.py                    ← ejecuta TODO el pipeline M3
│   ├── run_m4_capacidad.py                     ← ejecuta TODO el pipeline M4
│   └── run_all.py                              ← encadena los dos (entorno limpio)
├── src/
│   ├── common/                                 ← carga del xlsx + rutas
│   ├── m3_demografia/                          ← features + train + export ONNX
│   ├── m4_capacidad/                           ← features + train + export ONNX
│   └── deprecated/                             ← código antiguo, solo referencia
└── wallet/, opencode.json                      ← (sin cambios)
```

Convenciones:

- Cada pipeline acaba con un `.onnx` + un fichero de **ficha del modelo**
  (`doc/modelos/m3.json`, `m4.json`): features exactas, orden, métricas de
  backtest, fecha, hash del dataset. Sin ficha no se sube el modelo.
- Los scripts `run_*.py` corrigen `sys.path`, cargan `data/*.x|x`, entrenan,
  validan con onnxruntime **desde disco** (coin uso Python ↔ ONNX < 1e-6) y dejan
  el fichero en `oracle/models/`. Son auto-contenidos y se ejecutan desde
  cualquier sitio, sin dependencia mutua.

---

## 6. Ficha de cada modelo (intención)

| | **M3 · Demografía** | **M4 · Capacidad / Vitalidad** |
|---|---|---|
| **Tarea** | Regresión del factor de crecimiento poblacional a horizonte h (1-10 años) | Regresión multi-salida de las 5 dimensiones (0-100) del municipio |
| **Instancia** | Una por (municipio, año base t, horizonte h) — el panel da ~150k muestras | Una por (municipio × año) — 2.480 muestras |
| **Features (M3)** | historial: CAGR 1/3/5/10 años, variabilidad, tasa migración/media edad, paro; sin el futuro | Ratios de saturación: hab/centro salud, hab/escuela, hab/parada, km²/hab, densidad, paro_100, edad, distancia, viviendas/1000 hab |
| **Target (M4)** | población(t+h) / población(t) | Las 5 dims recalculadas con la misma fórmula de percent-rank del panel de ese año |
| **Algoritmo** | GradientBoostingRegressor | RandomForestRegressor multi-output |
| **Export** | `skl2onnx` (ArrayFeatureExtractor-less) | `skl2onnx` |
| **Consumo en Oracle** | `PREDICTION(m3..., 1 poblacion, ...) → factor` | `PREDICTION(m4...) → 5 valores de dim` |

## 7. Estado de la app 215 (inventario real)

| Página | Nombre | Uso nuevo |
|---|---|---|
| 1 | Inicio | Enlace al dashboard |
| **10** | **Perfil Institucional** | **Aquí vive el simulador** (región rails PL/SQL, procesos AJAX `DATOS` y `SIMULAR`) |
| 20 | Perfil Ciudadano | Sin perfil: se eliminará o quedará congelada |
| 30 | Datos | Documentación de fuentes (se conserva) |

Tablas Oracle ya cargadas por el equipo (se conservan salvo cambios mínimos):

`PV_MUNICIPIOS` (155 × 36), `PV_MUN_BASE` (más ranks y 5 dims precalculadas),
`PV_SERIE` (2010-2025), `PV_PROYECCION(_SERIE)` (2026-2035 con la extrap. geométrica —
sustituida por M3 ONNX), `PV_VECINOS`, `PV_MUN_2035` (vista de mapa),
`PV_SIMULA` (package actual — como base para la v2).

---

## 8. Cuadro de mando / criterios de éxito

La única forma de saber que el proyecto funciona:

1. `python scripts/run_all.py` reproduce M3 y M4 desde el xlsx y dejan ambos
   .onnx en `oracle/models/` con ficha y validación ONNX↔Python en verde.
2. En APEX 215 pág 10 se elige municipio+parámetros+**año** → "Simular escenario"
   → los 5 dims y el índice del escenario aparecen visualmente (barras
   antes/después + semáforo + factor limitante) en segundos, y desaparecen al recargar.
3. `PREDICTION()` en SQL de los .onnx es extensión del dato: mismo valor ejecutado
   desde Python y desde PL/SQL (≤1e-6).

Si alguno de los 3 se cumple → funcionamos. Si todos → proyecto redondo.

---

## 9. Fuera del alcance (declarado, para no volver atrás)

- Perfiles de usuario (ciudadano / institución / empresario): retirados
- FastAPI / portal estático / algoritmos.html: retirados
- Optimizador de presupuesto multi-municipio: retirado
- Persistir simulaciones: prohibido por diseño (la simulación es efímera)
- Precios de vivienda predictivos (M1): los precios quedan como dato estático
- Isócronas GTFS, SARIMAX, inferencia causal, ampliar a los 179: roadmap futuro
