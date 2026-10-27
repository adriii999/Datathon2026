# Perfil Ciudadano — especificación funcional (C1)

> Contrato de entrada del perfil ciudadano y traducción **declarada** a pesos y filtros.
>
> **Principio:** no existe el peso "correcto". Los pesos son decisiones y se tratan
> como tales: se declaran, se acotan y se mide cuánto cambia el resultado al moverlos.

---

## 1. Entradas del formulario

| Campo | Tipo | Rango | Obligatorio | Para qué se usa |
|---|---|---|---|---|
| `presupuesto_max_m2_eur` | €/m² | 300–6.000 | Sí | Filtro duro de precio (compra: €/m²; alquiler: €/m²/mes) |
| `ingresos_netos_hogar_eur` | €/mes | 500–20.000 | Sí | Esfuerzo financiero |
| `tiempo_max_commute_min` | minutos | 10–120 | Sí | Filtro duro de tiempo a Madrid |
| `modo_transporte` | `publico` / `privado` | — | Sí | Peso de conectividad |
| `num_hijos` | entero | 0–6 | Sí | Peso de educación |
| `edad` | años | 18–90 | Sí | Peso de sanidad / educación |
| `tenencia` | `compra` / `alquiler` | — | Sí | Qué precio se evalúa |
| `m2_deseados` | m² | 40–250 | Sí (def. 90) | Cálculo de la cuota |

**Privacidad:** no se persiste ningún dato personal. Se calcula en la sesión.

---

## 2. Pesos base: iguales

| Dimensión | Peso base |
|---|---|
| Vivienda | 0,25 |
| Educación | 0,25 |
| Conectividad | 0,25 |
| Sanidad | 0,25 |

**Justificación:** no hay evidencia que permita preferir una dimensión sobre otra.
Cuando no hay evidencia, el reparto igual es la opción **menos arbitraria**, porque
no exige justificar por qué una vale más que otra. Cualquier otro reparto sería una
decisión política o personal que habría que defender.

---

## 3. Ajustes personales: supuestos declarados

Ningún ajuste es una verdad. Cada uno es un supuesto con justificación cualitativa
y un valor acotado que entra en el análisis de sensibilidad.

| Condición | Ajuste | Justificación | Rango de sensibilidad |
|---|---|---|---|
| Cada hijo | Educación **+0,05** (máx. +0,15) | La escolarización es la primera restricción de una familia | ±0,02 por hijo |
| Edad ≤ 45 | Educación **+0,05** | Etapa de formación de hogar | ±0,02 |
| Edad ≥ 65 | Sanidad **+0,10** | Uso sanitario creciente | ±0,03 |
| 46 ≤ edad < 65 | Sanidad **+0,05** | Transición | ±0,02 |
| Transporte público | Conectividad **+0,05** | Dependencia real de la red | ±0,02 |
| Alquiler | Vivienda **+0,05** | Sensibilidad al mercado de alquiler | ±0,02 |

Los rangos son los que se usarán en el análisis de sensibilidad (sección 7).

---

## 4. Cómo se aplican los ajustes: transferencia de peso

Se usa **transferencia**, no suma.

```
1. Se parte de los 4 pesos iguales (suma = 1,0).
2. Cada ajuste pide un incremento δ en su dimensión.
3. Esa masa se toma proporcionalmente de las dimensiones NO ajustadas.
4. La suma sigue siendo 1,0 en todo momento.
```

Ejemplo real (calculado con el módulo): familia con 2 hijos, 40 años, transporte público.

```
Incrementos pedidos:  educación +0,15  (2 hijos: +0,10  +  edad <= 45: +0,05)
                     conectividad +0,05  (transporte público)
Masa total a transferir: 0,20  →  se toma de sanidad y vivienda, a partes iguales

Resultado:
  educación      0,25 → 0,40
  conectividad   0,25 → 0,30
  sanidad        0,25 → 0,15
  vivienda       0,25 → 0,15
                       ─────
                        1,00
```

**Aviso:** la vivienda baja de 0,25 a 0,15. Es el coste de priorizar educación y
transporte, y es visible. Por eso hay un **tope de transferencia total del 20 %
(0,20)**: si la suma de ajustes lo supera, se escalan proporcionalmente y nunca
se deja una dimensión por debajo de 0,05.

**Ventaja de la transferencia:** cada ajuste tiene un efecto acotado y explicable.
Con suma + normalización, el efecto depende del número de dimensiones y no se puede
afirmar cuánto sube realmente una dimensión.

---

## 5. Filtros duros (antes de ordenar)

| Filtro | Condición | Efecto |
|---|---|---|
| Precio | `precio_m2_compra` (o `precio_m2_alquiler`) ≤ `presupuesto_max_m2_eur` | Se excluye, con motivo |
| Tiempo | `tiempo_medio_madrid_minutos` ≤ `tiempo_max_commute_min` | Se excluye, con motivo |
| Colapso | ICA < 40 o alerta no compensatoria | **No se excluye**: se marca en rojo |

Decisión de producto: **informar, no ocultar**. Un municipio colapsado aparece con
etiqueta y motivo.

---

## 6. Esfuerzo financiero

Compra (sistema francés, 80 % financiado, 3,5 % anual, 30 años):
```
precio_vivienda    = precio_m2_compra × m2_deseados
importe_financiado = precio_vivienda × 0,80
r = 0,035 / 12      n = 360
cuota_mensual = importe_financiado × r / (1 − (1 + r)^(−n))
esfuerzo_pct  = cuota_mensual / ingresos_netos_hogar × 100
```

Alquiler:
```
cuota_mensual = precio_m2_alquiler × m2_deseados
esfuerzo_pct  = cuota_mensual / ingresos_netos_hogar × 100
```

| Esfuerzo | Etiqueta |
|---|---|
| ≤ 30 % | Asequible |
| 30–35 % | Ajustado |
| 35–45 % | Alto |
| > 45 % | No asequible |

---

## 7. Análisis de sensibilidad (obligatorio)

No afirmamos que los pesos sean correctos. Medimos si la conclusión depende de ellos.

```
1. Para cada dimensión, se perturba su peso ± su rango de sensibilidad.
2. Se recalcula el ranking completo.
3. Se mide:
     - variación del top-5 (cuántos municipios entran/salen)
     - correlación de orden de Spearman con el ranking base
4. Se reporta: "el orden se mantiene en X de Y perturbaciones".
```

**Criterio de lectura:**
- Spearman ≥ 0,90 y top-5 estable → conclusión robusta.
- Spearman < 0,70 o top-5 cambia en más de 2 posiciones → conclusión sensible,
  y se declara qué municipios dependen del peso.

---

## 8. Salida por municipio

```json
{
  "nombre": "Robledo de Chavela",
  "afinidad_pct": 0.0,
  "pesos_aplicados": {"vivienda": 0.25, "educacion": 0.30, "conectividad": 0.30, "sanidad": 0.15},
  "desglose": {"vivienda": 0.0, "educacion": 0.0, "conectividad": 0.0, "sanidad": 0.0},
  "esfuerzo_financiero_pct": 0.0,
  "etiqueta_esfuerzo": "Asequible",
  "semaforo_ica": "ALTA",
  "alerta_no_compensatoria": false,
  "motivo_colapso": null,
  "fortalezas": ["..."],
  "carencias": ["..."],
  "riesgo_futuro": {"poblacion_2031": 0, "anio_saturacion": null, "alerta": "ESTABLE"},
  "motivo_recomendacion": "texto en español sin causalidad"
}
```

---

## 9. Criterios de aceptación

- [ ] Los pesos suman 1,0 exactamente para cualquier combinación de entradas.
- [ ] Ningún peso es negativo ni supera 0,60.
- [ ] La transferencia total está acotada al 20 %.
- [ ] Un perfil con 3 hijos pesa más en educación que uno sin hijos.
- [ ] Un perfil de 70 años pesa más en sanidad.
- [ ] Bajar el presupuesto nunca incorpora municipios más caros.
- [ ] Un municipio colapsado aparece marcado y con motivo.
- [ ] El esfuerzo financiero se explica con la fórmula y el tipo declarado.
- [ ] El ranking no modifica el ICA ni la regla no compensatoria.
- [ ] El análisis de sensibilidad se ejecuta y reporta Spearman y estabilidad del top-5.
