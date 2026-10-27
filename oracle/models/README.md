# oracle/models — salida de los pipelines

Aquí dejan su resultado los scripts `scripts/run_m3_demografia.py` y
`scripts/run_m4_capacidad.py`:

- `m3_demografia.onnx` — factor de crecimiento poblacional por horizonte
- `m4_capacidad.onnx` — 5 dimensiones (dim_dem, dim_viv, dim_ser, dim_acc, dim_emp) para un estado municipal simulado

Además, junto a cada `.onnx` se escribe su ficha `<modelo>.json`
(features y orden exacto, métricas de backtest, hash del dataset, fecha).

Fichas duplicadas en `doc/modelos/` para el jurado. **Sin ficha no se sube a Oracle.**
