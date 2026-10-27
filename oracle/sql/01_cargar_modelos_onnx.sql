-- =============================================================================
-- Datathon 2026 • Pueblos con vida (App APEX 215)
-- 01_cargar_modelos_onnx.sql
-- Carga de modelos ONNX (M3 Demografía y M4 Capacidad) en Oracle Database 23ai
-- Utiliza DBMS_VECTOR.LOAD_ONNX_MODEL con metadatos JSON exactos.
-- =============================================================================

-- Asegurar directorio donde residen los binarios ONNX exportados
-- CREATE OR REPLACE DIRECTORY ONNX_DIR AS '/opt/oracle/models';

BEGIN
    -- 1. Eliminar versiones previas si existen
    BEGIN
        DBMS_VECTOR.DROP_ONNX_MODEL(model_name => 'M3_DEMOGRAFIA');
    EXCEPTION WHEN OTHERS THEN NULL;
    END;

    BEGIN
        DBMS_VECTOR.DROP_ONNX_MODEL(model_name => 'M4_CAPACIDAD');
    EXCEPTION WHEN OTHERS THEN NULL;
    END;

    -- 2. Cargar M3 · Demografía (16 features -> log_factor de crecimiento)
    DBMS_VECTOR.LOAD_ONNX_MODEL(
        directory  => 'ONNX_DIR',
        file_name  => 'm3_demografia.onnx',
        model_name => 'M3_DEMOGRAFIA',
        metadata   => JSON('{
            "function": "regression",
            "regressionOutput": "variable",
            "input": {
                "input": [
                    "log_pob",
                    "cagr_1",
                    "cagr_3",
                    "cagr_5",
                    "cagr_10",
                    "vol_5",
                    "edad_media",
                    "pct_menor_15",
                    "pct_65mas",
                    "reemplazamiento",
                    "paro_100",
                    "var_paro",
                    "migracion_3y",
                    "distancia_sol_km",
                    "tiempo_centro_min",
                    "horizonte"
                ]
            }
        }')
    );

    -- 3. Cargar M4 · Capacidad / Vitalidad (14 features -> 5 dimensiones)
    DBMS_VECTOR.LOAD_ONNX_MODEL(
        directory  => 'ONNX_DIR',
        file_name  => 'm4_capacidad.onnx',
        model_name => 'M4_CAPACIDAD',
        metadata   => JSON('{
            "function": "multitask_regression",
            "regressionOutput": "variable",
            "input": {
                "input": [
                    "log_pob",
                    "salud_10k",
                    "escuelas_1000",
                    "paradas_1000",
                    "telefono_1000",
                    "viv_1000",
                    "crec_10y_pct",
                    "pct_menor_15",
                    "pct_65mas",
                    "reemplazamiento",
                    "paro_100",
                    "var_paro",
                    "tiempo_centro_min",
                    "distancia_sol_km"
                ]
            }
        }')
    );
END;
/
