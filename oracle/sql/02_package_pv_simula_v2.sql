-- =============================================================================
-- Datathon 2026 • Pueblos con vida (App APEX 215 - Página 10)
-- 02_package_pv_simula_v2.sql
-- Package PL/SQL PV_SIMULA (Especificación y Cuerpo)
-- Ejecuta M3 y M4 ONNX in-database y devuelve JSON para el proceso AJAX "SIMULAR"
-- =============================================================================

CREATE OR REPLACE PACKAGE pv_simula_pkg AS
    -- Función principal consumida por el proceso AJAX SIMULAR de APEX
    FUNCTION simular_escenario(
        p_cod_ine       IN VARCHAR2,
        p_anio_destino  IN NUMBER   DEFAULT 2030,
        p_choque_hab    IN NUMBER   DEFAULT 0,
        p_choque_viv    IN NUMBER   DEFAULT 0,
        p_choque_salud  IN NUMBER   DEFAULT 0,
        p_choque_esc    IN NUMBER   DEFAULT 0,
        p_choque_par    IN NUMBER   DEFAULT 0
    ) RETURN CLOB;
END pv_simula_pkg;
/

CREATE OR REPLACE PACKAGE BODY pv_simula_pkg AS

    FUNCTION simular_escenario(
        p_cod_ine       IN VARCHAR2,
        p_anio_destino  IN NUMBER   DEFAULT 2030,
        p_choque_hab    IN NUMBER   DEFAULT 0,
        p_choque_viv    IN NUMBER   DEFAULT 0,
        p_choque_salud  IN NUMBER   DEFAULT 0,
        p_choque_esc    IN NUMBER   DEFAULT 0,
        p_choque_par    IN NUMBER   DEFAULT 0
    ) RETURN CLOB
    IS
        v_res JSON_OBJECT_T := JSON_OBJECT_T();
        v_antes JSON_OBJECT_T := JSON_OBJECT_T();
        v_despues JSON_OBJECT_T := JSON_OBJECT_T();

        -- Variables base 2025
        v_pob_base          NUMBER;
        v_log_pob           NUMBER;
        v_horizonte         NUMBER;
        v_factor_m3         NUMBER;
        v_pob_proyectada    NUMBER;
        v_pob_simulada      NUMBER;

        -- Métricas e indicadores del municipio
        r_mun               PV_MUN_BASE%ROWTYPE;

        -- Dimensiones M4 Antes
        v_dim_dem_ant       NUMBER := 50;
        v_dim_viv_ant       NUMBER := 50;
        v_dim_ser_ant       NUMBER := 50;
        v_dim_acc_ant       NUMBER := 50;
        v_dim_emp_ant       NUMBER := 50;
        v_indice_ant        NUMBER := 50;
        v_limit_ant         VARCHAR2(50) := 'Equilibrado';

        -- Dimensiones M4 Después (Simuladas)
        v_dim_dem_des       NUMBER := 50;
        v_dim_viv_des       NUMBER := 50;
        v_dim_ser_des       NUMBER := 50;
        v_dim_acc_des       NUMBER := 50;
        v_dim_emp_des       NUMBER := 50;
        v_indice_des        NUMBER := 50;
        v_limit_des         VARCHAR2(50) := 'Equilibrado';
        v_delta_indice      NUMBER;

        -- Ratios simulados para entrada a M4
        v_sim_log_pob       NUMBER;
        v_sim_salud_10k     NUMBER;
        v_sim_esc_1000      NUMBER;
        v_sim_par_1000      NUMBER;
        v_sim_tel_1000      NUMBER;
        v_sim_viv_1000      NUMBER;

        -- Función auxiliar para obtener argmin (factor limitante)
        FUNCTION calcular_factor_limitante(
            d_dem NUMBER, d_viv NUMBER, d_ser NUMBER, d_acc NUMBER, d_emp NUMBER
        ) RETURN VARCHAR2 IS
            v_min NUMBER := d_dem;
            v_nom VARCHAR2(50) := 'Demografía';
        BEGIN
            IF d_viv < v_min THEN v_min := d_viv; v_nom := 'Vivienda'; END IF;
            IF d_ser < v_min THEN v_min := d_ser; v_nom := 'Servicios'; END IF;
            IF d_acc < v_min THEN v_min := d_acc; v_nom := 'Accesibilidad'; END IF;
            IF d_emp < v_min THEN v_min := d_emp; v_nom := 'Empleo'; END IF;
            RETURN v_nom;
        END;

    BEGIN
        -- 1. Cargar datos del estado base (2025)
        SELECT * INTO r_mun FROM PV_MUN_BASE WHERE cod_ine = LPAD(p_cod_ine, 5, '0');
        v_pob_base := r_mun.poblacion_total;
        v_horizonte := GREATEST(1, LEAST(10, p_anio_destino - 2025));

        -- 2. Ejecutar M3 Demografía: Predicción de crecimiento hacia el año destino
        BEGIN
            SELECT EXP(PREDICTION(M3_DEMOGRAFIA USING 
                LN(v_pob_base) AS log_pob,
                r_mun.cagr_1 AS cagr_1,
                r_mun.cagr_3 AS cagr_3,
                r_mun.cagr_5 AS cagr_5,
                r_mun.cagr_10 AS cagr_10,
                r_mun.vol_5 AS vol_5,
                r_mun.edad_media AS edad_media,
                r_mun.pct_menor_15 AS pct_menor_15,
                r_mun.pct_65mas AS pct_65mas,
                r_mun.reemplazamiento AS reemplazamiento,
                r_mun.paro_100 AS paro_100,
                r_mun.var_paro AS var_paro,
                r_mun.migracion_3y AS migracion_3y,
                r_mun.distancia_sol_km AS distancia_sol_km,
                r_mun.tiempo_centro_min AS tiempo_centro_min,
                v_horizonte AS horizonte
            )) INTO v_factor_m3 FROM DUAL;
        EXCEPTION WHEN OTHERS THEN
            v_factor_m3 := 1.0; -- Fallback determinista si ONNX no está cargado aún
        END;

        v_pob_proyectada := ROUND(v_pob_base * v_factor_m3);
        v_pob_simulada   := v_pob_proyectada + NVL(p_choque_hab, 0);

        -- 3. Dimensiones ANTES (corte base de la tabla)
        v_dim_dem_ant := NVL(r_mun.dim_dem, 50);
        v_dim_viv_ant := NVL(r_mun.dim_viv, 50);
        v_dim_ser_ant := NVL(r_mun.dim_ser, 50);
        v_dim_acc_ant := NVL(r_mun.dim_acc, 50);
        v_dim_emp_ant := NVL(r_mun.dim_emp, 50);
        v_indice_ant  := ROUND(0.20*v_dim_dem_ant + 0.15*v_dim_viv_ant + 0.25*v_dim_ser_ant + 0.25*v_dim_acc_ant + 0.15*v_dim_emp_ant, 1);
        v_limit_ant   := calcular_factor_limitante(v_dim_dem_ant, v_dim_viv_ant, v_dim_ser_ant, v_dim_acc_ant, v_dim_emp_ant);

        -- 4. Construir ratios simulados con los choques de inversión / dotación
        v_sim_log_pob   := LN(GREATEST(v_pob_simulada, 1));
        v_sim_salud_10k := ((r_mun.centros_salud + NVL(p_choque_salud, 0)) / GREATEST(v_pob_simulada, 1)) * 10000;
        v_sim_esc_1000  := ((r_mun.escuelas_alta + NVL(p_choque_esc, 0)) / GREATEST(v_pob_simulada, 1)) * 1000;
        v_sim_par_1000  := ((r_mun.paradas_total + NVL(p_choque_par, 0)) / GREATEST(v_pob_simulada, 1)) * 1000;
        v_sim_tel_1000  := (r_mun.lineas_telefono / GREATEST(v_pob_simulada, 1)) * 1000;
        v_sim_viv_1000  := ((r_mun.unidades_residenciales + NVL(p_choque_viv, 0)) / GREATEST(v_pob_simulada, 1)) * 1000;

        -- 5. Ejecutar M4 Capacidad: Scoring de las 5 dimensiones
        BEGIN
            SELECT 
                PREDICTION(M4_CAPACIDAD, 'dim_dem' USING v_sim_log_pob AS log_pob, v_sim_salud_10k AS salud_10k, v_sim_esc_1000 AS escuelas_1000, v_sim_par_1000 AS paradas_1000, v_sim_tel_1000 AS telefono_1000, v_sim_viv_1000 AS viv_1000, r_mun.crec_10y_pct AS crec_10y_pct, r_mun.pct_menor_15 AS pct_menor_15, r_mun.pct_65mas AS pct_65mas, r_mun.reemplazamiento AS reemplazamiento, r_mun.paro_100 AS paro_100, r_mun.var_paro AS var_paro, r_mun.tiempo_centro_min AS tiempo_centro_min, r_mun.distancia_sol_km AS distancia_sol_km),
                PREDICTION(M4_CAPACIDAD, 'dim_viv' USING v_sim_log_pob AS log_pob, v_sim_salud_10k AS salud_10k, v_sim_esc_1000 AS escuelas_1000, v_sim_par_1000 AS paradas_1000, v_sim_tel_1000 AS telefono_1000, v_sim_viv_1000 AS viv_1000, r_mun.crec_10y_pct AS crec_10y_pct, r_mun.pct_menor_15 AS pct_menor_15, r_mun.pct_65mas AS pct_65mas, r_mun.reemplazamiento AS reemplazamiento, r_mun.paro_100 AS paro_100, r_mun.var_paro AS var_paro, r_mun.tiempo_centro_min AS tiempo_centro_min, r_mun.distancia_sol_km AS distancia_sol_km),
                PREDICTION(M4_CAPACIDAD, 'dim_ser' USING v_sim_log_pob AS log_pob, v_sim_salud_10k AS salud_10k, v_sim_esc_1000 AS escuelas_1000, v_sim_par_1000 AS paradas_1000, v_sim_tel_1000 AS telefono_1000, v_sim_viv_1000 AS viv_1000, r_mun.crec_10y_pct AS crec_10y_pct, r_mun.pct_menor_15 AS pct_menor_15, r_mun.pct_65mas AS pct_65mas, r_mun.reemplazamiento AS reemplazamiento, r_mun.paro_100 AS paro_100, r_mun.var_paro AS var_paro, r_mun.tiempo_centro_min AS tiempo_centro_min, r_mun.distancia_sol_km AS distancia_sol_km),
                PREDICTION(M4_CAPACIDAD, 'dim_acc' USING v_sim_log_pob AS log_pob, v_sim_salud_10k AS salud_10k, v_sim_esc_1000 AS escuelas_1000, v_sim_par_1000 AS paradas_1000, v_sim_tel_1000 AS telefono_1000, v_sim_viv_1000 AS viv_1000, r_mun.crec_10y_pct AS crec_10y_pct, r_mun.pct_menor_15 AS pct_menor_15, r_mun.pct_65mas AS pct_65mas, r_mun.reemplazamiento AS reemplazamiento, r_mun.paro_100 AS paro_100, r_mun.var_paro AS var_paro, r_mun.tiempo_centro_min AS tiempo_centro_min, r_mun.distancia_sol_km AS distancia_sol_km),
                PREDICTION(M4_CAPACIDAD, 'dim_emp' USING v_sim_log_pob AS log_pob, v_sim_salud_10k AS salud_10k, v_sim_esc_1000 AS escuelas_1000, v_sim_par_1000 AS paradas_1000, v_sim_tel_1000 AS telefono_1000, v_sim_viv_1000 AS viv_1000, r_mun.crec_10y_pct AS crec_10y_pct, r_mun.pct_menor_15 AS pct_menor_15, r_mun.pct_65mas AS pct_65mas, r_mun.reemplazamiento AS reemplazamiento, r_mun.paro_100 AS paro_100, r_mun.var_paro AS var_paro, r_mun.tiempo_centro_min AS tiempo_centro_min, r_mun.distancia_sol_km AS distancia_sol_km)
            INTO v_dim_dem_des, v_dim_viv_des, v_dim_ser_des, v_dim_acc_des, v_dim_emp_des
            FROM DUAL;
        EXCEPTION WHEN OTHERS THEN
            v_dim_dem_des := GREATEST(0, LEAST(100, v_dim_dem_ant - (NVL(p_choque_hab,0) / 200)));
            v_dim_viv_des := GREATEST(0, LEAST(100, v_dim_viv_ant + (NVL(p_choque_viv,0) * 2) - (NVL(p_choque_hab,0) / 300)));
            v_dim_ser_des := GREATEST(0, LEAST(100, v_dim_ser_ant + (NVL(p_choque_salud,0) * 10) + (NVL(p_choque_esc,0) * 5) - (NVL(p_choque_hab,0) / 250)));
            v_dim_acc_des := GREATEST(0, LEAST(100, v_dim_acc_ant + (NVL(p_choque_par,0) * 3)));
            v_dim_emp_des := v_dim_emp_ant;
        END;

        -- Regla de saturación crítica: truncamiento
        IF v_dim_ser_des < 25 OR v_dim_viv_des < 20 THEN
            v_indice_des := LEAST(39.9, ROUND(0.20*v_dim_dem_des + 0.15*v_dim_viv_des + 0.25*v_dim_ser_des + 0.25*v_dim_acc_des + 0.15*v_dim_emp_des, 1));
        ELSE
            v_indice_des := ROUND(0.20*v_dim_dem_des + 0.15*v_dim_viv_des + 0.25*v_dim_ser_des + 0.25*v_dim_acc_des + 0.15*v_dim_emp_des, 1);
        END IF;

        v_limit_des    := calcular_factor_limitante(v_dim_dem_des, v_dim_viv_des, v_dim_ser_des, v_dim_acc_des, v_dim_emp_des);
        v_delta_indice := ROUND(v_indice_des - v_indice_ant, 1);

        -- 6. Construir JSON de salida
        v_antes.put('poblacion', v_pob_base);
        v_antes.put('indice', v_indice_ant);
        v_antes.put('dim_dem', ROUND(v_dim_dem_ant, 1));
        v_antes.put('dim_viv', ROUND(v_dim_viv_ant, 1));
        v_antes.put('dim_ser', ROUND(v_dim_ser_ant, 1));
        v_antes.put('dim_acc', ROUND(v_dim_acc_ant, 1));
        v_antes.put('dim_emp', ROUND(v_dim_emp_ant, 1));
        v_antes.put('factor_limitante', v_limit_ant);

        v_despues.put('anio', p_anio_destino);
        v_despues.put('poblacion', v_pob_simulada);
        v_despues.put('indice', v_indice_des);
        v_despues.put('dim_dem', ROUND(v_dim_dem_des, 1));
        v_despues.put('dim_viv', ROUND(v_dim_viv_des, 1));
        v_despues.put('dim_ser', ROUND(v_dim_ser_des, 1));
        v_despues.put('dim_acc', ROUND(v_dim_acc_des, 1));
        v_despues.put('dim_emp', ROUND(v_dim_emp_des, 1));
        v_despues.put('factor_limitante', v_limit_des);

        v_res.put('municipio', r_mun.municipio);
        v_res.put('cod_ine', r_mun.cod_ine);
        v_res.put('delta_indice', v_delta_indice);
        v_res.put('antes', v_antes);
        v_res.put('despues', v_despues);

        RETURN v_res.to_clob();
    END simular_escenario;

END pv_simula_pkg;
/
