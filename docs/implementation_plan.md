# Plan de Implementación: Optimización Municipal y Validación 2026

## Objetivo
1.  **Habilitar Nivel Municipal**: Refactorizar la limpieza para identificar y filtrar datos municipales (`NombreBeneficiarioAportante`).
2.  **Validación Robusta (2020-2025 vs 2026)**: Optimizar el flujo para que todos los modelos y horizontes (Mensual, Trimestral, Semestral) se entrenen estrictamente con datos hasta 2025 y se validen contra el año 2026.

## Cambios Propuestos

### 1. Refactorización de `01_limpieza_inicial.py`
- **Identificación de Nivel**: Crear columna `TipoEntidad` ('Municipio', 'Departamento', 'Nacional').
    - Lógica: Buscar "MUNICIPIO", "ALCALDÍA", "DISTRITO" para marcar como Municipal.
- **Manejo de Negativos**: Restar de los ingresos en lugar de eliminarlos (Neteo Mensual).
- **Manejo de Fechas**: Estandarizar a `Vigencia-Mes` para consistencia fiscal.

### 2. Refactorización de `02_feature_engineering.py`
- **Filtrado Municipal**: Agregar opción para generar datasets depurados solo con registros `TipoEntidad == 'Municipal'`.
- **Ingeniería de Features**:
    - Asegurar que los lags y promedios móviles se calculen respetando los límites temporales (no data leakage del 2026).
- **Split Estricto**: 
    - **Train**: Enero 2020 - Diciembre 2025.
    - **Test**: Enero 2026 - Diciembre 2026 (o hasta donde haya datos reales).

### 2.1. Análisis Exploratorio Avanzado (Nuevo Script `02b_analisis_box_jenkins.py`)
- **Objetivo**: Determinar la naturaleza estocástica de las series para cada horizonte (Mensual, Trimestral, Semestral).
- **Metodología Box-Jenkins**:
    - **Estacionariedad**: Tests Dickey-Fuller Aumentado (ADF) y KPSS.
    - **Autocorrelación**: Gráficos ACF y PACF para identificar órdenes AR(p) y MA(q).
    - **Descomposición**: Análisis de tendencia, estacionalidad y residuos (STL).
- **Salida**: Reporte automático sugiriendo parámetros (p,d,q)(P,D,Q)s para SARIMAX y confirmando estacionalidad para Prophet/XGBoost.

### 3. Refactorización de `03_modelos.py`
- **Entrenamiento**: Ajustar modelos (SARIMAX, Prophet, XGBoost, LSTM) para entrenar solo con el conjunto `Train` (2020-2025).
- **Predicción**: Generar pronósticos para todo el horizonte 2026 (12 meses).

### 4. Validación Multi-Horizonte (`04_validacion_final.py`)
- **Análisis por Horizonte**:
    - **Mensual**: Comparar 12 puntos de 2026.
    - **Trimestral**: Agregar predicciones y real a 4 trimestres de 2026.
    - **Semestral**: Agregar a 2 semestres de 2026.
- **Métricas**: Calcular MAPE, RMSE, MAE para cada horizonte por separado.
- **Comparativa**: Generar tabla comparativa final para definir el "Mejor Modelo" y "Mejor Horizonte" según el caso de uso.

## Entregables
- Scripts actualizados (`01` a `04`).
- Reporte en `results/metrics/resumen_validacion.txt` con la evaluación del 2026.
- Gráficos en `results/figures/` mostrando el ajuste 2020-2025 y el pronóstico 2026.
