# Estructura Final de la Tesis: Pronóstico de Rentas Cedidas con Machine Learning

Este documento integra todos los hallazgos, metodologías y resultados técnicos generados en el proyecto para conformar el documento final de tesis.

## 1. Introducción
*   **Contexto**: Importancia de las rentas cedidas para la financiación de la salud y educación en los departamentos.
*   **Problema**: Volatilidad, estacionalidad y falta de herramientas precisas de proyección.
*   **Objetivo**: Desarrollar comparar modelos de series de tiempo (ARIMA/SARIMA) y ML (XGBoost, LSTM) para mejorar la precisión del pronóstico.
*   **Alcance**: Periodo 2020-2025 (Entrenamiento) y proyección 2026. Análisis nacional y municipal (Top contribuidores).

## 2. Marco Teórico y Referencial
*   **Series de Tiempo**: Estacionariedad, Pruebas de Raíz Unitaria (ADF, KPSS), Estacionalidad.
*   **Metodología Box-Jenkins**: Identificación, Estimación, Validación.
*   **Modelos Avanzados**:
    *   **SARIMA**: Para capturar estacionalidad anual.
    *   **Prophet**: Para tendencias no lineales y puntos de cambio.
    *   **Redes Neuronales (LSTM)**: Para capturar secuencias complejas no lineales.
    *   **XGBoost**: Enfoque de regresión supervisada con lags.

## 3. Metodología (CRISP-DM / OSEMN)
Se siguió un flujo de trabajo estructurado en 6 fases:

1.  **Entendimiento del Negocio**: Definición de variables clave (Recaudo, Vigencia, Entidad).
2.  **Entendimiento de los Datos**:
    *   *Fuente:* Archivo parquet histórico (2018-2025).
    *   *Limpieza:* (`01_limpieza_inicial.py`) Manejo de recaudos negativos (neteo mensual), unificación de fechas.
3.  **Preparación de los Datos**:
    *   *Ingeniería de Características:* (`02_feature_engineering.py`) Creación de lags, medias móviles, variables calendario.
    *   *Nivel de Agregación:* Nacional, Departamental y Municipal.
4.  **Modelado**: (`03_modelos.py`) Entrenamiento de 5 familias de modelos.
5.  **Evaluación**: (`04_validacion_final.py`, `07_diagnostico_arima_sarima.py`).
    *   Métricas: MAPE, RMSE, MAE.
    *   Horizonte: Validación cruzada temporal (Walk-forward).
6.  **Despliegue**: (`06_dashboards.py`, `09_analisis_municipal.py`) Tableros de control y reportes automáticos.

## 4. Análisis de Resultados

### 4.1 Caracterización de los Datos (Descriptiva)
*   **Tendencia General**: Crecimiento sostenido con picos estacionales en diciembre/enero.
*   **Concentración Municipal (Pareto)**:
    *   Ver gráfico: [Pareto Top 20](pareto_top_20.png)
    *   *Hallazgo:* El 80% del recaudo proviene del ~15% de los municipios. Bogotá, Medellín y Barranquilla lideran.
*   **Estacionalidad**:
    *   Ver gráfico: [Descomposición Estacional](dashboard_horizonte_mensual.png)
    *   Confirmación de componente anual fuerte.

### 4.2 Diagnóstico y Selección de Modelos
*   **Comparativa ARIMA vs SARIMA**:
    *   El componente estacional (S) fue significativo. SARIMA(1,1,1)(1,1,1,12) superó a ARIMA simple.
    *   *Evidencia:* Tabla de AIC en `results/reports/informe_tecnico_datos_modelos.md`.

### 4.3 Desempeño de los Modelos (Ranking)
Según el RMSE y MAPE en el conjunto de prueba (2025):

1.  **SARIMA**: Mejor desempeño global para capturar los picos estacionales. *Recomendado para presupuesto anual.*
2.  **XGBoost**: Muy preciso en la tendencia media, pero suaviza los extremos.
3.  **Prophet**: Útil para detectar cambios de tendencia, pero con mayor error en la estacionalidad intra-anual simple.
4.  **LSTM**: Requiere más datos históricos para converger; tendió al sobreajuste en series cortas municipales.

*   Ver gráfico: [Comparativa Serie de Tiempo](serie_tiempo_comparativa_top5.png)

### 4.4 Análisis de Volatilidad y Riesgo (NUEVO)
*   **Incertidumbre Histórica**: El cálculo de la desviación estándar móvil revela que la volatilidad no es constante, sino que se agrupa en clústers (heterocedasticidad condicionada).
*   **Riesgo de Liquidez (VaR)**: Se estimó un Value at Risk (VaR 95%) que alerta sobre contracciones mensuales extremas.
*   **Proyección de Incertidumbre**: El *Fan Chart* muestra la dispersión de escenarios futuros.
    *   Ver gráfico: [Fan Chart](fan_chart_incertidumbre.png)

### 4.5 Modelos Híbridos/Ensemble (NUEVO - FASE 4)
*   **Metodología Stacking**: Combinación de predicciones de SARIMAX, Prophet y XGBoost usando un meta-modelo (Ridge Regression).
*   **Resultado**: Reducción del error (MAPE) al compensar las debilidades individuales (e.g., suavizado de XGBoost vs estacionalidad de SARIMAX).
*   **Validación**: Uso de *Walk-Forward Validation* para asegurar robustez temporal.
    *   Ver gráfico: [Ensemble Forecast](predictions/ensemble_forecast.png)


## 5. Pronósticos 2026
*   **Proyección Nacional**: Se estima un crecimiento del X% (ver `results/tables/dataset_modelos_mensual.csv`).
*   **Proyección Municipal**:
    *   Bogotá: [Pronóstico](prediccion_Distrito_BOGOTÁ.png)
    *   Medellín: [Pronóstico](prediccion_Municipio_MEDELLIN.png)

## 6. Discusión
*   **Sincronización Fiscal**: El análisis de correlación (`correlacion_top_municipios.png`) muestra que los grandes centros urbanos tienen ciclos de recaudo sincronizados, lo que sugiere políticas de fiscalización centralizadas efectivas.
*   **Impacto de la Calidad de Datos**: La limpieza de "recaudos negativos" fue crucial para evitar sesgos en la media móvil.

## 7. Conclusiones y Recomendaciones
*   **Conclusión Principal**: Los modelos estacionales (SARIMA) son superiores para este tipo de renta debido a la fuerte periodicidad tributaria.
*   **Recomendación 1**: Adoptar SARIMA para la planeación financiera estratégica anual.
*   **Recomendación 2**: Usar XGBoost para simulaciones de escenarios (shocks económicos) gracias a su flexibilidad con variables exógenas.
*   **Recomendación 3**: Focalizar la fiscalización en los meses valle identificados en el análisis estacional.

## Anexos Técnicos
*   Repositorio GitHub: [Enlace]
*   Scripts Python:
    *   `01_limpieza_inicial.py` (ETL)
    *   `07_diagnostico_arima_sarima.py` (Validación estadística)
    *   `03_modelos.py` (Entrenamiento ML)
    *   `09_analisis_municipal.py` (Desglose territorial)
