# Plan de Mejora y Profundización de Tesis

Este documento propone una hoja de ruta para **refactorizar técnicamente el proyecto** y **ampliar el alcance académico** de la tesis con enfoques analíticos avanzados.

---

## 1. Refactorización Técnica (Mejora del Código)

El código actual es funcional pero puede ser más robusto, modular y fácil de mantener.

### A. Modularización
*   **Problema:** Funciones repetidas (ej. carga de datos, cálculo de MAPE) en múltiples scripts.
*   **Solución:** Crear un módulo `utils.py` compartido.
    *   `utils.io`: Carga y guardado de datos (CSV/Excel).
    *   `utils.metrics`: Funciones de error (MAPE, RMSE, MAE).
    *   `utils.plots`: Estilos gráficos unificados para la tesis.

### B. Gestión de Configuración
*   **Problema:** Rutas de archivos y parámetros (fechas, hiperparámetros) "quemados" en el código.
*   **Solución:** Implementar `config.yaml` o `settings.py`.
    *   Centralizar rutas de datos (`data/raw`, `data/processed`).
    *   Definir rangos de fechas globales (TRAIN_START, TRAIN_END, TEST_YEAR).

### C. Pipeline de Ejecución
*   **Problema:** Ejecución manual script por script.
*   **Solución:** Crear un orquestador `main.py` o `run_pipeline.py` que ejecute todo el flujo en orden: ETL -> Feature Eng -> Modelos -> Reportes.

---

## 2. Nuevos Enfoques Analíticos (Complemento Académico)

Para elevar el nivel de la tesis, se recomienda explorar estas dimensiones adicionales:

### A. Modelado de Volatilidad (GARCH)
*   **Hipótesis:** La varianza del recaudo no es constante (heterocedasticidad). Los meses de alto recaudo también son más volátiles.
*   **Acción:** Ajustar un modelo GARCH a los residuos del modelo SARIMA.
*   **Valor Tesis:** Demostrar que se puede predecir no solo el valor esperado, sino el **riesgo** del recaudo.

### B. Variables Exógenas (Impacto Macroeconómico)
*   **Hipótesis:** El recaudo depende del ciclo económico.
*   **Acción:** Incorporar variables como **PIB Semestral**, **Tasa de Desempleo** o **IPC** como regresores externos en SARIMAX (SARIMAX con X) o XGBoost.
*   **Valor Tesis:** Validar la sensibilidad de las rentas cedidas a la economía nacional.

### C. Deep Learning Moderno (Transformers / Tion)
*   **Hipótesis:** Los modelos de atención (Transformers) pueden capturar dependencias a largo plazo mejor que LSTM.
*   **Acción:** Implementar un modelo simple basado en **Temporal Fusion Transformer (TFT)** (librería `darts` o `pytorch-forecasting`).
*   **Valor Tesis:** Comparar el estado del arte (SOTA) vs modelos tradicionales.

### D. Análisis de Intervención (Causal Inference)
*   **Hipótesis:** Reformas tributarias o eventos específicos (COVID-19) causaron quiebres estructurales.
*   **Acción:** Usar librería `CausalImpact` (Google) para medir el efecto de la pandemia en 2020-2021.
*   **Valor Tesis:** Cuantificar la pérdida de recaudo atribuible a crisis externas.

---

## 3. Hoja de Ruta Sugerida

1.  **Fase 1: Refactorización (1 día)**
    *   Extraer funciones comunes a `utils/`.
    *   Crear archivo de configuración.
    *   Limpiar imports y comentarios.

2.  **Fase 2: Profundización (2-3 días)**
    *   Seleccionar **UNA** de las nuevas áreas (recomiendo **Variables Exógenas** o **Volatilidad** por su relevancia fiscal).
    *   Implementar script experimental (`10_analisis_avanzado.py`).
    *   Agregar capítulo de "Análisis Exploratorio Avanzado" a la tesis.

3.  **Fase 3: Actualización Final**
    *   Regenerar reporte final con las nuevas métricas.
    *   Actualizar repositorio.
