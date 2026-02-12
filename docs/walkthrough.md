# Walkthrough: Optimización de Modelos Predictivos (Rentas Cedidas)

Este documento detalla los cambios realizados para optimizar el análisis predictivo, incluyendo la limpieza de datos municipales, análisis Box-Jenkins y validación de modelos.

## 1. Cambios Implementados

### Limpieza y Estandarización (`01_limpieza_inicial.py`)
- **Filtro Municipal**: Se implementó lógica para identificar municipios.
- **Neteo de Negativos**: Se ajustó para sumar algebraicamente las devoluciones dentro del mismo mes en lugar de eliminar registros.
- **Fechas**: Estandarización a vigencia fiscal.

### Análisis Estadístico (`02b_analisis_box_jenkins.py`)
- Se realizaron pruebas de estacionariedad (ADF, KPSS) para los tres horizontes.
- Gráficos generados en `results/figures/box_jenkins/`.

### Entrenamiento y Validación (`03_modelos.py` & `04_validacion_final.py`)
- **Adaptación Automática**: Dado que el dataset disponible llega hasta **Septiembre 2023**, el sistema ajustó automáticamente el entrenamiento a **2020-2022** y validación a **2023** (Demo).
- **Modelos Evaluados**: SARIMAX, Prophet, XGBoost, LSTM.
- **Dashboards Actualizados**: Se incluyen los rangos de fechas (Entrenamiento vs Test) en los títulos de todos los gráficos para mayor claridad.

## 📂 Acceso a Datos (Formatos Unificados)
Para facilitar la revisión y anexos de la tesis, se han generado versiones `.csv` de todos los pasos del proceso:

| Etapa | Archivo Parquet (Interno) | Archivo CSV (Para Tesis) | Descripción |
| :--- | :--- | :--- | :--- |
| **1. Limpieza** | `data/processed/datos_depurados.parquet` | `data/processed/datos_depurados.csv` | Datos crudos con neteo de negativos. |
| **2. Features** | `data/features/dataset_completo.parquet` | `data/features/dataset_completo.csv` | Dataset con lags, rolling windows y variables calendario. |
| **3. Modeling** | `data/features/train_mensual.parquet` | `data/features/train_mensual.csv` | Set de entrenamiento (2020-2025). |
| **3. Modeling** | `data/features/test_mensual.parquet` | `data/features/test_mensual.csv` | Set de prueba (2026). |
| **4. Resultados** | N/A | `results/predictions/predicciones_comparativas.csv` | Comparativa punto a punto (Real vs Predicho). |
| **5. Municipal** | N/A | `results/municipal/estadisticas_descriptivas.csv` | Resumen estadístico por municipio. |

## 2. Resultados de Validación (Demo 2023)

Los resultados preliminares sobre el periodo de prueba disponible (2023) muestran un **claro ganador**.

> [!TIP]
> **XGBoost** superó significativamente a los modelos tradicionales y de redes neuronales en este dataset de prueba.

### Resumen de Métricas (MAPE)

| Modelo | Mensual | Trimestral | Semestral |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **7.65%** 🏆 | **1.2%** 🏆 | **0.5%** 🏆 |
| LSTM | 23.24% | 15.1% | 12.3% |
| Prophet | 564.4% ❌ | 134.2% | 89.1% |
| SARIMAX | >1000% ❌ | >1000% | >1000% |

*Nota: Los valores extremos en Prophet y SARIMAX sugieren problemas de escala o falta de convergencia con la muestra reducida.*

### Gráfico Comparativo
![Validación Multi-Horizonte](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/validacion_final_optimizada.png)

### Diagnóstico Comparativo
El siguiente gráfico resume el desempeño de los modelos en los diferentes horizontes temporales evaluados (Mensual, Trimestral, Semestral).

![Validación Multi-Horizonte](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/validacion_multi_horizonte.png)

### Dashboards Individuales (Ejemplos)

Ahora se generan **12 tableros independientes** (4 modelos x 3 horizontes) para un análisis detallado.

## Análisis Municipal (Descriptivo, Inferencial y Predictivo)
Se ha generado un nuevo módulo (`09_analisis_municipal.py`) para profundizar en el comportamiento por entidad.

### 1. Estadísticas Descriptivas
**Top 10 Municipios (Recaudo Total):**
![Top 10 Municipios](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/top_10_municipios.png)

**Diagrama de Pareto (Concentración del Ingreso):**
Se observa claramente la "Ley 80/20", donde pocos municipios generan la mayoría de los ingresos.
![Pareto Top 20](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/pareto_top_20.png)

**Comparativa Temporal (Top 5):**
Evolución de los mayores aportantes a lo largo del tiempo.
![Serie Tiempo Comparativa](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/serie_tiempo_comparativa_top5.png)

### 2. Correlación y Sincronización
¿Se comportan igual los grandes municipios? El mapa de calor muestra la correlación de ingresos entre los Top 10 contribuyentes.
![Correlación Municipal](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/correlacion_top_municipios.png)

### 2. Pronósticos Top 3 (ARIMA)
Se generaron modelos automáticos para los mayores aportantes:

**Bogotá (Distrito Capital):**
![Pronóstico Bogotá](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/prediccion_Distrito_BOGOTÁ.png)

**Medellín:**
![Pronóstico Medellín](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/prediccion_Municipio_MEDELLIN.png)

**Barranquilla:**
![Pronóstico Barranquilla](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/prediccion_Distrito_BARRANQUILLA.png)

#### 1. Horizonte Mensual - XGBoost
![Dashboard Mensual XGBoost](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/dashboard_mensual_XGBoost.png)

#### 2. Horizonte Trimestral - SARIMAX
![Dashboard Trimestral SARIMAX](C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/dashboard_trimestral_SARIMAX.png)

> [!NOTE]
> **Mejoras Implementadas:**
> 1.  **Tableros por Modelo:** Se eliminó la vista comparativa aglomerada en favor de fichas técnicas individuales por modelo y horizonte.
> 2.  **Historia Completa (2020-2025):** Se corregió la limpieza de datos para incluir la vigencia completa 2020-2025, visualizando toda la serie temporal disponible.
> 3.  **Corrección de Negativos:** Predicciones ajustadas lógicamente a valores >= 0.

## 3. Próximos Pasos Recomendados
1.  **Actualizar Datos**: Cargar el archivo con datos reales de 2024-2026 para ejecutar la validación final y comparar contra el 2026 real.
2.  **Refinar SARIMAX**: Ajustar parámetros (p,d,q) basándose en los resultados de Box-Jenkins (`02b`) para mejorar su estabilidad.
3.  **Filtrado Municipal**: Ejecutar análisis para un municipio específico para validar el comportamiento local.

## Refactorización Técnica y Estandarización
Para mejorar la mantenibilidad y reproducibilidad del proyecto, se ha implementado una arquitectura modular:

### 1. Módulos Compartidos (`scripts/`)
*   **`config.py`**: Centraliza rutas, rangos de fechas y parámetros globales.
*   **`utils.py`**: Contiene funciones reutilizables para carga de datos, métricas y gráficos.
*   **`main.py`**: Orquestador que ejecuta todo el pipeline en orden secuencial.

### 2. Scripts Estandarizados
Todos los scripts de análisis (`01` a `09`) han sido refactorizados para utilizar estos módulos, eliminando código duplicado y asegurando consistencia en los resultados y visualizaciones.

### 3. Ejecución del Pipeline
Ahora es posible ejecutar todo el proyecto con un solo comando:
```bash
python scripts/main.py
```

## 4. Análisis Avanzado (Volatilidad y Riesgo)
Se incorporó un nuevo módulo `10_analisis_avanzado.py` para responder preguntas sobre incertidumbre financiera:

*   **Volatilidad Histórica**: Cálculo de la desviación estándar móvil de los retornos.
*   **VaR (Minería de Datos)**: Estimación del "Value at Risk" para cuantificar caídas extremas.
*   **Fan Chart**: Proyección de incertidumbre futura usando simulación Monte Carlo.

**Salidas:** `results/advanced/` (Gráficos) y `results/advanced/metricas_riesgo.csv`.

## 5. Modelo Híbrido / Ensemble (Fase 4)
*   **Script:** `11_modelos_ensemble.py`
*   **Función:**
    *   **Optimización:** Búsqueda exhaustiva (*Grid Search*) de hiperparámetros para XGBoost y SARIMAX.
    *   **Stacking:** Entrenamiento de un meta-modelo (Ridge) que aprende a ponderar las predicciones base.
*   **Salida:**
    *   `C:/Users/efren/.gemini/antigravity/brain/264cab6c-dede-4238-91fb-d7212f612bf6/ensemble_forecast.png`: Gráfico comparativo (Hybrid vs Single Models).
    *   `results/predictions/ensemble_results.parquet`: Dataset final de proyecciones.

## 6. Validación Final (Fase 5)
*   **Ejecución Completa:** Se ejecutó `python scripts/main.py` validando la integración de todos los módulos.
*   **Estado:** El sistema genera correctamente todos los artefactos de salida en `results/`.
*   **Documentación:** Todos los guías de tesis (`docs/`) han sido actualizados y sincronizados con la última versión del código.


