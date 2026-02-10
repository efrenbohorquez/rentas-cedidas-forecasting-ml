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

### Dashboards Individuales (Ejemplos)

Ahora se generan **12 tableros independientes** (4 modelos x 3 horizontes) para un análisis detallado.

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
