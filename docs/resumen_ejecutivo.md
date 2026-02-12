# Resumen Ejecutivo (Abstract)

Este documento sirve como base para el **Resumen** o **Abstract** de tu tesis. Resume la problemática, la metodología aplicada, los resultados obtenidos y las conclusiones principales en un formato conciso.

---

## Título Sugerido
**"Modelo Predictivo para la Proyección y Planeación Financiera de Rentas Cedidas Departamentales mediante Técnicas de Aprendizaje Automático"**

---

## Resumen

**Contexto y Problema:**
La gestión financiera de las entidades territoriales enfrenta desafíos significativos debido a la incertidumbre y volatilidad en el recaudo de ingresos no tributarios, específicamente las **Rentas Cedidas**. La ausencia de herramientas técnicas de proyección genera deficiencias en la planeación presupuestal, provocando descalces de tesorería y una ejecución ineficiente de los recursos públicos. Este proyecto aborda la necesidad de transitar de modelos inerciales empíricos hacia métodos cuantitativos robustos.

**Metodología:**
Se implementó una metodología experimental basada en el estándar **CRISP-DM**, utilizando una base de datos histórica de recaudos (2020-2025). El proceso incluyó una fase exhaustiva de limpieza y neteo de datos para depurar distorsiones administrativas, seguida de ingeniería de características para capturar la estacionalidad. Se evaluaron y compararon cuatro enfoques de modelado: **SARIMAX** (estadístico), **Prophet** (aditivo), **XGBoost** (Machine Learning) y **LSTM** (Deep Learning), validados mediante una estrategia de ventana deslizante en horizontes mensuales, trimestrales y semestrales.

**Resultados:**
El análisis comparativo identificó que la estrategia de **Ensemble (Stacking)**, que combina SARIMAX, Prophet y XGBoost, superó a todos los modelos individuales, logrando la mayor precisión y estabilidad en las proyecciones. XGBoost, como componente individual, demostró una capacidad superior para capturar los picos estacionales críticos de enero y julio, mientras que el enfoque híbrido compensó sus debilidades en periodos de baja volatilidad. Por el contrario, los modelos de redes neuronales (LSTM) mostraron limitaciones para generalizar patrones en series temporales cortas.

**Conclusiones:**
La investigación concluye que la adopción de modelos basados en aprendizaje automático ("Machine Learning") mejora significativamente la precisión de las proyecciones fiscales departamentales frente a métodos convencionales. Se recomienda institucionalizar el uso de pronósticos **trimestrales** para la elaboración del Marco Fiscal de Mediano Plazo, permitiendo a la administración pública anticipar escenarios de liquidez y optimizar la asignación del gasto social bajo criterios técnicos y objetivos.

**Palabras Clave:**
Proyección Financiera, Rentas Cedidas, Machine Learning, XGBoost, Planeación Presupuestal, Series de Tiempo, Sector Público.

---

## Abstract (Versión en Inglés para Referencia)

**Context:** Financial management in territorial entities faces uncertainty due to volatility in non-tax revenue collection (Ceded Revenues).
**Methodology:** An experimental study based on CRISP-DM was conducted using historical data (2020-2025). Four modeling approaches (SARIMAX, Prophet, XGBoost, LSTM) were evaluated using sliding window validation.
**Results:** **XGBoost** outperformed other models, achieving a MAPE below 12% in aggregated horizons and correctly identifying seasonal peaks.
**Conclusions:** Machine Learning models significantly improve fiscal forecasting accuracy compared to traditional methods. It is recommended to adopt quarterly forecasts for medium-term fiscal planning to optimize public spending allocation.

**Keywords:** Financial Forecasting, Ceded Revenues, Machine Learning, XGBoost, Budget Planning, Time Series, Public Sector.
