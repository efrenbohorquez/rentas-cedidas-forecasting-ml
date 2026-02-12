# Guía para la Redacción de Hallazgos (Tesis)

Esta guía estructura los **Hallazgos** de tu tesis basándose en los resultados técnicos obtenidos en el proyecto de "Proyección de Rentas Cedidas". Utiliza esta estructura para redactar tu capítulo de resultados y conclusiones.

---

## 1. Estructura Sugerida del Capítulo

Recomiendo dividir el capítulo de Hallazgos en cuatro secciones clave:

1.  **Caracterización de los Datos (El insumo)**
2.  **Desempeño Comparativo de Modelos (La competencia)**
3.  **Impacto del Horizonte Temporal (La validación)**
4.  **Implicaciones para la Toma de Decisiones (El valor agregado)**

---

## 2. Desarrollo de Hallazgos (Basado en tus Resultados)

### A. Caracterización de los Datos
*   **Hallazgo:** La serie temporal de Rentas Cedidas presenta una **estacionalidad semestral muy marcada**, con picos significativos en **Enero y Julio**.
*   **Evidencia:** El análisis exploratorio y las predicciones de 2025 muestran que el recaudo salta de promedios de ~22MM a picos de >110MM en estos meses.
*   **Interpretación:** Cualquier modelo que no capture explícitamente esta estacionalidad (como LSTM en nuestro caso) fallará en los momentos críticos de flujo de caja.

### B. Desempeño Comparativo de Modelos
*   **Hallazgo Principal:** **XGBoost fue el modelo superior**, superando a los métodos estadísticos clásicos (SARIMAX) y de Deep Learning (LSTM).
*   **Detalle por Modelo:**
    *   **XGBoost:** Logró capturar la magnitud de los picos de Enero y Julio (prediciendo ~134MM vs ~124MM reales en Ene-25). Su capacidad para manejar relaciones no lineales fue clave.
    *   **SARIMAX y Prophet:** Aunque identificaron la tendencia, tendieron a subestimar la volatilidad extrema. Además, requerían post-procesamiento para evitar predicciones negativas (hallazgo técnico importante: la necesidad de restricciones de dominio `ReLu`).
    *   **LSTM:** Mostró un comportamiento conservador, convergiendo hacia la media (~60MM constantes) y fallando en capturar la variabilidad estacional. *Esto es un hallazgo negativo valioso: la complejidad de una red neuronal no siempre garantiza mejores resultados en series cortas.*

### C. Análisis Multi-Horizonte (Mensual vs Trimestral vs Semestral)
*   **Hallazgo:** La precisión de los pronósticos mejora significativamente al agregar los datos a horizontes **Trimestrales y Semestrales**.
*   **Evidencia:** Mientras que el error mensual (MAPE) puede ser volátil debido a desfases de un mes en el recaudo, los tableros trimestrales muestran una tendencia mucho más suave y alineada con la realidad.
*   **Conclusión:** Para la planeación financiera estratégica, se recomienda utilizar el horizonte **Trimestral** como balance ideal entre detalle y precisión.

### D. Disponibilidad y Calidad de Datos
*   **Hallazgo Crítico:** La ventana de información histórica (2020-2025) es el límite inferior para entrenar modelos robustos.
*   **Observación:** Se identificó que características como los "rezagos" (lags) consumen el primer año de datos (2020), dejando efectivamente 2021-2024 para entrenamiento. Esto limita la capacidad del modelo para "aprender" patrones de largo plazo (ciclos económicos plurianuales), aunque es suficiente para la estacionalidad anual.

### E. Análisis Municipal y Concentración del Recaudo
*   **Hallazgo (Pareto):** Existe una **alta concentración del ingreso**, donde menos del 20% de los municipios generan el 80% del recaudo total (Ley de Pareto).
*   **Evidencia:** El análisis descriptivo (`pareto_top_20.png`) muestra que Bogotá, Medellín y Barranquilla son determinantes, mientras que la "cola larga" de municipios pequeños aporta marginalmente pero introduce ruido.
*   **Sincronización:** La matriz de correlación (`correlacion_top_municipios.png`) revela que los grandes centros urbanos están altamente sincronizados en sus ciclos tributarios, validando la hipótesis de patrones macroeconómicos compartidos.

### F. Análisis de Volatilidad y Riesgo (NUEVO)
*   **Hallazgo (Estrés de Liquidez):** El cálculo del **Value at Risk (VaR)** histórico al 95% indica caídas mensuales extremas (superiores al 80%), lo cual confirmó la naturaleza cíclica del recaudo.
*   **Interpretación Financiera:** Esto no es "ruido", sino un **riesgo estructural de liquidez**. Las entidades deben prever que, inmediatamente después de los picos de recaudo (Enero/Julio), enfrentarán "valles" donde los ingresos se contraen drásticamente.
*   **Cono de Incertidumbre:** El *Fan Chart* (`fan_chart_incertidumbre.png`) proyecta que la variabilidad se expande en el tiempo, sugiriendo que las proyecciones a más de 6 meses deben acompañarse de fondos de estabilización u otros instrumentos de cobertura.
*   **Distribución de Retornos:** La prueba de normalidad (`distribucion_retornos.png`) rechaza la hipótesis de una distribución normal, mostrando "colas pesadas". Esto implica que los eventos extremos (positivos o negativos) son más frecuentes de lo que prediría un modelo de riesgo estándar.

### G. Modelo Híbrido y Optimización (NUEVO - FASE 4)
*   **Superioridad del Ensemble:** La implementación de una estrategia de **Stacking** (combinando SARIMAX, Prophet y XGBoost mediante un meta-modelo de Ridge Regression) demostró una reducción en el error de pronóstico (MAPE) comparado con los modelos individuales en aislamiento.
*   **Sinergia de Modelos:**
    *   **SARIMAX:** Captura bien la estacionalidad lineal y la inercia autodependiente.
    *   **Prophet:** Aporta robustez ante cambios estructurales y días festivos (si aplica).
    *   **XGBoost:** Modela relaciones no lineales complejas que los modelos estadísticos tradicionales pierden.
    *   **Resultado:** El modelo híbrido aprovecha las fortalezas de cada uno, suavizando los errores individuales y generando una proyección más robusta para 2026.
*   **Optimización de Hiperparámetros:** La aplicación de *Grid Search* y validación cruzada (*Walk-Forward Validation*) aseguró que los parámetros seleccionados no estén sobreajustados al pasado, aumentando la confianza en la generalización del modelo.
*   **Gráfico Clave:** Ver `predictions/ensemble_forecast.png` para la comparativa visual de la señal híbrida vs datos reales.

### H. Validación de Horizonte Final (Agosto - Octubre 2025)

Como parte de la estrategia de inspección continua (Fase 6), se contrastó el desempeño "Out-of-Sample" del Modelo Híbrido (Ensemble Stacking) frente a una validación específica usando Regresión Ridge con características de Fourier.

**Tabla Comparativa de Resultados (Q3 2025):**

| Fecha | Real (COP) | Ensemble (Stacking) | Ridge (Validación) | Desviación Ensemble |
| :--- | :--- | :--- | :--- | :--- |
| **Ago-2025** | $30,419M | ~$22,044M | ~$23,587M | **Subestimación (-27%)** |
| **Sep-2025** | $22,557M | ~$24,118M | ~$26,473M | **Sobreestimación (+6.9%)** |
| **Oct-2025** | $22,564M | ~$20,443M | ~$23,984M | **Subestimación (-9.4%)** |

**Hallazgo Clave:**
El modelo **Ensemble** demuestra un comportamiento más conservador y estable, amortiguando la volatilidad extrema observada en los datos reales. Por otro lado, la validación con **Ridge** captura mejor los picos de recaudo (como en Agosto), pero tiende a sobreestimar los meses subsiguientes (Septiembre).

**Conclusión Técnica:** Se recomienda utilizar el **Ensemble** para la planificación presupuestal base debido a su prudencia, mientras que el modelo **Ridge** ajustado sirve como escenario de sensibilidad optimista.

---

## 3. Ejemplo de Redacción (Para Copiar/Adaptar)

> *"Como hallazgo principal de esta investigación, se determinó que el modelo basado en árboles de decisión (XGBoost) demostró una capacidad superior para modelar la dinámica no lineal del recaudo de rentas cedidas en comparación con los enfoques econométricos tradicionales (SARIMAX) y de redes neuronales (LSTM). Específicamente, XGBoost logró reducir el error porcentual medio absoluto (MAPE) en el horizonte mensual, capturando con mayor fidelidad los picos estacionales de enero y julio, críticos para la tesorería departamental.*
>
> *Adicionalmente, se evidenció que la desagregación temporal mensual presenta una alta volatilidad estocástica que dificulta la precisión punto a punto (>20% error). Sin embargo, al realizar la validación en horizontes agregados (trimestral y semestral), el error se estabiliza por debajo del 12%, validando la hipótesis de que estos modelos son herramientas robustas para la planeación fiscal de mediano plazo, mas no necesariamente para el control de caja diario."*

---

## 4. Gráficos de Soporte
Para sustentar estos hallazgos en el documento, asegúrate de incluir:
1.  **La gráfica de la serie temporal completa (2020-2025)** mostrando los picos repetitivos (Evidencia de Estacionalidad).
2.  **El gráfico de barras del Ranking de Modelos** (del Dashboard Mensual) donde XGBoost aparece con el menor error.
3.  **La comparación visual de predicciones** donde se vea la línea plana del LSTM contrastada con los picos acertados de XGBoost.
