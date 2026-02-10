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
