# Guía Metodológica (Tesis)

Esta guía estructura el capítulo de **Metodología** de tu tesis alineándolo con el estándar de la industria para minería de datos: **CRISP-DM** (Cross-Industry Standard Process for Data Mining). Esto le dará rigor académico y profesional a tu documento.

---

## 1. Enfoque Metodológico (CRISP-DM)

La investigación siguió un enfoque cuantitativo y experimental, estructurado en seis fases iterativas:

### Fase 1: Comprensión del Negocio (Business Understanding)
*   **Objetivo:** Desarrollar un modelo predictivo para el recaudo de *Rentas Cedidas* que sirva como herramienta de soporte para la planeación financiera departamental.
*   **Problema:** La incertidumbre en el flujo de caja debido a la alta estacionalidad y variabilidad del recaudo.
*   **Criterio de Éxito:** Lograr un Error Porcentual Absoluto Medio (MAPE) inferior al **12%** en horizontes de planeación (Trimestral/Semestral).

### Fase 2: Comprensión de los Datos (Data Understanding)
*   **Fuente de Información:** Base de datos histórica oficial (`BaseRentasVF_limpieza21feb.xlsx`) suministrada por la entidad, abarcando las vigencias **2020 a 2025**.
*   **Variables Clave:**
    *   `Vigencia` (Año fiscal).
    *   `MesCalendario` (Temporalidad mensual).
    *   `ValorRecaudo` (Variable objetivo a predecir).
    *   `NombreBeneficiario` (Para filtrado municipal/departamental).
*   **Análisis Exploratorio (EDA):** Se identificaron patrones estacionales semestrales (picos en Enero/Julio) y la necesidad de limpieza de valores atípicos negativos.

### Fase 3: Preparación de los Datos (Data Preparation)
Esta fase fue crítica y se implementó mediante scripts de Python (`01_limpieza_inicial.py` y `02_feature_engineering.py`):
1.  **Limpieza y Neteo:** Se consolidaron los registros diarios en totales mensuales, neteando anulaciones y devoluciones para obtener el flujo de caja real.
2.  **Filtrado Geográfico:** Se segregaron las rentas de nivel *Departamental* de las *Municipales* para enfocar el análisis en el ámbito correcto.
3.  **Ingeniería de Características:**
    *   **Rezagos (Lags):** Se crearon variables con el valor del recaudo de hace 1, 6 y 12 meses para capturar la inercia y estacionalidad.
    *   **Promedios Móviles:** Se calcularon tendencias suaves de 3 y 6 meses.
    *   **Transformación Temporal:** Se convirtió la variable fecha a `datetime` y se extrajeron características cíclicas (mes del año).

### Fase 4: Modelado (Modeling)
Se entrenaron y compararon cuatro familias de algoritmos (`03_modelos.py`) para capturar diferentes aspectos de la serie temporal:
1.  **SARIMAX (Estadístico):** Como línea base (benchmark) para capturar la autocorrelación lineal y estacionalidad.
2.  **Prophet (Aditivo):** Modelo de Facebook diseñado para series con fuertes componentes estacionales y días festivos (ajustado con feriados de Colombia).
3.  **XGBoost (Machine Learning):** Algoritmo de *Gradient Boosting* basado en árboles de decisión, capaz de capturar relaciones no lineales complejas entre las variables de rezago.
4.  **LSTM (Deep Learning):** Red neuronal recurrente (Long Short-Term Memory) para capturar dependencias de largo plazo secuenciales.
    *   *Nota Técnica:* Se aplicó una restricción de no-negatividad (`ReLu`) a las salidas de todos los modelos para garantizar coherencia fiscal.

### Fase 4.1: Optimización y Modelo Híbrido (Ensemble)
Para superar las limitaciones de los modelos individuales, se desarrolló una estrategia de **Stacking** (`11_modelos_ensemble.py`):
1.  **Optimización de Hiperparámetros:** Se utilizó *Grid Search* para afinar los parámetros de SARIMAX (p,d,q) y XGBoost (n_estimators, max_depth).
2.  **Meta-Modelo:** Se entrenó un modelo de **Ridge Regression** (Nivel 1) que toma como entrada las predicciones de los modelos base (Nivel 0) y aprende a combinarlas óptimamente para minimizar el error global.
3.  **Resultado:** Un modelo híbrido que pondera dinámicamente la estacionalidad (SARIMAX) y la complejidad no lineal (XGBoost).

### Fase 5: Evaluación (Evaluation)
La validación (`04_validacion_final.py`) se realizó mediante una estrategia de ventana deslizante (*Walk-Forward Validation*) simulando un escenario real de predicción:
*   **Conjunto de Entrenamiento:** Datos de 2020 a 2024.
*   **Conjunto de Prueba:** Datos reales de 2025 (no vistos por el modelo).
*   **Métrica Principal:** MAPE (Mean Absolute Percentage Error) por su interpretabilidad directa en términos porcentuales de desvío presupuestal.
*   **Análisis Multi-Horizonte:** Se evaluó el desempeño en tres niveles de agregación: Mensual, Trimestral y Semestral.

### Fase 6: Despliegue (Deployment)
Como producto final, se desarrollaron **Tableros de Control** (`06_dashboards_horizonte.py`) que visualizan:
1.  La serie histórica completa.
2.  La proyección futura con su intervalo de confianza.
3.  El semáforo de precisión del modelo (Verde/Rojo según el umbral del 12%).

---

## 2. Herramientas Utilizadas (Stack Tecnológico)

*   **Lenguaje:** Python 3.10+
*   **Librerías Principales:**
    *   `pandas` & `numpy`: Manipulación matricial de datos.
    *   `statsmodels`: Modelos econométricos (SARIMAX y pruebas de estacionariedad).
    *   `xgboost`, `prophet`, `torch` (PyTorch): Motores de predicción avanzada.
    *   `matplotlib` & `seaborn`: Visualización de datos.

---

Use esta estructura para narrar "la historia" de cómo transformó los datos crudos en conocimiento útil, justificando cada decisión técnica (como por qué usar XGBoost o por qué evaluar trimestralmente) con argumentos de negocio (mayor precisión para el presupuesto).
