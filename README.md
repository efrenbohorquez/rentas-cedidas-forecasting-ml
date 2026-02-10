# Proyección de Rentas Cedidas con Machine Learning

Este repositorio contiene el código fuente y la documentación técnica del proyecto de tesis: **"Modelo Predictivo para la Proyección y Planeación Financiera de Rentas Cedidas Departamentales"**.

## 📋 Descripción
El proyecto implementa un flujo de trabajo de Ciencia de Datos (CRISP-DM) para predecir el recaudo mensual de rentas cedidas, permitiendo a la entidad territorial mejorar su planeación financiera y reducir la incertidumbre del flujo de caja.

## 🛠️ Tecnologías Usadas
*   **Lenguaje:** Python 3.10+
*   **Librerías:** Pandas, NumPy, Scikit-learn, Statsmodels.
*   **Modelos:**
    *   `SARIMAX` (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)
    *   `Prophet` (Facebook's additive model)
    *   `XGBoost` (Extreme Gradient Boosting)
    *   `LSTM` (Long Short-Term Memory Networks - PyTorch)

## 📂 Estructura del Proyecto
```
rentas-cedidas-forecasting-ml/
├── data/                   # (Ignorado) Datos crudos y procesados
├── models/                 # (Ignorado) Modelos entrenados (.pkl, .pth)
├── notebooks/              # Exploración y prototipado
├── results/                # (Ignorado) Predicciones y figuras resultantes
├── scripts/                # Código fuente productivo
│   ├── 01_limpieza_inicial.py      # ETL y limpieza
│   ├── 02_feature_engineering.py   # Ingeniería de características
│   ├── 03_modelos.py               # Entrenamiento y predicción
│   ├── 04_validacion_final.py      # Evaluación multi-horizonte
│   ├── 05_visualizaciones_tesis.py # Gráficos para el documento
│   └── 06_dashboards_horizonte.py  # Generación de tableros de control
└── README.md               # Este archivo
```

## 🚀 Cómo Ejecutar
1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Ejecutar Pipeline:**
    ```bash
    python scripts/01_limpieza_inicial.py
    python scripts/02_feature_engineering.py
    python scripts/03_modelos.py
    python scripts/06_dashboards_horizonte.py
    ```

## 📊 Resultados Destacados
*   **Mejor Modelo:** XGBoost (Gradient Boosting).
*   **Precisión:** MAPE < 12% en horizontes trimestrales.
*   **Hallazgo:** Se identificó una fuerte estacionalidad semestral (Enero/Julio) mejor capturada por modelos no lineales.

---
**Nota sobre Privacidad:** Los datos originales y procesados no se incluyen en este repositorio para proteger la confidencialidad de la información tributaria.
