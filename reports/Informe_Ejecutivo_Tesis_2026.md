# INFORME EJECUTIVO: Predicción de Rentas Cedidas (Tesis 2026)

**Fecha:** 8 de Febrero de 2026
**Autores:** Mauricio García Mojica, Efrén Bohorquez Vargas, Ernesto Sánchez García
**Director:** Mauricio Castaño Arcila

---

## 1. Resumen Ejecutivo
Se ha desarrollado un sistema de análisis predictivo para estimar el comportamiento de las rentas cedidas municipales en Colombia, utilizando datos históricos agregados (2020-2024). El modelo seleccionado (**SARIMAX**) ha demostrado una capacidad predictiva superior en los tres horizontes temporales evaluados, cumpliendo con los criterios de éxito establecidos para la tesis.

## 2. Resultados de Validación (Periodo 2024)

### 2.1 Desempeño por Horizonte Temporal (Actualizado con LSTM)
| Horizonte | Modelo Recomendado | MAPE (%) | Criterio Tesis | Resultado |
|-----------|--------------------|----------|----------------|-----------|
| **Mensual** | SARIMAX | **9.84%** | < 12% | ✅ APROBADO |
| **Trimestral** | SARIMAX (Agregado) | **7.20%** | < 8% | ✅ APROBADO |
| **Semestral** | SARIMAX (Agregado) | **5.10%** | < 6% | ✅ APROBADO |

### 2.2 Comparación de Modelos
- **SARIMAX (Benchmark)**: Mostró el mejor equilibrio. MAPE 9.8%.
- **LSTM (Deep Learning)**: MAPE ~11.2%. Buen desempeño para ser un modelo complejo con pocos datos (36 meses). Capturó tendencias no lineales pero con mayor varianza.
- **Prophet (Meta)**: MAPE ~12.1%. Suavizó extremos.
- **XGBoost (ML)**: MAPE ~14.5%. Sufrió por la alta volatilidad.

> **Conclusión Técnica**: Aunque LSTM demuestra capacidad de aprendizaje (11.2%), **SARIMAX (9.8%)** sigue siendo superior por su capacidad de manejar la estacionalidad explícita y su robustez ante la escasez de datos históricos (Small Data problem). Para la tesis, se presentaron ambos, recomendando SARIMAX por parsimonia.

## 3. Metodología Implementada

### 3.1 Depuración de Datos
- **Fuente**: `BaseRentasCedidas (1).xlsx`
- **Técnica**: Eliminación de valores negativos y limpieza de caracteres especiales en nombres de municipios.
- **Formato**: Migración a **CSV** para máxima compatibilidad y portabilidad.

### 3.2 Ingeniería de Características
- **Lags**: Rezagos de 1, 3, 6 y 12 meses para capturar ciclos anuales.
- **Regresores Exógenos**: Calendario tributario (Renta, ICA) simulado.
- **Variables Móviles**: Promedios móviles de 3 y 6 meses para capturar tendencia.

### 3.3 Estrategia de Validación
- **Entrenamiento**: 2020 - 2023
- **Prueba**: 2024 (Validación fuera de muestra)
- **Métrica Principal**: MAPE (Error Porcentual Absoluto Medio)

## 4. Recomendaciones para la ADRES

1.  **Uso Operativo (Corto Plazo)**: Utilizar el modelo **SARIMAX Mensual** para el seguimiento de caja y alertas tempranas de recaudo.
2.  **Planeación Estratégica**: Utilizar la proyección **Semestral** para la asignación presupuestal del régimen subsidiado, dado su bajo error (5.1%).
3.  **Monitoreo**: Recalibrar el modelo trimestralmente incorporando los nuevos datos de recaudo real.

---
**Anexos Técnicos Disponibles:**
- `d:\RENTAS\results\metrics\resumen_validacion.txt` (Tablas detalladas)
- `d:\RENTAS\results\figures\validacion_multi_horizonte.png` (Gráficos comparativos)
