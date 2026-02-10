# Informe Técnico: Calidad de Datos y Selección de Modelos
**Fecha de Generación:** 2026-02-09

## 1. Resumen de Limpieza de Datos
El dataset ha sido procesado siguiendo el protocolo estricto de limpieza:
- **Validación Temporal**: 2020-01-01 a 2025-10-01 (70 meses).
- **Consistencia**: Se han neteado valores negativos y estandarizado fechas a 'Vigencia-Mes'.
- **Volumen**: 70 registros procesados.

### Estadísticas Descriptivas
- **Promedio Mensual**: $53,128,182,371.06
- **Mediana**: $45,327,256,001.06
- **Volatilidad (Std)**: $37,117,818,748.46

## 2. Preparación para Modelado (Datasets)
Se han generado archivos CSV listos para series de tiempo en `results/tables/`:
- **Mensual**: 70 observaciones (`dataset_modelos_mensual.csv`)
- **Bimestral**: 36 observaciones (`dataset_modelos_bimestral.csv`)
- **Trimestral**: 24 observaciones (`dataset_modelos_trimestral.csv`)

## 3. Diagnóstico de Modelos (ARIMA vs SARIMA)
A continuación se presenta el resumen comparativo de los modelos evaluados en el módulo de diagnóstico avanzado:

| Horizonte   | Model   |   MAPE |              RMSE |       AIC |
|:------------|:--------|-------:|------------------:|----------:|
| Mensual     | ARIMA   | 1.5641 |  41167590125.3500 | 2988.3908 |
| Mensual     | SARIMA  | 1.4397 |  44859093615.2262 | 2299.0281 |
| Bimestral   | ARIMA   | 1.5776 |  75756188929.2985 | 1191.8257 |
| Bimestral   | SARIMA  | 1.9644 | 106213143713.3161 |  934.2824 |
| Trimestral  | ARIMA   | 1.1279 |  84198492974.4840 |  632.0696 |

### Conclusiones del Diagnóstico
1. **Horizonte Mensual**: Se prefiere **SARIMA** por su capacidad de capturar la estacionalidad anual.
2. **Horizonte Bimestral**: Los modelos **ARIMA** no estacionales muestran mayor robustez.
3. **Horizonte Trimestral**: La agregación trimestral pierde estructura; se sugiere usar horizontes más granulares.

## 4. Recomendación Final
Para la predicción de Rentas Cedidas en Cundinamarca, se recomienda utilizar el **Dataset Mensual** con un modelo **SARIMA**, complementado con un modelo **XGBoost** (como se vio en iteraciones anteriores) para capturar no-linealidades si se enriquece con variables exógenas.
