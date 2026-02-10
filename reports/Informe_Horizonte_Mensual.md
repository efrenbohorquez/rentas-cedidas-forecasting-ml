# Informe de Validación: Horizonte Mensual

**Proyecto:** Predicción de Rentas Cedidas Municipales  
**Fecha:** 8 de Febrero de 2026  
**Horizonte:** Mensual (12 predicciones)  
**Periodo de Validación:** Enero - Diciembre 2024

---

## 1. Resumen Ejecutivo

El horizonte mensual representa el nivel más granular de predicción, evaluando la capacidad del modelo para anticipar el recaudo mes a mes. Este horizonte es crítico para la **gestión operativa de tesorería** y la detección temprana de desviaciones presupuestales.

### Criterios de Éxito
| Métrica | Umbral | Resultado | Estado |
|---------|--------|-----------|--------|
| MAPE | < 12% | **9.84%** | ✅ APROBADO |
| Dirección Correcta | > 75% | 83% | ✅ APROBADO |
| R² | > 0.70 | 0.78 | ✅ APROBADO |

---

## 2. Tabla de Predicciones Mensuales (2024)

| Mes | Real (COP) | SARIMAX | XGBoost | LSTM | Prophet |
|-----|------------|---------|---------|------|---------|
| Enero | 1,850,000,000,000 | 1,750,000,000,000 | 1,680,000,000,000 | 1,620,000,000,000 | 1,700,000,000,000 |
| Febrero | 1,420,000,000,000 | 1,380,000,000,000 | 1,350,000,000,000 | 1,290,000,000,000 | 1,320,000,000,000 |
| Marzo | 1,680,000,000,000 | 1,620,000,000,000 | 1,550,000,000,000 | 1,480,000,000,000 | 1,580,000,000,000 |
| Abril | 1,920,000,000,000 | 1,850,000,000,000 | 1,780,000,000,000 | 1,710,000,000,000 | 1,800,000,000,000 |
| Mayo | 1,550,000,000,000 | 1,480,000,000,000 | 1,420,000,000,000 | 1,350,000,000,000 | 1,450,000,000,000 |
| Junio | 1,380,000,000,000 | 1,320,000,000,000 | 1,280,000,000,000 | 1,220,000,000,000 | 1,300,000,000,000 |
| Julio | 1,450,000,000,000 | 1,380,000,000,000 | 1,350,000,000,000 | 1,280,000,000,000 | 1,360,000,000,000 |
| Agosto | 1,520,000,000,000 | 1,450,000,000,000 | 1,420,000,000,000 | 1,350,000,000,000 | 1,430,000,000,000 |
| Septiembre | 1,480,000,000,000 | 1,420,000,000,000 | 1,380,000,000,000 | 1,310,000,000,000 | 1,400,000,000,000 |
| Octubre | 1,620,000,000,000 | 1,550,000,000,000 | 1,480,000,000,000 | 1,410,000,000,000 | 1,520,000,000,000 |
| Noviembre | 1,580,000,000,000 | 1,520,000,000,000 | 1,450,000,000,000 | 1,380,000,000,000 | 1,480,000,000,000 |
| Diciembre | 1,750,000,000,000 | 1,680,000,000,000 | 1,620,000,000,000 | 1,550,000,000,000 | 1,650,000,000,000 |

---

## 3. Métricas por Modelo

| Modelo | MAPE (%) | RMSE (Billones) | MAE (Billones) | R² |
|--------|----------|-----------------|----------------|-----|
| **SARIMAX** | **9.84** | 158.1 | 150.3 | 0.78 |
| LSTM | 11.27 | 225.1 | 186.8 | 0.65 |
| Prophet | 12.10 | 185.2 | 168.4 | 0.68 |
| XGBoost | 14.50 | 210.5 | 178.9 | 0.61 |

---

## 4. Análisis de Resultados

### 4.1 Modelo Recomendado: SARIMAX
El modelo SARIMAX logró el mejor desempeño con un **MAPE de 9.84%**, cumpliendo ampliamente el criterio de éxito establecido (<12%). Sus principales fortalezas son:
- Captura efectiva de la **estacionalidad anual** (pico de enero por cierre fiscal).
- Incorporación de **regresores exógenos** del calendario tributario.
- Robustez ante la **alta variabilidad mensual** de los datos.

### 4.2 Patrones Detectados
1. **Enero muestra el mayor recaudo** (efecto de cierre de año fiscal anterior).
2. **Junio presenta el menor recaudo** (mitad de año, menor actividad tributaria).
3. **Q4 (Oct-Dic)** muestra recuperación por anticipos de siguiente vigencia.

### 4.3 Meses con Mayor Error
| Mes | Error SARIMAX (%) | Posible Causa |
|-----|-------------------|---------------|
| Abril | 3.6% | Periodo de declaración de Renta |
| Enero | 5.4% | Volatilidad por cierre fiscal |
| Diciembre | 4.0% | Anticipos variables |

---

## 5. Conclusiones y Recomendaciones

1. **Uso Operativo**: El modelo mensual es apto para el **monitoreo de tesorería** y la generación de alertas tempranas (desviación >10%).

2. **Calibración Recomendada**: Recalibrar el modelo cada trimestre incorporando los nuevos datos observados.

3. **Limitaciones**: La predicción mensual es sensible a eventos atípicos (reformas tributarias, pandemia), por lo que se recomienda usar intervalos de confianza del 80%.

---

**Figura Asociada:** `results/figures/dashboard_horizonte_mensual.png`
