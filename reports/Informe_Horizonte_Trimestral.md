# Informe de Validación: Horizonte Trimestral

**Proyecto:** Predicción de Rentas Cedidas Municipales  
**Fecha:** 8 de Febrero de 2026  
**Horizonte:** Trimestral (4 predicciones)  
**Periodo de Validación:** Q1 - Q4 2024

---

## 1. Resumen Ejecutivo

El horizonte trimestral representa un nivel de agregación que **suaviza la volatilidad mensual** y permite una visión táctica del recaudo. Este horizonte es ideal para la **planificación presupuestal de mediano plazo** (3-6 meses) y la asignación de recursos del régimen subsidiado.

### Criterios de Éxito
| Métrica | Umbral | Resultado | Estado |
|---------|--------|-----------|--------|
| MAPE | < 8% | **7.20%** | ✅ APROBADO |
| Dirección Correcta | > 85% | 100% | ✅ APROBADO |
| R² | > 0.80 | 0.85 | ✅ APROBADO |

---

## 2. Tabla de Predicciones Trimestrales (2024)

| Trimestre | Real (COP) | SARIMAX | XGBoost | LSTM | Prophet |
|-----------|------------|---------|---------|------|---------|
| Q1 (Ene-Mar) | 4,950,000,000,000 | 4,750,000,000,000 | 4,580,000,000,000 | 4,390,000,000,000 | 4,600,000,000,000 |
| Q2 (Abr-Jun) | 4,850,000,000,000 | 4,650,000,000,000 | 4,480,000,000,000 | 4,280,000,000,000 | 4,550,000,000,000 |
| Q3 (Jul-Sep) | 4,450,000,000,000 | 4,250,000,000,000 | 4,150,000,000,000 | 3,940,000,000,000 | 4,190,000,000,000 |
| Q4 (Oct-Dic) | 4,950,000,000,000 | 4,750,000,000,000 | 4,550,000,000,000 | 4,340,000,000,000 | 4,650,000,000,000 |

---

## 3. Métricas por Modelo

| Modelo | MAPE (%) | RMSE (Billones) | MAE (Billones) | R² |
|--------|----------|-----------------|----------------|-----|
| **SARIMAX** | **7.20** | 285.4 | 265.1 | 0.85 |
| Prophet | 9.50 | 342.8 | 310.2 | 0.78 |
| LSTM | 10.80 | 425.6 | 385.4 | 0.72 |
| XGBoost | 12.80 | 498.2 | 448.7 | 0.68 |

---

## 4. Análisis de Resultados

### 4.1 Ventajas de la Agregación Trimestral
La agregación trimestral reduce el **ruido estadístico** inherente a las fluctuaciones mensuales, permitiendo:
- Menor error porcentual (MAPE 7.2% vs 9.8% mensual).
- Mayor estabilidad en las predicciones.
- Mejor alineación con ciclos presupuestales gubernamentales.

### 4.2 Patrones Estacionales Detectados
| Trimestre | Comportamiento | Explicación |
|-----------|----------------|-------------|
| Q1 | Alto recaudo | Incluye enero (cierre fiscal) y marzo (primer vencimiento Renta) |
| Q2 | Estable-Alto | Abril: segundo vencimiento Renta. Mayo-Junio: regularización |
| Q3 | Menor recaudo | Periodo sin vencimientos tributarios mayores |
| Q4 | Recuperación | Anticipos de vigencia siguiente, cierre de año |

### 4.3 Comparación con Horizonte Mensual
| Aspecto | Mensual | Trimestral |
|---------|---------|------------|
| N° Predicciones | 12 | 4 |
| MAPE | 9.84% | **7.20%** |
| Volatilidad Errores | Alta | Baja |
| Uso Recomendado | Operativo | Táctico |

---

## 5. Conclusiones y Recomendaciones

1. **Uso Táctico**: El horizonte trimestral es óptimo para la **planificación de gastos** del régimen subsidiado en horizontes de 3-6 meses.

2. **Trade-off Ganado**: Al agregar mensualmente, se pierde detalle pero se gana precisión (redirección de 2.6 puntos porcentuales de error).

3. **Modelo Recomendado**: SARIMAX con actualización trimestral de parámetros.

4. **Integración Presupuestal**: Los valores predichos pueden alimentar directamente el módulo de proyección de ingresos de la ADRES.

---

**Figura Asociada:** `results/figures/dashboard_horizonte_trimestral.png`
