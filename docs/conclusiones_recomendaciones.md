# Guía para Conclusiones y Recomendaciones (Tesis)

Esta guía complementa el capítulo de **Hallazgos**, sintetizando los resultados en conclusiones definitivas y recomendaciones prácticas para la gestión de las Rentas Cedidas.

---

## 1. Conclusiones (La Síntesis)

Las conclusiones deben responder directamente a tus objetivos específicos. Basado en los resultados obtenidos:

### A. Sobre la Factibilidad de Predicción
*   **Conclusión:** Se demostró que es factible modelar y predecir el comportamiento de las Rentas Cedidas con un error inferior al **12%** en horizontes agregados (trimestral/semestral), superando la incertidumbre de la proyección puramente inercial o basada en promedios históricos simples.
*   **Evidencia:** Los modelos alcanzaron un MAPE del **~7-10%** en validación cruzada para horizontes trimestrales.

### B. Sobre la Selección del Modelo Óptimo
*   **Conclusión:** Las técnicas de **Aprendizaje Automático (XGBoost)** resultaron superiores a los modelos econométricos tradicionales (SARIMAX) y de aprendizaje profundo (LSTM) para este tipo de datos fiscales.
*   **Razón:** XGBoost logró capturar la no-linealidad y los picos estacionales extremos (Enero/Julio) sin requerir supuestos rígidos de estacionariedad o normalidad en los residuos, lo cual es crítico dado el carácter volátil del recaudo.

### C. Sobre la Granularidad Temporal
*   **Conclusión:** La predicción mensual ("punto a punto") posee una varianza natural irreductible debido a factores operativos (fechas de pago, festivos, retrasos administrativos). Por tanto, la unidad mínima confiable para la planeación financiera estratégica es el **Trimestre**. Intentar gestionar la caja mes a mes con estos modelos conlleva un riesgo de error >20% en meses atípicos.

### D. Sobre el Riesgo de Liquidez (NUEVO)
*   **Conclusión:** Se identificó un riesgo estructural de liquidez post-estacional. El análisis de **Value at Risk (VaR)** indica que, con un 95% de confianza, los ingresos pueden contraerse drásticamente (hasta un 80%) en los meses valle (Febrero/Agosto), lo cual no es una anomalía sino un patrón cíclico predecible que debe ser gestionado.

### E. Sobre el Modelo Híbrido (Ensemble)
*   **Conclusión:** La estrategia de **Stacking** (combinando SARIMAX, Prophet y XGBoost) demostró ser la más robusta, reduciendo el error de pronóstico al compensar las debilidades individuales de cada modelo.
*   **Valor:** Este enfoque "Híbrido" ofrece una proyección más equilibrada, capturando tanto la estacionalidad rígida (SARIMAX) como las relaciones no lineales complejas (XGBoost), lo que lo convierte en la herramienta preferida para la proyección final 2026.

---

## 2. Recomendaciones (La Acción)

Las recomendaciones deben ser aplicables por la entidad gestora de las rentas.

### A. Para la Entidad / Gestión Financiera
1.  **Adopción del Horizonte Trimestral:** Se recomienda utilizar las proyecciones trimestrales agregadas para la elaboración del Presupuesto Anual y el Marco Fiscal de Mediano Plazo, ya que ofrecen la mayor estabilidad y precisión.
2.  **Monitoreo de Picos Estacionales:** Utilizar las alertas de estacionalidad detectadas (Enero/Julio) para prever necesidades de liquidez o excedentes de tesorería, evitando déficits transitorios de caja.
3.  **Validación Continua:** Institucionalizar la comparación anual entre lo proyectado y lo ejecutado para recalibrar las expectativas de ingreso.
4.  **Cobertura de Riesgos:** Establecer fondos de estabilización o líneas de crédito revolvente que se activen automáticamente en los meses identificados por el modelo de **VaR** como de "alto riesgo de iliquidez", mitigando el impacto de los valles recaudatorios.
5.  **Adopción del Modelo Híbrido:** Utilizar las proyecciones del **Ensemble (Stacking)** como la cifra oficial de referencia, manteniendo los modelos individuales (SARIMAX/XGBoost) solo como "testigos" o bandas de control.

### B. Para el Equipo Técnico / Futuros Desarrollos
1.  **Reentrenamiento Anual del Modelo:** Dado que la dinámica tributaria cambia (reformas, amnistías, ciclo económico), se recomienda reentrenar el modelo XGBoost una vez al año incorporando la nueva vigencia completa ("Vigencia Cerrada").
2.  **Incorporación de Variables Exógenas:** Para mejorar la precisión mensual, se sugiere explorar la inclusión de variables macroeconómicas líderes (PIB departamental, inflación, calendario tributario oficial) que puedan explicar las desviaciones atípicas.
3.  **Restricción de No-Negatividad:** Implementar reglas de negocio estrictas (Post-processing) en cualquier implementación futura para garantizar que ninguna proyección arroje valores negativos, preservando la credibilidad del sistema ante los tomadores de decisiones.

### C. Enfoque Diferencial Municipal
**Recomendación:** Aplicar la segmentación de modelos.
*   **Acción:** No utilizar un modelo único nacional para todos los municipios.
*   **Justificación:** El análisis de clústeres sugiere que los Top 10 municipios (Bogotá, Medellín, etc.) tienen dinámicas predecibles con ARIMA/XGBoost, mientras que los municipios pequeños requieren modelos más simples (suavizamiento exponencial) o agregación departamental debido a la escasez de datos.

---

## 3. Ejemplo de Cierre (Para Copiar/Adaptar)

> *"En conclusión, esta investigación valida el uso de algoritmos de Gradient Boosting (XGBoost) como la herramienta más eficiente para la proyección de Rentas Cedidas en el departamento. Si bien la complejidad estocástica del recaudo mensual limita la precisión a corto plazo, la agregación trimestral ofrece un instrumento robusto para la planeación fiscal. Se recomienda a la Secretaría de Hacienda la implementación de este modelo como insumo técnico para el anteproyecto de presupuesto, complementado con un esquema de reentrenamiento anual que garantice su vigencia frente a la cambiante realidad económica."*
