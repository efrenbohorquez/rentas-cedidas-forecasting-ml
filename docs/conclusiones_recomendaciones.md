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

---

## 2. Recomendaciones (La Acción)

Las recomendaciones deben ser aplicables por la entidad gestora de las rentas.

### A. Para la Entidad / Gestión Financiera
1.  **Adopción del Horizonte Trimestral:** Se recomienda utilizar las proyecciones trimestrales agregadas para la elaboración del Presupuesto Anual y el Marco Fiscal de Mediano Plazo, ya que ofrecen la mayor estabilidad y precisión.
2.  **Monitoreo de Picos Estacionales:** Utilizar las alertas de estacionalidad detectadas (Enero/Julio) para prever necesidades de liquidez o excedentes de tesorería, evitando déficits transitorios de caja.
3.  **Validación Continua:** Institucionalizar la comparación anual entre lo proyectado y lo ejecutado para recalibrar las expectativas de ingreso.

### B. Para el Equipo Técnico / Futuros Desarrollos
1.  **Reentrenamiento Anual del Modelo:** Dado que la dinámica tributaria cambia (reformas, amnistías, ciclo económico), se recomienda reentrenar el modelo XGBoost una vez al año incorporando la nueva vigencia completa ("Vigencia Cerrada").
2.  **Incorporación de Variables Exógenas:** Para mejorar la precisión mensual, se sugiere explorar la inclusión de variables macroeconómicas líderes (PIB departamental, inflación, calendario tributario oficial) que puedan explicar las desviaciones atípicas.
3.  **Restricción de No-Negatividad:** Implementar reglas de negocio estrictas (Post-processing) en cualquier implementación futura para garantizar que ninguna proyección arroje valores negativos, preservando la credibilidad del sistema ante los tomadores de decisiones.

---

## 3. Ejemplo de Cierre (Para Copiar/Adaptar)

> *"En conclusión, esta investigación valida el uso de algoritmos de Gradient Boosting (XGBoost) como la herramienta más eficiente para la proyección de Rentas Cedidas en el departamento. Si bien la complejidad estocástica del recaudo mensual limita la precisión a corto plazo, la agregación trimestral ofrece un instrumento robusto para la planeación fiscal. Se recomienda a la Secretaría de Hacienda la implementación de este modelo como insumo técnico para el anteproyecto de presupuesto, complementado con un esquema de reentrenamiento anual que garantice su vigencia frente a la cambiante realidad económica."*
