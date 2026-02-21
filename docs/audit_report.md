# Auditoría de Código y Datos - Rentas Cedidas

## Resumen Ejecutivo
El código actual en `01_limpieza_inicial.py` proporciona una base funcional para la limpieza de datos, pero presenta **riesgos críticos** para el objetivo de "Predicción a Nivel Municipal" y el manejo contable de las Rentas Cedidas.

## Hallazgos Críticos

### 1. Granularidad de los Datos (Nivel Municipal vs Departamental)
- **Estado**: **Confirmado por Usuario**.
- **Hallazgo**: La columna `NombreBeneficiarioAportante` contiene información mixta (Municipios y Departamentos).
- **Acción Requerida**: Implementar lógica de filtrado para segregar registros municipales de los departamentales antes del entrenamiento de modelos. Se debe verificar si existen identificadores únicos (NIT) para automatizar esta separación y evitar ambigüedades por nombres similares.

### 2. Manejo de Valores Negativos y Ceros
- **Código Actual**: 
  ```python
  df = df[df['recaudo'] > 0]
  ```
  Elimina filas con valor 0 o negativo.
- **Riesgo**: 
  - **Negativos**: En contabilidad pública, los negativos suelen ser correcciones o devoluciones. Eliminarlos infla el recaudo histórico real.
  - **Ceros**: Para series de tiempo (SARIMAX, Prophet), la *ausencia* de datos (días sin recaudo) es información valiosa. Al eliminar las filas, se crean "saltos" temporales que pueden confundir a los modelos si no se rellenan posteriormente.

### 3. Definición Temporal (Caja vs Causación)
- **Código Actual**: Usa `FechaRecaudo` (fecha exacta de la transacción bancaria).
- **Contexto**: Para "Financiamiento del Régimen Subsidiado", la presupuestación suele basarse en la `Vigencia` y `Mes` de la obligación.
- **Recomendación**: Evaluar si el modelo debe predecir el flujo de caja diario (`FechaRecaudo`) o la disponibilidad presupuestal mensual (`Vigencia` + `Mes`). La volatilidad diaria puede ser ruido irrelevante para la planeación financiera estratégica.

## Recomendaciones Técnicas

### 4. Calidad de Código y Estructura
- **Rutas "Hardcoded"**: La ruta `BaseRentasVF_limpieza21feb.xlsx` está fija.
  - *Sugerencia*: Usar un archivo de configuración o argumentos de línea de comando.
- **Manejo de Errores**: El bloque `try-except` captura todas las excepciones de forma genérica.
- **Validación de Columnas**: El script verifica columnas pero no valida tipos de datos esperados más allá de la conversión forzada (`errors='coerce'`), lo que podría silenciar errores de formato masivos.

## Plan de Acción Sugerido

1.  **Validar Fuente de Datos**: Confirmar si existe un campo "Municipio" en el dataset completo o si se requiere una fuente adicional.
2.  **Ajustar Limpieza**: 
    - No eliminar negativos automáticamente; agregarlos al periodo correspondiente (neteo).
    - Preservar ceros si representan "días sin recaudo" para modelos diarios, o agregar por mes antes de entrenar.
3.  **Refinar Variable Temporal**: Crear una columna de fecha basada en `Vigencia-Mes` para análisis de largo plazo.
