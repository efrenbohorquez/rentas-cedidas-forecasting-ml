import pandas as pd
import numpy as np
import os
import config
import utils

def load_data():
    """Carga los datos depurados."""
    # Usamos el dataset histórico completo generado en feature engineering
    return utils.load_data(config.FULL_HISTORIC_DATA_FILE)

def analyze_data_quality(df):
    """Genera estadísticas descriptivas de calidad de datos."""
    stats = {
        'Total Registros': len(df),
        'Rango Fechas': f"{df['fecha'].min().date()} a {df['fecha'].max().date()}",
        'Total Meses': df['fecha'].nunique(),
        'Monto Total Recaudado': df['recaudo'].sum(),
        'Promedio Mensual': df['recaudo'].mean(),
        'Mediana Mensual': df['recaudo'].median(),
        'Desviación Estándar': df['recaudo'].std(),
        'Registros Cero/Nulos': (df['recaudo'] <= 0).sum()
    }
    return stats

def aggregate_and_export(df, output_dir):
    """Agrega datos y exporta a CSV para modelado."""
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.set_index('fecha')
    
    # Mensual
    df_m = df.resample('M').sum()
    df_m.to_csv(output_dir / 'dataset_modelos_mensual.csv')
    
    # Bimestral
    df_b = df.resample('2M').sum()
    df_b.to_csv(output_dir / 'dataset_modelos_bimestral.csv')
    
    # Trimestral
    df_q = df.resample('Q').sum()
    df_q.to_csv(output_dir / 'dataset_modelos_trimestral.csv')
    
    return {'Mensual': len(df_m), 'Bimestral': len(df_b), 'Trimestral': len(df_q)}

def generate_report(stats, export_counts, diagnostic_path, output_path):
    """Genera el reporte final en Markdown."""
    
    # Cargar diagnóstico si existe
    diagnostic_table = "| Modelo | Horizonte | RMSE | MAPE |\n|---|---|---|---|\n| No disponible | - | - | - |"
    
    if os.path.exists(diagnostic_path):
        try:
            df_diag = pd.read_csv(diagnostic_path)
            # Formatear tabla markdown
            if not df_diag.empty:
                diagnostic_table = df_diag[['Horizonte', 'Model', 'MAPE', 'RMSE', 'AIC']].to_markdown(index=False, floatfmt=".4f")
        except Exception as e:
            diagnostic_table = f"Error cargando tabla diagnóstico: {e}"

    report_content = f"""# Informe Técnico: Calidad de Datos y Selección de Modelos
**Fecha de Generación:** {pd.Timestamp.now().date()}

## 1. Resumen de Limpieza de Datos
El dataset ha sido procesado siguiendo el protocolo estricto de limpieza:
- **Validación Temporal**: {stats['Rango Fechas']} ({stats['Total Meses']} meses).
- **Consistencia**: Se han neteado valores negativos y estandarizado fechas a 'Vigencia-Mes'.
- **Volumen**: {stats['Total Registros']} registros procesados.

### Estadísticas Descriptivas
- **Promedio Mensual**: ${stats['Promedio Mensual']:,.2f}
- **Mediana**: ${stats['Mediana Mensual']:,.2f}
- **Volatilidad (Std)**: ${stats['Desviación Estándar']:,.2f}

## 2. Preparación para Modelado (Datasets)
Se han generado archivos CSV listos para series de tiempo en `results/tables/`:
- **Mensual**: {export_counts['Mensual']} observaciones (`dataset_modelos_mensual.csv`)
- **Bimestral**: {export_counts['Bimestral']} observaciones (`dataset_modelos_bimestral.csv`)
- **Trimestral**: {export_counts['Trimestral']} observaciones (`dataset_modelos_trimestral.csv`)

## 3. Diagnóstico de Modelos (ARIMA vs SARIMA)
A continuación se presenta el resumen comparativo de los modelos evaluados en el módulo de diagnóstico avanzado:

{diagnostic_table}

### Conclusiones del Diagnóstico
1. **Horizonte Mensual**: Se prefiere **SARIMA** por su capacidad de capturar la estacionalidad anual.
2. **Horizonte Bimestral**: Los modelos **ARIMA** no estacionales muestran mayor robustez.
3. **Horizonte Trimestral**: La agregación trimestral pierde estructura; se sugiere usar horizontes más granulares.

## 4. Recomendación Final
Para la predicción de Rentas Cedidas en Cundinamarca, se recomienda utilizar el **Dataset Mensual** con un modelo **SARIMA**, complementado con un modelo **XGBoost** (como se vio en iteraciones anteriores) para capturar no-linealidades si se enriquece con variables exógenas.
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Reporte generado en: {output_path}")

def main():
    print("🚀 Generando Reporte Final (Refactorizado)...")
    
    output_dir_tables = config.RESULTS_DIR / 'tables'
    output_dir_reports = config.RESULTS_DIR / 'reports'
    
    output_dir_tables.mkdir(parents=True, exist_ok=True)
    output_dir_reports.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar Datos
    df = load_data()
    if df is None: return

    # 2. Analizar Calidad
    stats = analyze_data_quality(df)
    
    # 3. Exportar CSVs
    export_counts = aggregate_and_export(df, output_dir_tables)
    
    # 4. Generar Reporte
    # Asumimos que el path de diagnóstico sigue siendo relativo o deberia estar en config?
    # Por ahora config.RESULTS_DIR / ...
    diagnostic_path = config.RESULTS_DIR / 'figures/diagnostico_arima_sarima/reporte_comparativo.csv'
    report_path = output_dir_reports / 'informe_tecnico_datos_modelos.md'
    
    generate_report(stats, export_counts, diagnostic_path, report_path)
    print("✅ Generación de reportes finalizada.")

if __name__ == "__main__":
    main()
