import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import os
import warnings

warnings.filterwarnings("ignore")

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Carga y prepara los datos históricos."""
    df = pd.read_parquet('data/features/dataset_historico_completo.parquet')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.set_index('fecha')
    
    # Asegurar que solo usamos datos hasta 2025 (o lo que haya disponible)
    # y filtrar datos negativos o ceros si es necesario para logaritmos
    df = df[df['recaudo'] > 0]
    
    return df['recaudo']

def aggregate_data(series, freq):
    """Agrega la serie de tiempo a la frecuencia deseada."""
    if freq == 'M':
        return series.resample('M').sum()
    elif freq == '2M': # Bimestral
        return series.resample('2M').sum()
    elif freq == 'Q': # Trimestral
        return series.resample('Q').sum()
    return series

def evaluate_model(y_true, y_pred, model_name):
    """Calcula métricas de evaluación."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'Model': model_name, 'RMSE': rmse, 'MAPE': mape}

def plot_diagnostics(residuals, title, save_path):
    """Genera gráficos de diagnóstico para los residuos."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    # 1. Residuos en el tiempo
    axes[0, 0].plot(residuals)
    axes[0, 0].set_title('Residuos en el Tiempo')
    axes[0, 0].axhline(0, color='red', linestyle='--')

    # 2. Histograma y KDE
    sns.histplot(residuals, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribución de Residuos')

    # 3. ACF
    plot_acf(residuals, ax=axes[1, 0], lags=min(len(residuals)//2 - 1, 20))
    axes[1, 0].set_title('Autocorrelación (ACF)')

    # 4. PACF
    plot_pacf(residuals, ax=axes[1, 1], lags=min(len(residuals)//2 - 1, 20))
    axes[1, 1].set_title('Autocorrelación Parcial (PACF)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_analysis(freq_name, freq_code, m, output_dir):
    """Ejecuta el análisis completo para una frecuencia dada."""
    print(f"\n{'='*50}")
    print(f"ANÁLISIS {freq_name.upper()} ({freq_code})")
    print(f"{'='*50}")

    # 1. Preparar Datos
    y = load_data()
    y_agg = aggregate_data(y, freq_code)
    
    # Split Train/Test (Último año para test, o 20% si es muy corto)
    test_size = 6 if freq_code == '2ME' else (4 if freq_code == 'QE' else 12)
    # Ajustar si hay pocos datos
    if len(y_agg) < 20:
        test_size = int(len(y_agg) * 0.2)
        
    train, test = y_agg[:-test_size], y_agg[-test_size:]
    
    print(f"Datos Totales: {len(y_agg)} | Train: {len(train)} | Test: {len(test)}")

    results = []

    # ---------------------------------------------------------
    # 2. Modelo ARIMA (No Estacional)
    # ---------------------------------------------------------
    print(f"\nEntrenando ARIMA (m=1)...")
    arima_model = auto_arima(train, seasonal=False, stepwise=True,
                             suppress_warnings=True, error_action='ignore')
    print(f"Mejor ARIMA: {arima_model.order}")
    
    y_pred_arima = arima_model.predict(n_periods=len(test))
    y_pred_arima = pd.Series(y_pred_arima, index=test.index)
    
    metrics_arima = evaluate_model(test, y_pred_arima, 'ARIMA')
    metrics_arima['AIC'] = arima_model.aic()
    metrics_arima['Order'] = str(arima_model.order)
    results.append(metrics_arima)

    # Diagnósticos ARIMA
    plot_diagnostics(arima_model.resid(), 
                     f'Diagnóstico ARIMA {arima_model.order} - {freq_name}', 
                     f'{output_dir}/diagnostico_{freq_name}_arima.png')

    # ---------------------------------------------------------
    # 3. Modelo SARIMA (Estacional)
    # ---------------------------------------------------------
    print(f"Entrenando SARIMA (m={m})...")
    try:
        sarima_model = auto_arima(train, seasonal=True, m=m, stepwise=True,
                                  suppress_warnings=True, error_action='ignore')
        print(f"Mejor SARIMA: {sarima_model.order} x {sarima_model.seasonal_order}")
        
        y_pred_sarima = sarima_model.predict(n_periods=len(test))
        y_pred_sarima = pd.Series(y_pred_sarima, index=test.index)
        
        metrics_sarima = evaluate_model(test, y_pred_sarima, 'SARIMA')
        metrics_sarima['AIC'] = sarima_model.aic()
        metrics_sarima['Order'] = str(sarima_model.order) + " x " + str(sarima_model.seasonal_order)
        results.append(metrics_sarima)

        # Diagnósticos SARIMA
        plot_diagnostics(sarima_model.resid(), 
                         f'Diagnóstico SARIMA {metrics_sarima["Order"]} - {freq_name}', 
                         f'{output_dir}/diagnostico_{freq_name}_sarima.png')
                         
    except Exception as e:
        print(f"Error entrenando SARIMA: {e}")
        y_pred_sarima = None

    # ---------------------------------------------------------
    # 4. Comparación Gráfica
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Entrenamiento', color='gray', alpha=0.6)
    plt.plot(test.index, test, label='Real (Test)', color='black', linewidth=2)
    plt.plot(y_pred_arima.index, y_pred_arima, label=f'ARIMA {metrics_arima["Order"]}', linestyle='--')
    
    if y_pred_sarima is not None:
        plt.plot(y_pred_sarima.index, y_pred_sarima, label=f'SARIMA {metrics_sarima["Order"]}', linestyle='-.')
        
    plt.title(f'Pronóstico {freq_name}: ARIMA vs SARIMA', fontsize=14)
    plt.legend()
    plt.savefig(f'{output_dir}/comparativa_{freq_name}.png')
    plt.close()

    return results

def main():
    output_dir = 'results/figures/diagnostico_arima_sarima'
    os.makedirs(output_dir, exist_ok=True)
    
    report_data = []

    # 1. Análisis Mensual (m=12)
    res_m = run_analysis('Mensual', 'M', 12, output_dir)
    for r in res_m: r['Horizonte'] = 'Mensual'; report_data.append(r)

    # 2. Análisis Bimestral (m=6)
    res_b = run_analysis('Bimestral', '2M', 6, output_dir)
    for r in res_b: r['Horizonte'] = 'Bimestral'; report_data.append(r)

    # 3. Análisis Trimestral (m=4)
    res_q = run_analysis('Trimestral', 'Q', 4, output_dir)
    for r in res_q: r['Horizonte'] = 'Trimestral'; report_data.append(r)

    # Resumen Final
    df_report = pd.DataFrame(report_data)
    df_report = df_report[['Horizonte', 'Model', 'Order', 'MAPE', 'RMSE', 'AIC']]
    
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO FINAL")
    print("="*80)
    print(df_report.to_string(index=False))
    
    # Guardar reporte
    df_report.to_csv(f'{output_dir}/reporte_comparativo.csv', index=False)
    
    # Análisis de "Mejor Modelo"
    print("\nRECOMENDACIÓN DE MODELO MÁS APROXIMADO:")
    for horizon in ['Mensual', 'Bimestral', 'Trimestral']:
        subset = df_report[df_report['Horizonte'] == horizon]
        best_row = subset.loc[subset['MAPE'].idxmin()]
        print(f"- {horizon}: El mejor modelo es **{best_row['Model']}** con MAPE de {best_row['MAPE']:.2%}")

if __name__ == "__main__":
    main()
