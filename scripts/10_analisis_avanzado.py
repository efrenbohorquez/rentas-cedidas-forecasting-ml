import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
import config
import utils
import warnings

warnings.filterwarnings('ignore')

def analyze_volatility():
    print("📉 Iniciando Análisis de Volatilidad y Riesgo (Tesis)...")
    
    # 1. Cargar datos históricos agregados
    df = utils.load_data(config.FULL_DATA_PARQUET)
    if df is None: return
    
    df = df.set_index('fecha').sort_index()
    
    # 2. Calcular Retornos (Log-Returns)
    # R_t = ln(P_t / P_{t-1})
    # La volatilidad se mide sobre los retornos, no sobre el nivel.
    df['log_retorno'] = np.log(df['recaudo'] / df['recaudo'].shift(1))
    df = df.dropna()
    
    # 3. Volatilidad Histórica (Rolling Standard Deviation)
    # Ventana de 12 meses (Anualizada)
    window = 12
    df['volatilidad_historica'] = df['log_retorno'].rolling(window=window).std() * np.sqrt(12) # Anualizada
    
    # --- GRÁFICOS ---
    utils.setup_plot_style()
    os.makedirs(config.RESULTS_DIR / 'advanced', exist_ok=True)
    
    # FIG 1: Retornos y Clústers de Volatilidad
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['log_retorno'], label='Log-Retornos Mensuales', color='grey', alpha=0.7)
    plt.title('Log-Retornos de Rentas Cedidas (Estacionariedad)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['volatilidad_historica'], label=f'Volatilidad Móvil ({window} meses)', color='red')
    plt.title('Volatilidad Histórica Anualizada (Medida de Riesgo)')
    plt.legend()
    
    plt.tight_layout()
    utils.save_plot("volatilidad_historica.png", subfolder="advanced")
    
    # 4. VaR (Value at Risk) Histórico
    # "Con un 95% de confianza, el recaudo no caerá más de X%"
    var_95 = df['log_retorno'].quantile(0.05)
    print(f"⚠️ VaR Mensual (95%): {var_95:.2%}")
    print(f"   Interpretación: Existe un 5% de probabilidad de que el recaudo caiga más de {abs(var_95):.2%} en un mes.")
    
    # FIG 2: Distribución de Retornos vs Normal
    plt.figure()
    sns.histplot(df['log_retorno'], kde=True, stat='density', label='Datos Reales')
    
    # Ajuste Normal
    mu, std = norm.fit(df['log_retorno'])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2, label='Distribución Normal')
    
    plt.title('Distribución de Retornos (Evaluación de Normalidad)')
    plt.legend()
    utils.save_plot("distribucion_retornos.png", subfolder="advanced")
    
    # 5. Fan Chart (Cono de Incertidumbre para 2026)
    # Usamos el último valor de recaudo y proyectamos riesgo
    print("🔮 Generando Fan Chart (Cono de Incertidumbre)...")
    
    last_recaudo = df['recaudo'].iloc[-1]
    last_date = df.index[-1]
    
    # Proyección simple (media histórica de crecimiento)
    # Nota: Esto es solo para ilustrar el RIESGO, no reemplaza a SARIMAX
    mu_ret = df['log_retorno'].mean()
    sigma_ret = df['log_retorno'].std()
    
    sim_months = 12
    dates_future = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=sim_months, freq='MS')
    
    # Simulación Monte Carlo (1000 iteraciones)
    simulations = []
    for i in range(1000):
        # Random Walk con Drift
        shock = np.random.normal(mu_ret, sigma_ret, sim_months)
        path_ret = np.cumsum(shock)
        path_price = last_recaudo * np.exp(path_ret)
        simulations.append(path_price)
        
    sim_matrix = np.array(simulations)
    
    # Calcular percentiles
    p10 = np.percentile(sim_matrix, 10, axis=0)
    p50 = np.percentile(sim_matrix, 50, axis=0) # Mediana
    p90 = np.percentile(sim_matrix, 90, axis=0)
    
    # Plot Fan Chart
    plt.figure()
    # Histórico reciente
    recent = df['recaudo'].iloc[-24:]
    plt.plot(recent.index, recent, label='Histórico Reciente', color='black')
    
    # Proyección
    plt.plot(dates_future, p50, label='Proyección Central (Mediana)', color='blue', linestyle='--')
    plt.fill_between(dates_future, p10, p90, color='blue', alpha=0.2, label='Intervalo de Confianza (80%)')
    
    plt.title('Fan Chart: Proyección de Riesgo (Monte Carlo)')
    plt.legend()
    utils.save_plot("fan_chart_incertidumbre.png", subfolder="advanced")
    
    # Guardar métricas de riesgo
    risk_metrics = pd.DataFrame({
        'Metrica': ['VaR (95%) Mensual', 'Volatilidad (Std) Mensual', 'Volatilidad Anualizada'],
        'Valor': [var_95, std, std * np.sqrt(12)],
        'Interpretacion': [
            'Caída máxima mensual al 95% conf.',
            'Desviación estándar de retornos mens.',
            'Riesgo anualizado'
        ]
    })
    
    risk_metrics.to_csv(config.RESULTS_DIR / 'advanced/metricas_riesgo.csv', index=False)
    print("✅ Análisis Avanzado completado. Resultados en 'results/advanced/'")

if __name__ == '__main__':
    import os
    analyze_volatility()
