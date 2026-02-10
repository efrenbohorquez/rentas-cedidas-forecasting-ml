import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import warnings

warnings.filterwarnings("ignore")

def check_stationarity(timeseries, title):
    print(f"\n🔬 Resultados de Estacionariedad: {title}")
    
    # ADF Test
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print(f"  👉 ADF Statistic: {dftest[0]:.4f}")
    print(f"  👉 p-value: {dftest[1]:.4f}")
    is_stationary_adf = dftest[1] < 0.05
    print(f"     {'✅ Estacionaria (ADF)' if is_stationary_adf else '❌ No Estacionaria (ADF)'}")

    # KPSS Test
    kpsstest = kpss(timeseries.dropna(), regression='c', nlags="auto")
    print(f"  👉 KPSS Statistic: {kpsstest[0]:.4f}")
    print(f"  👉 p-value: {kpsstest[1]:.4f}")
    is_stationary_kpss = kpsstest[1] > 0.05
    print(f"     {'✅ Estacionaria (KPSS)' if is_stationary_kpss else '❌ No Estacionaria (KPSS)'}")
    
    return is_stationary_adf and is_stationary_kpss

def plot_analysis(series, horizon_name, freq):
    plt.figure(figsize=(12, 10))
    
    # 1. Serie Temporal
    plt.subplot(4, 1, 1)
    plt.plot(series, label='Original')
    plt.title(f'Serie de Tiempo - {horizon_name}')
    plt.legend(loc='best')
    
    # 2. Descomposición (si hay suficientes datos)
    if len(series) > freq * 2:
        res = seasonal_decompose(series, model='additive', period=freq)
        plt.subplot(4, 1, 2)
        plt.plot(res.trend, label='Tendencia')
        plt.legend(loc='best')
        plt.subplot(4, 1, 3)
        plt.plot(res.seasonal, label='Estacionalidad')
        plt.legend(loc='best')
    else:
        plt.subplot(4, 1, 2)
        plt.text(0.5, 0.5, "Datos insuficientes para descomposición", ha='center')
        plt.subplot(4, 1, 3)

    # 4. ACF & PACF
    plt.subplot(4, 2, 7)
    plot_acf(series, ax=plt.gca(), lags=min(24, len(series)//2 - 1))
    plt.title('Autocorrelación (ACF)')
    
    plt.subplot(4, 2, 8)
    plot_pacf(series, ax=plt.gca(), lags=min(24, len(series)//2 - 1))
    plt.title('Autocorrelación Parcial (PACF)')
    
    plt.tight_layout()
    os.makedirs('results/figures/box_jenkins', exist_ok=True)
    plt.savefig(f'results/figures/box_jenkins/analisis_{horizon_name}.png')
    plt.close()
    print(f"📊 Gráficos guardados en results/figures/box_jenkins/analisis_{horizon_name}.png")

def run_analysis():
    print("🚀 Iniciando Análisis Box-Jenkins...")
    
    # 1. Cargar datos
    ruta = 'data/processed/datos_depurados.parquet'
    if not os.path.exists(ruta):
        print(f"❌ No se encuentra: {ruta}")
        return
        
    df = pd.read_parquet(ruta)
    
    # Filtrar por rango de análisis solicitado (2020-2025)
    # El usuario pide analizar características SOBRE este periodo.
    df = df[(df['vigencia'] >= 2020) & (df['vigencia'] <= 2025)]
    
    # Agregación Nacional para el análisis general
    # (Si se requiere municipal, habría que filtrar antes)
    print("🌍 Agregando datos a nivel Nacional para análisis estructural...")
    df_agg = df.groupby('fecha')['recaudo'].sum().reset_index().sort_values('fecha')
    df_agg.set_index('fecha', inplace=True)
    
    # --- HORIZONTE 1: MENSUAL ---
    series_m = df_agg['recaudo']
    print(f"\n📅 --- ANÁLISIS MENSUAL ({len(series_m)} periodos) ---")
    if not check_stationarity(series_m, "Mensual Original"):
        print("💡 Sugerencia: Aplicar Diferenciación (d=1)")
    plot_analysis(series_m, "Mensual", 12)
    
    # --- HORIZONTE 2: TRIMESTRAL ---
    series_q = df_agg['recaudo'].resample('Q').sum()
    print(f"\nquarterly --- ANÁLISIS TRIMESTRAL ({len(series_q)} periodos) ---")
    check_stationarity(series_q, "Trimestral Original")
    plot_analysis(series_q, "Trimestral", 4)
    
    # --- HORIZONTE 3: SEMESTRAL ---
    # Resample '6M' puede no coincidir exactamente con semestre calendario, hacemos group manual
    df_agg['semestre'] = np.where(df_agg.index.month <= 6, 1, 2)
    df_agg['periodo'] = df_agg.index.year.astype(str) + '-S' + df_agg['semestre'].astype(str)
    series_s = df_agg.groupby('periodo')['recaudo'].sum()
    
    print(f"\nsemesterly --- ANÁLISIS SEMESTRAL ({len(series_s)} periodos) ---")
    # Muy pocos datos para tests robustos (2020-2025 = 12 semestres)
    if len(series_s) < 10:
        print("⚠️ Pocos datos para pruebas estadísticas robustas.")
    check_stationarity(series_s, "Semestral Original")
    plot_analysis(series_s, "Semestral", 2)
    
    print("\n✅ Análisis Box-Jenkins completado.")

if __name__ == '__main__':
    run_analysis()
