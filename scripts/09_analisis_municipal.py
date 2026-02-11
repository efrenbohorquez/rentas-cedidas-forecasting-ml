import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
import warnings
import config
import utils

# Suppress warnings
warnings.filterwarnings('ignore')

def analyze_municipalities():
    print("📊 Iniciando Análisis Descriptivo Municipal (Refactorizado)...")
    
    # 1. Cargar datos
    df = utils.load_data(config.CLEANED_DATA_PARQUET)
    if df is None: return

    # Filtro municipal
    if 'tipo_entidad' in df.columns:
        df_mun = df[df['tipo_entidad'] == 'Municipal'].copy()
    else:
        df_mun = df.copy()

    # Agrupar por Entidad
    total_por_entidad = df_mun.groupby('entidad')['recaudo'].sum().sort_values(ascending=False).reset_index()
    total_recaudo = total_por_entidad['recaudo'].sum()
    
    # Pareto Analysis
    total_por_entidad['acumulado'] = total_por_entidad['recaudo'].cumsum() / total_recaudo
    total_por_entidad['porcentaje'] = total_por_entidad['recaudo'] / total_recaudo
    
    utils.save_data(total_por_entidad, config.RESULTS_DIR / "municipal/estadisticas_descriptivas.parquet")
    
    # --- GRÁFICOS DESCRIPTIVOS ---
    utils.setup_plot_style()
    
    # 1. Bar Plot: Top 10 Municipios
    plt.figure()
    top_10 = total_por_entidad.head(10)
    sns.barplot(data=top_10, x='recaudo', y='entidad', palette=config.PALETTE_DEFAULT)
    plt.title(f'Top 10 Municipios por Recaudo Total ({config.TRAIN_START_YEAR}-{config.TRAIN_END_YEAR})')
    plt.xlabel('Recaudo Total')
    utils.save_plot("top_10_municipios.png", subfolder="municipal")

    # 2. Pareto Plot (Top 20)
    top_20 = total_por_entidad.head(20)
    fig, ax1 = plt.subplots(figsize=config.FIG_SIZE_DEFAULT)
    
    sns.barplot(data=top_20, x='entidad', y='recaudo', color='tab:blue', ax=ax1, alpha=0.6)
    ax1.set_ylabel('Recaudo Total', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    sns.lineplot(data=top_20, x='entidad', y='acumulado', color='tab:red', marker='o', ax=ax2, linewidth=2)
    ax2.set_ylabel('% Acumulado (Pareto)', color='tab:red')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(0.8, color='grey', linestyle='--', linewidth=1)
    
    plt.title('Diagrama de Pareto: Top 20 Municipios - Concentración del Ingreso')
    utils.save_plot("pareto_top_20.png", subfolder="municipal")

    # 3. Time Series Comparison (Top 5)
    top_5_names = total_por_entidad['entidad'].head(5).tolist()
    df_top5 = df_mun[df_mun['entidad'].isin(top_5_names)]
    
    plt.figure()
    sns.lineplot(data=df_top5, x='fecha', y='recaudo', hue='entidad', style='entidad', markers=True, dashes=False)
    plt.title(f'Evolución Temporal: Top 5 Municipios')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    utils.save_plot("serie_tiempo_comparativa_top5.png", subfolder="municipal")

    # --- INFERENCIAL (CORRELACIÓN) ---
    print("\n🔗 Iniciando Análisis de Correlación (Sincronización)...")
    pivot_top10 = df_mun[df_mun['entidad'].isin(top_10['entidad'])].pivot_table(
        index='fecha', columns='entidad', values='recaudo', aggfunc='sum'
    ).fillna(0)
    
    corr_matrix = pivot_top10.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlación de Recaudos (Top 10)')
    utils.save_plot("correlacion_top_municipios.png", subfolder="municipal")

    # --- PREDICTIVO (INDIVIDUAL ARIMA) ---
    print("\n🔮 Iniciando Predicciones Individuales (Top Municipios)...")
    
    preds_list = []
    
    for entidad in top_5_names[:3]:
        print(f"   Procesando: {entidad}...")
        df_ent = df_mun[df_mun['entidad'] == entidad].set_index('fecha').sort_index()
        
        # Resample mensual para asegurar continuidad
        ts = df_ent['recaudo'].resample('MS').sum().fillna(0)
        
        # Validar longitud mínima
        if len(ts) < 24:
            print(f"   ⚠️ Saltando {entidad} (pocos datos: {len(ts)})")
            continue
            
        # Ajustar modelo AutoARIMA
        try:
            model = auto_arima(ts, seasonal=True, m=12, suppress_warnings=True, stepwise=True)
            forecast = model.predict(n_periods=12) # Proyectar 1 año
            forecast = np.maximum(forecast, 0)
            
            # Graficar
            plt.figure()
            plt.plot(ts.index, ts, label='Histórico')
            future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=12, freq='MS')
            plt.plot(future_dates, forecast, label='Pronóstico (ARIMA)', color='red', linestyle='--')
            plt.title(f'Pronóstico Recaudo: {entidad}')
            plt.legend()
            
            safe_name = entidad.replace('/', '_').replace(' ', '_')
            utils.save_plot(f"prediccion_{safe_name}.png", subfolder="municipal")
            
            preds_list.append({
                'Entidad': entidad,
                'Modelo': str(model.order),
                'Promedio_Histórico': ts.mean(),
                'Promedio_Pronóstico': forecast.mean()
            })
            
        except Exception as e:
            print(f"   ❌ Error en {entidad}: {e}")

    if preds_list:
        pd.DataFrame(preds_list).to_csv(config.RESULTS_DIR / "municipal/resumen_modelos_municipales.csv", index=False)

    print("✅ Módulo Municipal completado (Refactorizado).")

if __name__ == '__main__':
    analyze_municipalities()
