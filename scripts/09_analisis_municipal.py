import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')

# Configuración de Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Carga los datos depurados con detalle municipal."""
    ruta = 'data/processed/datos_depurados.parquet'
    if not os.path.exists(ruta):
        print(f"❌ No se encuentra: {ruta}")
        return None
    return pd.read_parquet(ruta)

def analyze_descriptive(df):
    """Análisis Descriptivo y Pareto por Municipio."""
    print("📊 Iniciando Análisis Descriptivo Municipal...")
    
    # Filtrar solo municipales
    df_mun = df[df['tipo_entidad'] == 'Municipal'].copy()
    
    # Total por Entidad
    total_por_entidad = df_mun.groupby('entidad')['recaudo'].sum().sort_values(ascending=False).reset_index()
    total_recaudo = total_por_entidad['recaudo'].sum()
    
    # Pareto
    total_por_entidad['participacion'] = total_por_entidad['recaudo'] / total_recaudo
    total_por_entidad['acumulado'] = total_por_entidad['participacion'].cumsum()
    
    # Top 80% (Pareto)
    pareto_80 = total_por_entidad[total_por_entidad['acumulado'] <= 0.82] # Aprox 80%
    print(f"   Municipios que aportan el 80% del recaudo: {len(pareto_80)} de {len(total_por_entidad)}")
    print(f"   Top 5: {total_por_entidad['entidad'].head(5).tolist()}")
    
    # Guardar stats
    os.makedirs('results/municipal', exist_ok=True)
    total_por_entidad.to_csv('results/municipal/estadisticas_descriptivas.csv', index=False)
    
    # --- GRÁFICOS DESCRIPTIVOS ---
    
    # 1. Bar Plot: Top 10 Municipios
    plt.figure(figsize=(12, 6))
    top_10 = total_por_entidad.head(10)
    sns.barplot(data=top_10, x='recaudo', y='entidad', palette='viridis')
    plt.title('Top 10 Municipios por Recaudo Total (2020-2025)')
    plt.xlabel('Recaudo Total')
    plt.ylabel('Municipio')
    plt.tight_layout()
    plt.savefig('results/municipal/top_10_municipios.png')
    plt.close()

    # 2. Pareto Plot (Top 20 para visualización de curva)
    top_20 = total_por_entidad.head(20)
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Bar plot (Left axis)
    sns.barplot(data=top_20, x='entidad', y='recaudo', color='tab:blue', ax=ax1, alpha=0.6)
    ax1.set_ylabel('Recaudo Total', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel('Municipio', fontsize=12)
    
    # Line plot (Right axis - Cumulative %)
    ax2 = ax1.twinx()
    sns.lineplot(data=top_20, x='entidad', y='acumulado', color='tab:red', marker='o', ax=ax2, linewidth=2)
    ax2.set_ylabel('% Acumulado (Pareto)', color='tab:red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 1.1)
    
    # Línea de referencia 80%
    ax2.axhline(0.8, color='grey', linestyle='--', linewidth=1)
    ax2.text(0, 0.81, '80% Recaudo', color='grey')
    
    plt.title('Diagrama de Pareto: Top 20 Municipios', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/municipal/pareto_top_20.png')
    plt.close()

    # 3. Time Series Comparison (Top 5)
    top_5_names = total_por_entidad['entidad'].head(5).tolist()
    df_top5 = df_mun[df_mun['entidad'].isin(top_5_names)]
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_top5, x='fecha', y='recaudo', hue='entidad', style='entidad', markers=True, dashes=False)
    plt.title('Evolución Temporal del Recaudo: Top 5 Municipios', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Recaudo', fontsize=12)
    plt.legend(title='Municipio', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/municipal/serie_tiempo_comparativa_top5.png')
    plt.close()

    print("   Gráficos descriptivos generados: top_10, pareto, series_top5")
    
    return total_por_entidad, df_mun

def analyze_inferential(df_mun, top_n=5):
    """Análisis Inferencial: Correlación entre municipios Top."""
    print("🔗 Iniciando Análisis de Correlación (Sincronización)...")
    
    # Obtener Top N municipios
    top_entidades = df_mun.groupby('entidad')['recaudo'].sum().nlargest(top_n).index.tolist()
    
    # Pivoted table: Index=Fecha, Columns=Entidad
    pivot_df = df_mun[df_mun['entidad'].isin(top_entidades)].pivot_table(
        index='fecha', columns='entidad', values='recaudo', aggfunc='sum'
    ).fillna(0)
    
    # Matriz de Correlación
    corr_matrix = pivot_df.corr()
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlación de Recaudo entre Top {top_n} Municipios')
    plt.tight_layout()
    plt.savefig('results/municipal/correlacion_top_municipios.png')
    plt.close()
    
    print("   Matriz de correlación guardada.")
    return top_entidades

def analyze_predictive_top(df_mun, top_entidades):
    """Genera pronósticos ARIMA para los Top Municipios."""
    print("🔮 Iniciando Predicciones Individuales (Top Municipios)...")
    
    resultados_pred = []
    
    os.makedirs('results/municipal/figures', exist_ok=True)

    for entidad in top_entidades:
        print(f"   Procesando: {entidad}...")
        
        # Preparar serie
        df_ent = df_mun[df_mun['entidad'] == entidad].groupby('fecha')['recaudo'].sum().reset_index()
        df_ent = df_ent.set_index('fecha').asfreq('MS').fillna(0)
        
        # Split
        train = df_ent[df_ent.index.year <= 2024] # Validar con último año completo
        test = df_ent[df_ent.index.year == 2025]  # Si hay datos 2025
        
        # Si no hay suficientes datos, saltar
        if len(train) < 12:
            print(f"   ⚠️ Saltando {entidad} (pocos datos)")
            continue
            
        try:
            # Auto ARIMA
            model = auto_arima(train['recaudo'], seasonal=True, m=12, 
                             suppress_warnings=True, error_action='ignore')
            
            # Forecast
            n_periods = 12
            forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train['recaudo'], label='Entrenamiento')
            if len(test) > 0:
                plt.plot(test.index, test['recaudo'], label='Real (Test)')
            
            # Crear índice futuro
            future_index = pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
            plt.plot(future_index, forecast, label='Predicción', color='red', linestyle='--')
            plt.fill_between(future_index, conf_int[:, 0], conf_int[:, 1], color='red', alpha=0.1)
            
            plt.title(f'Pronóstico Recaudo: {entidad} (Modelo: {model})')
            plt.legend()
            
            filename = f"prediccion_{entidad.replace(' ', '_').replace('/', '-')}.png"
            plt.savefig(f'results/municipal/figures/{filename}')
            plt.close()
            
            resultados_pred.append({
                'Entidad': entidad,
                'Modelo': str(model),
                'AIC': model.aic()
            })
            
        except Exception as e:
            print(f"   ❌ Error en {entidad}: {e}")
            
    # Guardar resumen modelos
    pd.DataFrame(resultados_pred).to_csv('results/municipal/resumen_modelos_municipales.csv', index=False)

def main():
    df = load_data()
    if df is None: return
    
    # 1. Descriptivo
    stats_df, df_mun = analyze_descriptive(df)
    
    # 2. Inferencial (Top 10)
    top_entidades = analyze_inferential(df_mun, top_n=10)
    
    # 3. Predictivo (Top 5 para detalle)
    analyze_predictive_top(df_mun, top_entidades[:5])
    
    print("✅ Módulo Municipal completado. Resultados en 'results/municipal/'")

if __name__ == "__main__":
    main()
