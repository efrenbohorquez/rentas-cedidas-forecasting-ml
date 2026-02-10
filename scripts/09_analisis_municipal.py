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
