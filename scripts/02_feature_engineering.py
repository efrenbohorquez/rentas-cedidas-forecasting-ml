import pandas as pd
import numpy as np
import config
import utils
import warnings

warnings.filterwarnings('ignore')

def feature_engineering():
    print("🚀 Iniciando Feature Engineering (Refactorizado)...")
    
    # 1. Cargar datos
    df = utils.load_data(config.CLEANED_DATA_FILE)
    if df is None: return
    
    # Asegurar que fecha es datetime (al cargar de Excel puede venir como string o object si hubo conversión previa)
    df['fecha'] = pd.to_datetime(df['fecha'])

    # 2. Filtro Municipal
    if 'tipo_entidad' in df.columns:
        print("🔍 Filtrando solo entidades MUNICIPALES...")
        df_mun = df[df['tipo_entidad'] == 'Municipal'].copy()
        if df_mun.empty:
            print("⚠️ ADVERTENCIA: No se encontraron municipios. Usando todo el dataset.")
            df_mun = df.copy()
    else:
        df_mun = df.copy()
        
    # 3. Agregación Temporal
    print("📊 Generando Serie Agregada (Suma de Municipios)...")
    df_agg = df_mun.groupby('fecha')['recaudo'].sum().reset_index().sort_values('fecha')
    
    # 4. Generar Features
    print("🛠️ Generando variables (Lags, Rolling, Calendario)...")
    
    for lag in [1, 3, 6, 12]:
        df_agg[f'recaudo_lag{lag}'] = df_agg['recaudo'].shift(lag)
        
    for window in [3, 6, 12]:
        df_agg[f'rolling_mean_{window}m'] = df_agg['recaudo'].rolling(window).mean()
        df_agg[f'rolling_std_{window}m'] = df_agg['recaudo'].rolling(window).std()
        
    df_agg['mes'] = df_agg['fecha'].dt.month
    df_agg['año'] = df_agg['fecha'].dt.year
    df_agg['trimestre'] = df_agg['fecha'].dt.quarter
    
    # Exógenas
    df_agg['es_periodo_renta'] = df_agg['mes'].isin([4, 5, 8, 9, 10]).astype(int)
    df_agg['es_pico_fin_año'] = df_agg['mes'].isin([12, 1]).astype(int)

    # 📌 TÉCNICA DE MEJORA (NotebookLM - Video 7: Ingeniería de Variables Temporales):
    # Codificación Cíclica (Cyclical Encoding):
    # Transformamos variables como 'mes' y 'trimestre' usando Seno y Coseno.
    # Esto permite que el modelo entienda que el mes 12 (Diciembre) está "cerca" del mes 1 (Enero),
    # capturando correctamente la estacionalidad anual.
    df_agg['sin_mes'] = np.sin(2 * np.pi * df_agg['mes'] / 12)
    df_agg['cos_mes'] = np.cos(2 * np.pi * df_agg['mes'] / 12)
    df_agg['sin_trimestre'] = np.sin(2 * np.pi * df_agg['trimestre'] / 4)
    df_agg['cos_trimestre'] = np.cos(2 * np.pi * df_agg['trimestre'] / 4)

    # 📌 TÉCNICA DE MEJORA (NotebookLM - Video: Estacionariedad en Series Temporales):
    # Diferenciación (Differencing):
    # Calculamos la diferencia entre t y t-1 para eliminar tendencias y estabilizar la media,
    # ayudando a modelos que asumen o se benefician de datos estacionarios.
    df_agg['recaudo_diff'] = df_agg['recaudo'].diff().fillna(0)

    
    # Dataset para modelos (sin nulos)
    df_model = df_agg.dropna()
    print(f"📉 Datos finales para modelo: {df_model.shape}")
    
    # 5. Split
    # 5. Split (Global Configuration: Train <= 2025-07-31, Test >= 2025-08-01)
    split_date_train = pd.Timestamp(config.TRAIN_CUTOFF_DATE)
    print(f"✂️ Dividiendo Train (<= {config.TRAIN_CUTOFF_DATE}) vs Test (>= {config.TEST_START_DATE})...")
    
    train = df_model[df_model['fecha'] <= split_date_train]
    test = df_model[df_model['fecha'] > split_date_train]
    
    print(f"🚂 Train: {train.shape[0]} meses")
    print(f"🧪 Test:  {test.shape[0]} meses")
    
    if test.empty:
        print("⚠️ ADVERTENCIA: No hay datos para test con la fecha de corte actual.")
    
    # 6. Guardar
    utils.save_data(train, config.TRAIN_DATA_FILE)
    utils.save_data(test, config.TEST_DATA_FILE)
    utils.save_data(df_model, config.FULL_DATA_FILE)
    
    # Guardar histórico completo para visualizaciones
    utils.save_data(df_agg, config.DATA_FEATURES / 'dataset_historico_completo.xlsx')
    
    print("✅ Features generados y guardados (Refactorizado).")

if __name__ == '__main__':
    feature_engineering()
