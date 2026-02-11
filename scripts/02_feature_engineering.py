import pandas as pd
import numpy as np
import config
import utils
import warnings

warnings.filterwarnings('ignore')

def feature_engineering():
    print("🚀 Iniciando Feature Engineering (Refactorizado)...")
    
    # 1. Cargar datos
    df = utils.load_data(config.CLEANED_DATA_PARQUET)
    if df is None: return
    
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
    
    # Dataset para modelos (sin nulos)
    df_model = df_agg.dropna()
    print(f"📉 Datos finales para modelo: {df_model.shape}")
    
    # 5. Split
    print(f"✂️ Dividiendo Train (mask <= {config.TRAIN_END_YEAR}) vs Test ({config.TEST_YEAR})...")
    train = df_model[df_model['año'] <= config.TRAIN_END_YEAR]
    test = df_model[df_model['año'] == config.TEST_YEAR]
    
    print(f"🚂 Train: {train.shape[0]} meses")
    print(f"🧪 Test:  {test.shape[0]} meses")
    
    if test.empty:
        print("⚠️ ADVERTENCIA: No hay datos para test.")
    
    # 6. Guardar
    utils.save_data(train, config.TRAIN_DATA_PARQUET)
    utils.save_data(test, config.TEST_DATA_PARQUET)
    utils.save_data(df_model, config.FULL_DATA_PARQUET)
    
    # Guardar histórico completo para visualizaciones
    utils.save_data(df_agg, config.DATA_FEATURES / 'dataset_historico_completo.parquet')
    
    print("✅ Features generados y guardados (Refactorizado).")

if __name__ == '__main__':
    feature_engineering()
