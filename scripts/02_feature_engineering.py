import pandas as pd
import numpy as np
import os

def feature_engineering():
    print("🚀 Iniciando Feature Engineering (Optimizado 2020-2025 vs 2026)...")
    
    # 1. Cargar datos depurados
    ruta_input = 'data/processed/datos_depurados.parquet'
    if not os.path.exists(ruta_input):
        print(f"❌ No se encuentra: {ruta_input}")
        return

    df = pd.read_parquet(ruta_input)
    print(f"📂 Datos cargados: {df.shape}")
    
    # 2. Filtrado Municipal (Opcional pero recomendado por usuario)
    # Si existe la columna 'tipo_entidad', filtramos.
    if 'tipo_entidad' in df.columns:
        print("🔍 Filtrando solo entidades MUNICIPALES...")
        df_mun = df[df['tipo_entidad'] == 'Municipal'].copy()
        print(f"   Filas municipales: {len(df_mun)} (de {len(df)})")
        
        if len(df_mun) == 0:
            print("⚠️ ADVERTENCIA: No se encontraron municipios. Usando todo el dataset.")
            df_mun = df.copy()
    else:
        df_mun = df.copy()
        
    # 3. Agregación Temporal
    # Para este ejercicio, generaremos un dataset AGREGADO TOTAL (Suma de todos los municipios)
    # para validar los modelos. (Modelar 1100 a la vez es otro scope).
    print("📊 Generando Serie Agregada (Suma de Municipios)...")
    
    # Agrupar por fecha estandarizada
    df_agg = df_mun.groupby('fecha')['recaudo'].sum().reset_index()
    df_agg = df_agg.sort_values('fecha')
    
    # 4. Generar Features
    print("🛠️ Generando variables (Lags, Rolling, Calendario)...")
    
    # Lags
    for lag in [1, 3, 6, 12]:
        df_agg[f'recaudo_lag{lag}'] = df_agg['recaudo'].shift(lag)
        
    # Rolling Statistics
    for window in [3, 6, 12]:
        df_agg[f'rolling_mean_{window}m'] = df_agg['recaudo'].rolling(window).mean()
        df_agg[f'rolling_std_{window}m'] = df_agg['recaudo'].rolling(window).std()
        
    # Features Temporales
    df_agg['mes'] = df_agg['fecha'].dt.month
    df_agg['año'] = df_agg['fecha'].dt.year
    df_agg['trimestre'] = df_agg['fecha'].dt.quarter
    
    # Exógenas (Calendario Tributario Aproximado)
    df_agg['es_periodo_renta'] = df_agg['mes'].isin([4, 5, 8, 9, 10]).astype(int)
    df_agg['es_pico_fin_año'] = df_agg['mes'].isin([12, 1]).astype(int)
    
    # Limpieza de nulos iniciales por lags
    df_model = df_agg.dropna()
    print(f"📉 Datos finales para modelo: {df_model.shape}")
    
    # 5. Split Estricto: Train (2020-2025) vs Test (2026)
    print("✂️ Dividiendo Train (2020-2025) vs Test (2026)...")
    train = df_model[df_model['año'] <= 2025]
    test = df_model[df_model['año'] == 2026]
    
    print(f"🚂 Train: {train.shape[0]} meses ({train['fecha'].min().date()} a {train['fecha'].max().date()})")
    print(f"🧪 Test:  {test.shape[0]} meses ({test['fecha'].min().date()} a {test['fecha'].max().date()})")
    
    if len(test) == 0:
        print("⚠️ ADVERTENCIA: No hay datos para 2026 en el test set. Verificar fechas del Excel.")
    
    # 6. Guardar
    os.makedirs('data/features', exist_ok=True)
    train.to_parquet('data/features/train_mensual.parquet', index=False)
    test.to_parquet('data/features/test_mensual.parquet', index=False)

    # UNIFICACIÓN: Guardar también en CSV
    train.to_csv('data/features/train_mensual.csv', index=False)
    test.to_csv('data/features/test_mensual.csv', index=False)
    
    # Guardar dataset completo con features para análisis posterior
    df_model.to_parquet('data/features/dataset_completo.parquet', index=False)
    df_model.to_csv('data/features/dataset_completo.csv', index=False)
    
    # NUEVO: Guardar histórico completo (sin dropna) para visualizaciones
    df_agg.to_parquet('data/features/dataset_historico_completo.parquet', index=False)
    df_agg.to_csv('data/features/dataset_historico_completo.csv', index=False)
    
    print("✅ Features generados y guardados.")

if __name__ == '__main__':
    feature_engineering()
