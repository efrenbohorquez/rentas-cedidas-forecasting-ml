
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import config

# Suppress warnings
warnings.filterwarnings('ignore')

def crear_lags(df, columna_target, lags=[1, 3, 6, 12]):
    """
    Crea variables de rezago para capturar dependencias temporales.
    """
    df_lags = df.copy()
    for lag in lags:
        df_lags[f'recaudo_lag{lag}'] = df_lags[columna_target].shift(lag)
        # print(f"✅ Creado: recaudo_lag{lag} (rezago de {lag} mes(es))")
    return df_lags

def crear_rolling_features(df, columna_target, ventanas=[3, 6, 12]):
    """
    Crea estadísticas de ventanas móviles.
    """
    df_rolling = df.copy()
    for ventana in ventanas:
        df_rolling[f'rolling_mean_{ventana}m'] = df_rolling[columna_target].rolling(window=ventana).mean()
        df_rolling[f'rolling_std_{ventana}m'] = df_rolling[columna_target].rolling(window=ventana).std()
        df_rolling[f'rolling_min_{ventana}m'] = df_rolling[columna_target].rolling(window=ventana).min()
        df_rolling[f'rolling_max_{ventana}m'] = df_rolling[columna_target].rolling(window=ventana).max()
        # print(f"✅ Creadas rolling features para ventana de {ventana} meses")
    return df_rolling

def crear_ewma_features(df, columna_target, alphas=[0.3, 0.5, 0.7]):
    """
    Crea features con media móvil exponencial.
    """
    df_ewma = df.copy()
    for alpha in alphas:
        df_ewma[f'ewma_alpha{alpha}'] = df_ewma[columna_target].ewm(alpha=alpha, adjust=False).mean()
        # print(f"✅ Creado EWMA con alpha={alpha}")
    return df_ewma

def crear_features_temporales(df, columna_fecha):
    """
    Crea features basados en el calendario.
    """
    df_temp = df.copy()
    df_temp[columna_fecha] = pd.to_datetime(df_temp[columna_fecha])
    
    df_temp['año'] = df_temp[columna_fecha].dt.year
    df_temp['mes'] = df_temp[columna_fecha].dt.month
    df_temp['trimestre'] = df_temp[columna_fecha].dt.quarter
    df_temp['semestre'] = df_temp[columna_fecha].dt.month.apply(lambda x: 1 if x <= 6 else 2)
    
    # Features cíclicos
    df_temp['mes_sin'] = np.sin(2 * np.pi * df_temp['mes'] / 12)
    df_temp['mes_cos'] = np.cos(2 * np.pi * df_temp['mes'] / 12)
    
    # One-hot encoding de mes
    for mes in range(1, 13):
        df_temp[f'es_mes_{mes}'] = (df_temp['mes'] == mes).astype(int)
        
    # print("✅ Features temporales creados")
    return df_temp

def crear_regresores_tributarios(df, columna_fecha=None):
    """
    Crea variables binarias para periodos tributarios clave.
    """
    df_trib = df.copy()
    
    if columna_fecha:
        df_trib[columna_fecha] = pd.to_datetime(df_trib[columna_fecha])
        mes = df_trib[columna_fecha].dt.month
    elif 'mes' in df_trib.columns:
        mes = df_trib['mes']
    else:
        print("⚠️ No se encontró columna de mes")
        return df_trib
    
    # Periodo de Renta (abril-mayo)
    df_trib['es_periodo_renta'] = mes.isin([4, 5]).astype(int)
    
    # Periodo ICA (marzo, julio)
    df_trib['es_periodo_ica'] = mes.isin([3, 7]).astype(int)
    
    # Periodo de cierre fiscal (diciembre)
    df_trib['es_cierre_fiscal'] = (mes == 12).astype(int)
    
    # Periodo post-festivo (enero)
    df_trib['es_enero'] = (mes == 1).astype(int)
    
    # print("✅ Regresores tributarios creados")
    return df_trib

def crear_features_momentum(df, columna_target):
    """
    Crea features de cambio y aceleración.
    """
    df_mom = df.copy()
    
    df_mom['diff_1'] = df_mom[columna_target].diff(1)
    df_mom['diff_2'] = df_mom[columna_target].diff(1).diff(1)
    df_mom['pct_change'] = df_mom[columna_target].pct_change()
    df_mom['momentum_3m'] = df_mom['diff_1'].rolling(window=3).sum()
    
    # print("✅ Features de momentum creados")
    return df_mom

def pipeline_feature_engineering(df, columna_target, columna_fecha):
    """
    Ejecuta el pipeline completo de feature engineering.
    """
    print("="*80)
    print("INICIANDO PIPELINE DE FEATURE ENGINEERING")
    print("="*80)
    
    df_features = df.copy()
    
    # 1. Features temporales
    print("🔄 Paso 1: Features Temporales")
    df_features = crear_features_temporales(df_features, columna_fecha)
    
    # 2. Regresores tributarios
    print("🔄 Paso 2: Regresores Tributarios")
    df_features = crear_regresores_tributarios(df_features, columna_fecha=columna_fecha)
    
    # 3. Lags
    print("🔄 Paso 3: Variables de Rezago")
    df_features = crear_lags(df_features, columna_target, lags=[1, 3, 6, 12])
    
    # 4. Rolling features
    print("🔄 Paso 4: Rolling Statistics")
    df_features = crear_rolling_features(df_features, columna_target, ventanas=[3, 6, 12])
    
    # 5. EWMA
    print("🔄 Paso 5: EWMA Features")
    df_features = crear_ewma_features(df_features, columna_target, alphas=[0.3, 0.5, 0.7])
    
    # 6. Momentum
    print("🔄 Paso 6: Momentum Features")
    df_features = crear_features_momentum(df_features, columna_target)
    
    print("="*80)
    print("✅ PIPELINE COMPLETADO")
    print(f"Features creados: {df_features.shape[1] - df.shape[1]}")
    print(f"Total de columnas: {df_features.shape[1]}")
    print("="*80)
    
    return df_features

def agregar_trimestral(df_mensual, columna_target, columna_fecha):
    """
    Agrega datos mensuales a trimestral.
    """
    df_trim = df_mensual.copy()
    df_trim[columna_fecha] = pd.to_datetime(df_trim[columna_fecha])
    
    df_trim['periodo_trim'] = df_trim[columna_fecha].dt.to_period('Q')
    
    # Definir agregaciones for all columns generally, primarily Sum for target, Max for binary/flags, One for time
    # But simplifies to specific logic in notebook
    
    agg_dict = {
        columna_target: 'sum',
        'año': 'first',
        'trimestre': 'first',
        'es_periodo_renta': 'max',
        'es_periodo_ica': 'max',
        'es_cierre_fiscal': 'max',
        'es_enero': 'max'
    }
    
    # Handle missing columns safely
    agg_dict = {k: v for k, v in agg_dict.items() if k in df_trim.columns}

    df_trimestral = df_trim.groupby('periodo_trim').agg(agg_dict).reset_index()
    
    print(f"✅ Dataset trimestral creado: {len(df_trimestral)} trimestres")
    return df_trimestral

def agregar_semestral(df_mensual, columna_target, columna_fecha):
    """
    Agrega datos mensuales a semestral.
    """
    df_sem = df_mensual.copy()
    df_sem[columna_fecha] = pd.to_datetime(df_sem[columna_fecha])
    
    # Use existing semestre col or create it
    if 'semestre' not in df_sem.columns:
         df_sem['semestre'] = df_sem[columna_fecha].dt.month.apply(lambda x: 1 if x <= 6 else 2)

    df_semestral = df_sem.groupby(['año', 'semestre']).agg({
        columna_target: 'sum',
        'es_periodo_renta': 'max',
        'es_periodo_ica': 'max',
        'es_cierre_fiscal': 'max',
        'es_enero': 'max'
    }).reset_index()
    
    df_semestral['periodo_sem'] = df_semestral['año'].astype(str) + '-S' + df_semestral['semestre'].astype(str)
    
    print(f"✅ Dataset semestral creado: {len(df_semestral)} semestres")
    return df_semestral

def agregar_bimestral(df_mensual, columna_target, columna_fecha):
    """
    Agrega datos mensuales a bimestral.
    """
    df_bim = df_mensual.copy()
    df_bim[columna_fecha] = pd.to_datetime(df_bim[columna_fecha])
    
    # Calculate bimestre
    df_bim['bimestre'] = (df_bim[columna_fecha].dt.month - 1) // 2 + 1

    df_bimestral = df_bim.groupby(['año', 'bimestre']).agg({
        columna_target: 'sum',
        'es_periodo_renta': 'max',
        'es_periodo_ica': 'max',
        'es_cierre_fiscal': 'max',
        'es_enero': 'max'
    }).reset_index()
    
    df_bimestral['periodo_bim'] = df_bimestral['año'].astype(str) + '-B' + df_bimestral['bimestre'].astype(str)
    
    # Construct a date for splitting
    # approximate: year + (bimestre-1)*2 + 1 month
    df_bimestral['fecha_aprox'] = pd.to_datetime(
        df_bimestral.assign(day=1, month=lambda x: (x.bimestre-1)*2 + 1)
        .rename(columns={'año': 'year'})
        [['year', 'month', 'day']]
    )

    print(f"✅ Dataset bimestral creado: {len(df_bimestral)} bimestres")
    return df_bimestral

def dividir_train_test(df, columna_fecha, train_cutoff_date, test_start_date):
    """
    Divide datos en train y test basado en las fechas globales.
    """
    df_copy = df.copy()
    df_copy[columna_fecha] = pd.to_datetime(df_copy[columna_fecha])
    
    # Assuming the date column is compatible with string comparison or conversions
    # Ideally standardizing to timestamps for comparison
    
    # Train: <= TRAIN_CUTOFF_DATE
    train = df_copy[df_copy[columna_fecha] <= pd.Timestamp(train_cutoff_date)]
    
    # Test: >= TEST_START_DATE
    test = df_copy[df_copy[columna_fecha] >= pd.Timestamp(test_start_date)]
    
    # For quarterly/semestral, dates might be Period or abstract. 
    # If they are not Timestamps, we might need logic based on year/period.
    # But notebook showed simple year logic. We will try to be more precise if possible
    # or fallback to specific logic per horizon.
    
    # However, 'agregar_trimestral' returns 'periodo_trim' as Period. 
    # We might need to convert 'periodo_trim' back to timestamp (start time) for splitting?
    # Or keep it simple for now. 
    
    return train, test

def dividir_train_test_año(df, columna_test_anio=2026):
      # Fallback logic from notebook if date comparison is tricky
      if 'año' in df.columns:
          train = df[df['año'] < columna_test_anio]
          test = df[df['año'] >= columna_test_anio]
          return train, test
      return df, df

import traceback

def main():
    try:
        print("📂 Cargando datos depurados...")
        if not config.CLEANED_DATA_FILE.exists():
            print(f"❌ Error: No se encuentra {config.CLEANED_DATA_FILE}")
            return

        df = pd.read_excel(config.CLEANED_DATA_FILE)
        df.columns = df.columns.str.strip()
        print(f"📊 Columnas cargadas: {df.columns.tolist()}")
        
        # Ensure date column 'fecha' exists
        if 'fecha' not in df.columns:
            print("⚠️ 'fecha' column missing, attempting reconstruction...")
            if 'vigencia' in df.columns and 'mes_num' in df.columns:
                 # Ensure types are correct for dict construction
                 df_dict = pd.DataFrame({'year': df['vigencia'], 'month': df['mes_num'], 'day': 1})
                 df['fecha'] = pd.to_datetime(df_dict)
            else:
                print("❌ Error: No se puede reconstruir fecha. Faltan columna vigencia o mes_num")
                return
        else:
            print("✅ Columna 'fecha' detectada correctamente.")
        
        # Check for NaNs in fecha
        if df['fecha'].isnull().any():
             print(f"⚠️ {df['fecha'].isnull().sum()} fechas nulas encontradas. Eliminando...")
             df = df.dropna(subset=['fecha'])

        print("📊 Agregando a nivel mensual global...")
        # The cleaned data already has column 'recaudo'
        if 'recaudo' not in df.columns and 'recaudo_neto' in df.columns:
             df.rename(columns={'recaudo_neto': 'recaudo'}, inplace=True)
             
        if 'recaudo' not in df.columns:
            print(f"❌ Error: No se encuentra columna recaudo. Columnas disponibles: {df.columns.tolist()}")
            return

        df_monthly = df.groupby('fecha')['recaudo'].sum().reset_index()
        print(f"✅ Agregado mensual creado: {len(df_monthly)} registros")
        
        # Run Pipeline
        print("🚀 Ejecutando pipeline...")
        df_features_monthly = pipeline_feature_engineering(df_monthly, 'recaudo', 'fecha')
        
        # Remove rows with NaNs from lags if strategy requires
        df_features_monthly_clean = df_features_monthly.dropna().copy()
        print(f"📉 Filas eliminadas por NaN (lags): {len(df_features_monthly) - len(df_features_monthly_clean)}")
        
        # Create Horizons
        print("\n🌐 Creando datasets multi-horizonte...")
        
        # Monthly
        train_m, test_m = dividir_train_test(
            df_features_monthly_clean, 'fecha', 
            config.TRAIN_CUTOFF_DATE, config.TEST_START_DATE
        )
        print(f"✅ Mensual split: Train={len(train_m)}, Test={len(test_m)}")
        
        # Bimonthly (Bimensual)
        df_bimestral = agregar_bimestral(df_features_monthly, 'recaudo', 'fecha')
        train_b, test_b = dividir_train_test(
            df_bimestral, 'fecha_aprox',
            config.TRAIN_CUTOFF_DATE, config.TEST_START_DATE
        )
        print(f"✅ Bimestral split: Train={len(train_b)}, Test={len(test_b)}")

        # Quarterly
        # Use df_features_monthly to have all temporal and tributary features available for aggregation
        df_trimestral = agregar_trimestral(df_features_monthly, 'recaudo', 'fecha')
        
        # We CANNOT split trimestral easily by exact date if 'periodo_trim' is Period.
        # Converting to timestamp for split.
        if 'periodo_trim' in df_trimestral.columns:
            df_trimestral['fecha_inicio'] = df_trimestral['periodo_trim'].dt.start_time
            train_q, test_q = dividir_train_test(
                df_trimestral, 'fecha_inicio',
                config.TRAIN_CUTOFF_DATE, config.TEST_START_DATE
            )
            print(f"✅ Trimestral split: Train={len(train_q)}, Test={len(test_q)}")

        # Semestral
        df_semestral = agregar_semestral(df_features_monthly, 'recaudo', 'fecha')
        # Construct a date for splitting
        # approximate: year + (semestre-1)*6 + 1 month
        df_semestral['fecha_aprox'] = pd.to_datetime(
            df_semestral.assign(day=1, month=lambda x: (x.semestre-1)*6 + 1)
            .rename(columns={'año': 'year'})
            [['year', 'month', 'day']]
        )
        train_s, test_s = dividir_train_test(
            df_semestral, 'fecha_aprox',
            config.TRAIN_CUTOFF_DATE, config.TEST_START_DATE
        )
        print(f"✅ Semestral split: Train={len(train_s)}, Test={len(test_s)}")

        # Save Files
        print("\n💾 Guardando archivos...")
        
        # Monthly
        df_features_monthly_clean.to_excel(config.FULL_DATA_FILE, index=False)
        train_m.to_excel(config.TRAIN_DATA_FILE, index=False)
        test_m.to_excel(config.TEST_DATA_FILE, index=False)
        
        # Quarterly
        df_trimestral.to_excel(config.FULL_DATA_TRIMESTRAL, index=False)
        train_q.to_excel(config.TRAIN_DATA_TRIMESTRAL, index=False)
        test_q.to_excel(config.TEST_DATA_TRIMESTRAL, index=False)
        
        # Semestral
        df_semestral.to_excel(config.FULL_DATA_SEMESTRAL, index=False)
        train_s.to_excel(config.TRAIN_DATA_SEMESTRAL, index=False)
        test_s.to_excel(config.TEST_DATA_SEMESTRAL, index=False)

        # Bimonthly
        df_bimestral.to_excel(config.FULL_DATA_BIMESTRAL, index=False)
        train_b.to_excel(config.TRAIN_DATA_BIMESTRAL, index=False)
        test_b.to_excel(config.TEST_DATA_BIMESTRAL, index=False)
        
        
        print("✅ Proceso finalizado exitosamente.")

    except Exception as e:
        print("\n❌ CRITICAL ERROR IN MAIN:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
