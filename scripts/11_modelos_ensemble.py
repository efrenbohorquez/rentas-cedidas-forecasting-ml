import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import config
import utils
import joblib
import warnings

warnings.filterwarnings('ignore')

def run_ensemble():
    print("🚀 Iniciando Modelo Híbrido/Ensemble (Fase 4)...")
    
    # 1. Cargar datos
    train = utils.load_data(config.TRAIN_DATA_FILE)
    test = utils.load_data(config.TEST_DATA_FILE)
    
    if train is None or test is None: return

    # Preparar X, y para ML
    features = [c for c in train.columns if c not in ['recaudo', 'log_recaudo', 'fecha', 'vigencia']]
    X_train = train[features]
    y_train = train['recaudo']
    X_test = test[features]
    y_test = test['recaudo']
    
    # --- NIVEL 0: MODELOS BASE OPTIMIZADOS ---
    preds_train = pd.DataFrame(index=train.index)
    preds_test = pd.DataFrame(index=test.index)
    
    # A. SARIMAX Optimizado
    print("\n🔹 Entrenando SARIMAX (Evaluando)...")
    try:
        # Forzamos alignment usando .values
        model_sarima = SARIMAX(train['recaudo'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        res_sarima = model_sarima.fit(disp=False)
        preds_train['SARIMAX'] = res_sarima.fittedvalues.values
        preds_test['SARIMAX'] = res_sarima.forecast(steps=len(test)).values
    except Exception as e:
        print(f"❌ Error SARIMAX: {e}")
        preds_train['SARIMAX'] = train['recaudo'].mean()
        preds_test['SARIMAX'] = train['recaudo'].mean()

    # B. Prophet Tuned
    try:
        print("🔹 Entrenando Prophet (Tuned)...")
        df_p = train[['fecha', 'recaudo']].rename(columns={'fecha': 'ds', 'recaudo': 'y'})
        model_prophet = Prophet(seasonality_mode='multiplicative', 
                                yearly_seasonality=True,
                                changepoint_prior_scale=0.05)
        model_prophet.fit(df_p)
        
        future = pd.DataFrame({'ds': pd.concat([train['fecha'], test['fecha']])})
        forecast = model_prophet.predict(future)
        preds_train['Prophet'] = forecast.iloc[:len(train)]['yhat'].values
        preds_test['Prophet'] = forecast.iloc[len(train):]['yhat'].values
    except Exception as e:
        print(f"❌ Error Prophet: {e}")
        preds_train['Prophet'] = train['recaudo'].mean()
        preds_test['Prophet'] = train['recaudo'].mean()

    # C. XGBoost Optimizado
    try:
        print("🔹 Entrenando XGBoost (GridSearch)...")
        xgb_best, best_params = utils.optimize_xgboost(X_train, y_train, cv_splits=3)
        preds_train['XGBoost'] = xgb_best.predict(X_train)
        preds_test['XGBoost'] = xgb_best.predict(X_test)
    except Exception as e:
        print(f"❌ Error XGBoost: {e}")
        preds_train['XGBoost'] = train['recaudo'].mean()
        preds_test['XGBoost'] = train['recaudo'].mean()
    
    # --- NIVEL 1: META-MODELO (STACKING) ---
    print("\n🔗 Entrenando Meta-Modelo (Stacking)...")
    
    # Limpieza agresiva de NaNs
    preds_train = preds_train.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    X_stack_train = preds_train
    # Asegurar alineación de indices
    y_stack_train = y_train.loc[X_stack_train.index]
    
    print(f"   📊 Dimensiones Stacking Train: {X_stack_train.shape}")
    print("   Muestra de datos (head):")
    print(X_stack_train.head())
    
    meta_model_reliable = False
    
    if len(X_stack_train) > 5: # Mínimo datos razonables
        try:
            meta_model = Ridge(alpha=0.1)
            meta_model.fit(X_stack_train, y_stack_train)
            
            weights = pd.Series(meta_model.coef_, index=X_stack_train.columns)
            print("⚖️ Pesos del Ensemble:")
            print(weights)
            
            ensemble_pred = meta_model.predict(preds_test)
            preds_test['Ensemble'] = ensemble_pred
            meta_model_reliable = True
        except Exception as e:
            print(f"❌ Error entrenando Meta-Modelo: {e}")
    else:
        print("⚠️ Datos insuficientes para Stacking robusto.")

    if not meta_model_reliable:
        print("⚠️ Usando Promedio Simple como Ensemble (Fallback).")
        preds_test['Ensemble'] = preds_test[['SARIMAX', 'Prophet', 'XGBoost']].mean(axis=1)

    # --- EVALUACIÓN ---
    print("\n🏆 Resultados Finales (MAPE 2026):")
    results = {}
    for col in preds_test.columns:
        # Clip negativos
        pred_clean = np.maximum(preds_test[col], 0)
        mape = utils.mape(y_test, pred_clean)
        results[col] = mape
        print(f"   👉 {col}: {mape:.2f}%")
        
    # Guardar Resultados
    final_df = preds_test.copy()
    final_df['Real'] = y_test.values
    final_df['fecha'] = test['fecha'].values
    utils.save_data(final_df, config.RESULTS_DIR / 'predictions/ensemble_results.csv')
    
    # Graficar
    utils.setup_plot_style()
    plt.figure(figsize=(14, 7))
    plt.plot(final_df['fecha'], final_df['Real'], label='Real', color='black', linewidth=2)
    plt.plot(final_df['fecha'], final_df['Ensemble'], label=f'Ensemble (Hybrid)', color='green', linewidth=2, linestyle='--')
    plt.plot(final_df['fecha'], final_df['XGBoost'], label='XGBoost (Best Single)', color='blue', alpha=0.5)
    
    plt.title('Comparativa: Modelo Híbrido vs Modelos Individuales')
    plt.legend()
    utils.save_plot("ensemble_forecast.png", subfolder="predictions")
    
    print("✅ Ensemble completado.")

if __name__ == '__main__':
    run_ensemble()
