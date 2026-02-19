import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
import joblib
import config
import utils

def run_ensemble():
    print("🚀 Iniciando Stacking Ensemble...")
    
    # 1. Cargar datos
    train = utils.load_data(config.TRAIN_DATA_FILE)
    test = utils.load_data(config.TEST_DATA_FILE)
    
    if train is None or test is None: return

    # Indices
    train = train.set_index('fecha').sort_index()
    test = test.set_index('fecha').sort_index()
    
    # 2. Generar Predicciones Out-of-Sample para Train (Level 1 Features)
    # Necesitamos predicciones de los modelos base sobre TRAIN para entrenar el meta-modelo.
    # Dado que los modelos ya están entrenados en TODO TRAIN en 03_modelos.py, 
    # no podemos usarlos directamente para predecir sobre TRAIN (sería overfitting).
    # Lo ideal es re-entrenar con Cross-Validation, pero para simplificar y usar
    # los hiperparámetros ya encontrados, haremos CV aquí usando los params guardados.
    
    print("\n🔄 Generando predicciones base (CV) para Stacking...")
    
    # --- Cargar Hiperparámetros (Simulado/Leído) ---
    # XGB params
    import json
    try:
        with open(config.MODELS_DIR / 'xgboost_params.json', 'r') as f:
            xgb_params = json.load(f)
    except:
        xgb_params = {'n_estimators': 200, 'learning_rate': 0.05} # Default

    # SARIMA params (Assume loaded or re-optimized fast if needed, or just standard)
    # Re-running SARIMA CV is expensive. For this implementation, we might approximate
    # or just do a simple split if CV is too heavy. Let's do a 3-fold CV.
    
    meta_features_train = pd.DataFrame(index=train.index)
    meta_features_test = pd.DataFrame(index=test.index)
    
    # Definir splits
    tscv = TimeSeriesSplit(n_splits=3)
    
    # --- XGBOOST CV Predictions ---
    print("   🔹 XGBoost CV...")
    cols_feat = [c for c in train.columns if 'lag' in c or 'rolling' in c or 'mes' in c or 'año' in c or 'es_' in c]
    
    xgb_preds_cv = []
    indices_cv = []
    
    # Walk-forward generation for Meta-Training
    # Note: Standard CV logic for stacking usually requires generating predictions for the whole train set 
    # effectively. TimeSeriesSplit limits us to the later part of train. 
    # We will use the 'test' part of each fold to build the meta-training set.
    
    for train_idx, val_idx in tscv.split(train):
        X_t, y_t = train.iloc[train_idx][cols_feat], train.iloc[train_idx]['recaudo']
        X_v, y_v = train.iloc[val_idx][cols_feat], train.iloc[val_idx]['recaudo']
        
        model = xgb.XGBRegressor(objective='reg:squarederror', **xgb_params)
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        
        xgb_preds_cv.extend(preds)
        indices_cv.extend(train.index[val_idx])
        
    # Align meta-features (only for the periods we have predictions for)
    meta_y_train = train.loc[indices_cv, 'recaudo']
    meta_features_train.loc[indices_cv, 'XGBoost'] = xgb_preds_cv
    meta_features_train = meta_features_train.dropna() # Keep only rows where we have predictions
    
    # --- SARIMAX CV Predictions ---
    print("   🔹 SARIMAX CV...")
    sarima_preds_cv = []
    # Using fixed simple order for speed in ensemble loop if params file not robustly parsed yet
    # or use (1,1,1) (0,1,1,12) as robust default
    
    for train_idx, val_idx in tscv.split(train):
        # Statsmodels is sensitive to indices, use numpy for safety here or ensure freq
        y_t = train.iloc[train_idx]['recaudo']
        # y_v = train.iloc[val_idx]['recaudo']
        
        try:
            mod = utils.SARIMAX(y_t, order=(1,1,1), seasonal_order=(0,1,1,12), 
                                enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            preds = res.forecast(steps=len(val_idx))
            sarima_preds_cv.extend(preds)
        except:
            sarima_preds_cv.extend([y_t.mean()] * len(val_idx))

    if len(sarima_preds_cv) == len(meta_features_train):
        meta_features_train['SARIMAX'] = sarima_preds_cv
    else:
        # Fallback alignment
        print("   ⚠️ SARIMAX size mismatch, skipping for ensemble training to avoid error.")
        
    # --- PROPHET CV Predictions ---
    # Prophet is also heavy, maybe skip for this PoC or use simple logic
    
    
    # 3. Entrenar Meta-Modelo
    print("\n🧠 Entrenando Meta-modelo...")
    # Features disponibles
    valid_features = meta_features_train.columns.tolist()
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features_train[valid_features], meta_y_train)
    
    print(f"   ⚖️ Coeficientes: {dict(zip(valid_features, meta_model.coef_))}")
    print(f"   Intercepto: {meta_model.intercept_:.2f}")
    
    # 4. Generar Predicciones Finales (Test)
    print("\n🔮 Generando predicciones finales...")
    
    # Cargar modelos full entrenados (de 03_modelos.py) o re-generar predicciones
    # Para XGBoost, podemos cargar el modelo guardado
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(config.MODELS_DIR / 'xgboost_model.json')
    pred_xgb_test = np.maximum(model_xgb.predict(test[cols_feat]), 0)
    meta_features_test['XGBoost'] = pred_xgb_test
    
    # Para SARIMAX, cargamos el pkl
    try:
        sarimax_loaded = utils.SARIMAX(train['recaudo'], order=(1,1,1), seasonal_order=(0,1,1,12)).fit(disp=False) # Refit fast or load
        # Actually loading pickle is safer if compatible
        # sarimax_res = utils.SARIMAXResults.load(config.MODELS_DIR / 'sarimax_model.pkl')
        # But statsmodels pickling can be fragile across versions/envs. Let's rely on stored result or re-predict.
        # Since 03_modelos already stored results['SARIMAX'], we can load that!
        
        # Load predictions from CSV if available to ensure exact match?
        # Or just use the one we have.
        # Let's predict fresh to be safe.
        sarimax_pred_test = sarimax_loaded.get_forecast(steps=len(test)).predicted_mean
        meta_features_test['SARIMAX'] = sarimax_pred_test.values
    except:
        meta_features_test['SARIMAX'] = np.zeros(len(test))
        
    
    # Predicción Ensemble
    ensemble_pred = np.maximum(meta_model.predict(meta_features_test[valid_features]), 0)
    
    print(f"✅ Ensemble MAPE: {utils.mape(test['recaudo'], ensemble_pred):.2f}%")
    
    # 5. Guardar
    df_res = pd.DataFrame({'fecha': test.index, 'Ensemble': ensemble_pred})
    out_path = config.RESULTS_DIR / "predictions" / 'ensemble_predictions.csv'
    df_res.to_csv(out_path, index=False)
    print(f"💾 Guardado en: {out_path}")

if __name__ == '__main__':
    run_ensemble()
