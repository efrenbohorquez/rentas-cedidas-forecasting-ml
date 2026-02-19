
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn

# Add scripts directory to path
current_dir = Path(os.getcwd())
sys.path.append(str(current_dir))

try:
    import config
    import utils
    print("✅ Módulos locales importados.")
except ImportError as e:
    print(f"❌ Error importando módulos locales: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# --- Model Definitions ---

def entrenar_sarimax(train_series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    try:
        model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        return model.fit(disp=False)
    except Exception as e:
        print(f"   ❌ SARIMAX Error: {e}")
        return None

def entrenar_prophet(train_df, features_exogenas=[]):
    # Prophet requires 'ds' and 'y'
    df_p = train_df.rename(columns={'fecha': 'ds', 'recaudo': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, seasonality_mode='multiplicative')
    for reg in features_exogenas:
        if reg in df_p.columns:
            model.add_regressor(reg)
    model.fit(df_p)
    return model

def entrenar_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, output_dim=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def entrenar_lstm(X_train_tensor, y_train_tensor, input_dim, epochs=100):
    model = LSTMNet(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
    return model

# --- Main Pipeline ---

def run_models_for_horizon(horizon_name, train_path, test_path, seasonal_period, date_col='fecha'):
    print(f"\n" + "="*60)
    print(f"🚀 Procesando Horizonte: {horizon_name.upper()} (s={seasonal_period})")
    print(f"📂 Train: {train_path.name} | Test: {test_path.name}")
    print("="*60)

    try:
        train = pd.read_excel(train_path)
        test = pd.read_excel(test_path)
        
        # Rename date column to 'fecha' standard
        if date_col != 'fecha':
            if date_col in train.columns:
                train.rename(columns={date_col: 'fecha'}, inplace=True)
            if date_col in test.columns:
                test.rename(columns={date_col: 'fecha'}, inplace=True)
        
        # Ensure dates
        train['fecha'] = pd.to_datetime(train['fecha'])
        test['fecha'] = pd.to_datetime(test['fecha'])

        if test.empty:
            print("⚠️ Test set is empty. Skipping evaluation.")
            return []

        results = []

        # --- 1. SARIMAX ---
        print("   ⏳ Entrenando SARIMAX...")
        ts_train = train.set_index('fecha')['recaudo']
        model_s = entrenar_sarimax(ts_train, seasonal_order=(1,1,1,seasonal_period))
        if model_s:
            pred_s = model_s.forecast(steps=len(test))
            # Handle index matching if needed
            mape_s = mean_absolute_percentage_error(test['recaudo'], pred_s)
            results.append({'Horizonte': horizon_name, 'Modelo': 'SARIMAX', 'MAPE': mape_s, 
                            'Predicciones': list(pred_s)})
            print(f"   ✅ SARIMAX MAPE: {mape_s:.2%}")

        # --- 2. Prophet ---
        print("   ⏳ Entrenando Prophet...")
        # Identify regressors (numeric columns excluding target/date)
        regressors = [c for c in train.columns if c not in ['fecha', 'recaudo', 'vigencia', 'mes_num', 'año', 'mes', 'trimestre', 'semestre', 'year', 'periodo_trim', 'periodo_sem']]
        # Filter strictly numeric
        regressors = [c for c in regressors if pd.api.types.is_numeric_dtype(train[c])]
        
        model_p = entrenar_prophet(train, features_exogenas=regressors)
        
        future = test.rename(columns={'fecha': 'ds', 'recaudo': 'y'})
        future_pred = model_p.predict(future)
        pred_p = future_pred['yhat'].values
        mape_p = mean_absolute_percentage_error(test['recaudo'], pred_p)
        results.append({'Horizonte': horizon_name, 'Modelo': 'Prophet', 'MAPE': mape_p,
                        'Predicciones': list(pred_p)})
        print(f"   ✅ Prophet MAPE: {mape_p:.2%}")

        # --- 3. XGBoost ---
        print("   ⏳ Entrenando XGBoost...")
        feature_cols = [c for c in train.columns if c not in ['fecha', 'recaudo'] and pd.api.types.is_numeric_dtype(train[c])]
        X_train = train[feature_cols]
        y_train = train['recaudo']
        X_test = test[feature_cols]
        
        model_x = entrenar_xgboost(X_train, y_train)
        pred_x = model_x.predict(X_test)
        mape_x = mean_absolute_percentage_error(test['recaudo'], pred_x)
        results.append({'Horizonte': horizon_name, 'Modelo': 'XGBoost', 'MAPE': mape_x,
                        'Predicciones': list(pred_x)})
        print(f"   ✅ XGBoost MAPE: {mape_x:.2%}")

        return results

    except Exception as e:
        print(f"❌ Error procesando horizonte {horizon_name}: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    print("🎬 Iniciando Pipeline de Modelos Predictivos...")
    
    global_results = []
    
    # Define tasks: (Horizon Name, Train Path, Test Path, Seasonal Period, Date Column)
    tasks = [
        ('Mensual', config.TRAIN_DATA_MENSUAL, config.TEST_DATA_MENSUAL, 12, 'fecha'),
        ('Bimestral', config.TRAIN_DATA_BIMESTRAL, config.TEST_DATA_BIMESTRAL, 6, 'fecha_aprox'),
        ('Trimestral', config.TRAIN_DATA_TRIMESTRAL, config.TEST_DATA_TRIMESTRAL, 4, 'fecha_inicio'),
        ('Semestral', config.TRAIN_DATA_SEMESTRAL, config.TEST_DATA_SEMESTRAL, 2, 'fecha_aprox')
    ]
    
    for name, train_path, test_path, seasonal_period, date_col in tasks:
        if train_path.exists() and test_path.exists():
            res = run_models_for_horizon(name, train_path, test_path, seasonal_period, date_col)
            global_results.extend(res)
        else:
            print(f"⚠️ Archivos no encontrados para {name}: {train_path} o {test_path}")

    # Save summary
    if global_results:
        df_res = pd.DataFrame(global_results).drop(columns=['Predicciones'])
        best_models = df_res.loc[df_res.groupby("Horizonte")["MAPE"].idxmin()]
        
        print("\n🏆 Mejores Modelos por Horizonte:")
        print(best_models[['Horizonte', 'Modelo', 'MAPE']])
        
        output_file = config.DATA_PROCESSED / "resultados_modelos.xlsx"
        df_res.to_excel(output_file, index=False)
        print(f"\n💾 Resultados guardados en: {output_file}")
    else:
        print("\n⚠️ No se generaron resultados.")

if __name__ == "__main__":
    main()
