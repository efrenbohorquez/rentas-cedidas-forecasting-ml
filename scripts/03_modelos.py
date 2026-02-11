import pandas as pd
import numpy as np
import joblib
import warnings
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import config
import utils

# Suppress warnings
warnings.filterwarnings('ignore')

def run_models():
    print("🚀 Iniciando Modelado Predictivo (Refactorizado)...")
    
    # 1. Cargar datos
    train = utils.load_data(config.TRAIN_DATA_PARQUET)
    test = utils.load_data(config.TEST_DATA_PARQUET)
    full_df = utils.load_data(config.FULL_DATA_PARQUET)
    
    if train is None or test is None: return

    # Lógica de Fallback
    if len(test) == 0:
        print("⚠️ NO HAY DATOS PARA TEST (2026). Modo DEMO Activado.")
        last_date = full_df['fecha'].max()
        split_date = last_date - pd.DateOffset(months=12)
        train = full_df[full_df['fecha'] <= split_date].copy()
        test = full_df[full_df['fecha'] > split_date].copy()
        print(f"   📅 Split Demo: {split_date.date()}")
        
    # Directorios
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_dir = config.RESULTS_DIR / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Indices
    train = train.set_index('fecha').sort_index()
    test = test.set_index('fecha').sort_index()
    
    # --- SARIMAX ---
    print("\n📊 Entrenando SARIMAX (Optimizado)...")
    try:
        # Optimization logic moved to utils
        best_sarima, best_params_sarima = utils.optimize_sarima(train['recaudo'], seasonal_period=12)
        
        # Fit best model on full train data
        sarimax_fit = best_sarima
        
        y_pred = np.maximum(sarimax_fit.get_forecast(steps=len(test)).predicted_mean, 0)
        
        print(f"✅ SARIMAX MAPE: {utils.mape(test['recaudo'], y_pred):.2f}%")
        results['SARIMAX'] = y_pred.values
        sarimax_fit.save(config.MODELS_DIR / 'sarimax_model.pkl')
        
        # Save params
        with open(config.MODELS_DIR / 'sarimax_params.txt', 'w') as f:
            f.write(str(best_params_sarima))
            
    except Exception as e:
        print(f"❌ Error SARIMAX: {e}")
        results['SARIMAX'] = np.zeros(len(test))

    # --- PROPHET ---
    print("\n🔮 Entrenando Prophet...")
    try:
        df_prophet = train.reset_index()[['fecha', 'recaudo']].rename(columns={'fecha': 'ds', 'recaudo': 'y'})
        m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m_prophet.add_country_holidays(country_name='CO')
        m_prophet.fit(df_prophet)
        
        future = m_prophet.make_future_dataframe(periods=len(test), freq='MS')
        forecast = m_prophet.predict(future)
        y_pred = np.maximum(forecast.iloc[-len(test):]['yhat'].values, 0)
        
        print(f"✅ Prophet MAPE: {utils.mape(test['recaudo'], y_pred):.2f}%")
        results['Prophet'] = y_pred
        joblib.dump(m_prophet, config.MODELS_DIR / 'prophet_model.pkl')
    except Exception as e:
        print(f"❌ Error Prophet: {e}")
        results['Prophet'] = np.zeros(len(test))

    # --- XGBOOST ---
    print("\n🌳 Entrenando XGBoost (Optimizado)...")
    try:
        cols_feat = [c for c in train.columns if 'lag' in c or 'rolling' in c or 'mes' in c or 'año' in c or 'es_' in c]
        
        best_xgb, best_params_xgb = utils.optimize_xgboost(train[cols_feat], train['recaudo'])
        
        # Fit is already done in optimize via GridSearchCV, but best_estimator_ is returned fitted on the grid search data?
        # Actually GridSearchCV refits on the whole X, y passed to it by default (refit=True).
        # So we can just use best_xgb.
        
        y_pred = np.maximum(best_xgb.predict(test[cols_feat]), 0)
        
        print(f"✅ XGBoost MAPE: {utils.mape(test['recaudo'], y_pred):.2f}%")
        results['XGBoost'] = y_pred
        best_xgb.save_model(config.MODELS_DIR / 'xgboost_model.json')
        
        # Save params
        import json
        with open(config.MODELS_DIR / 'xgboost_params.json', 'w') as f:
            json.dump(best_params_xgb, f)
            
    except Exception as e:
        print(f"❌ Error XGBoost: {e}")
        results['XGBoost'] = np.zeros(len(test))

    # --- LSTM ---
    print("\n🧠 Entrenando LSTM...")
    try:
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train[['recaudo']])
        
        SEQ_LENGTH = 3
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data)-seq_length):
                xs.append(data[i:i+seq_length])
                ys.append(data[i+seq_length])
            return np.array(xs), np.array(ys)
            
        X_train_lstm, y_train_lstm = create_sequences(train_scaled, SEQ_LENGTH)
        X_train_t = torch.FloatTensor(X_train_lstm)
        y_train_t = torch.FloatTensor(y_train_lstm)
        
        class LSTMNet(nn.Module):
            def __init__(self):
                super(LSTMNet, self).__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
                self.linear = nn.Linear(64, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.linear(out[:, -1, :])
        
        model_lstm = LSTMNet()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.005)
        
        model_lstm.train()
        for epoch in range(150):
            optimizer.zero_grad()
            outputs = model_lstm(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
        model_lstm.eval()
        inputs = train_scaled[-SEQ_LENGTH:].tolist()
        preds_scaled = []
        for i in range(len(test)):
            seq = torch.FloatTensor([inputs[-SEQ_LENGTH:]])
            with torch.no_grad():
                pred = model_lstm(seq).item()
            preds_scaled.append(pred)
            inputs.append([pred])
            
        y_pred = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        y_pred = np.maximum(y_pred, 0)
        
        print(f"✅ LSTM MAPE: {utils.mape(test['recaudo'], y_pred):.2f}%")
        results['LSTM'] = y_pred
        torch.save(model_lstm.state_dict(), config.MODELS_DIR / 'lstm_model.pth')
    except Exception as e:
        print(f"❌ Error LSTM: {e}")
        results['LSTM'] = np.zeros(len(test))

    # --- GUARDAR RESULTADOS ---
    print("\n💾 Guardando resultados...")
    df_res = pd.DataFrame({'fecha': test.index, 'Real': test['recaudo'].values})
    
    for m, preds in results.items():
        if len(preds) == len(df_res):
            df_res[m] = preds
            
    # Guardar CSV directamente para compatibilidad
    csv_path = results_dir / 'predicciones_comparativas.csv'
    df_res.to_csv(csv_path, index=False)
    print(f"✅ Resultados guardados en: {csv_path}")

if __name__ == '__main__':
    run_models()
