import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings('ignore')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def run_models():
    print("🚀 Iniciando Modelado Predictivo (Optimizado)...")
    
    # 1. Cargar datos
    try:
        train = pd.read_parquet('data/features/train_mensual.parquet')
        test = pd.read_parquet('data/features/test_mensual.parquet')
        full_df = pd.read_parquet('data/features/dataset_completo.parquet')
        
        print(f"📂 Datos cargados: Train={train.shape}, Test={test.shape}")
        
        # Lógica de Fallback si no hay datos de 2026
        if len(test) == 0:
            print("⚠️ NO HAY DATOS PARA 2026. Cambiando a modo DEMOSTRACIÓN con datos disponibles.")
            # Usar últimos 12 meses disponibles como Test
            last_date = full_df['fecha'].max()
            split_date = last_date - pd.DateOffset(months=12)
            
            print(f"   📅 Nuevo Split: Test desde {split_date} hasta {last_date}")
            
            train = full_df[full_df['fecha'] <= split_date].copy()
            test = full_df[full_df['fecha'] > split_date].copy()
            
            print(f"   🚂 Train Demo: {len(train)}")
            print(f"   🧪 Test Demo:  {len(test)}")
            
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return

    os.makedirs('models', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    
    results = {}
    
    # Preparar datos comunes
    # Asegurar índice datetime
    train = train.set_index('fecha').sort_index()
    test = test.set_index('fecha').sort_index()
    
    # --- SARIMAX ---
    print("\n📊 Entrenando SARIMAX...")
    try:
        # Sugerencia Box-Jenkins: si d=1, usar order=(p,1,q)
        # Usaremos configuración robusta por defecto (1,1,1)x(0,1,1,12)
        model_sarimax = SARIMAX(train['recaudo'], order=(1,1,1), seasonal_order=(0,1,1,12), 
                               enforce_stationarity=False, enforce_invertibility=False)
        sarimax_fit = model_sarimax.fit(disp=False)
        
        pred_sarimax = sarimax_fit.get_forecast(steps=len(test))
        y_pred_sarimax = pred_sarimax.predicted_mean
        y_pred_sarimax = np.maximum(y_pred_sarimax, 0) # Clip negativos
        
        mape_sarimax = mean_absolute_percentage_error(test['recaudo'], y_pred_sarimax)
        print(f"✅ SARIMAX MAPE: {mape_sarimax:.2f}%")
        
        results['SARIMAX'] = y_pred_sarimax.values
        sarimax_fit.save('models/sarimax_model.pkl')
        
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
        
        y_pred_prophet = forecast.iloc[-len(test):]['yhat'].values
        # Prophet a veces da negativos, clipear?
        y_pred_prophet = np.maximum(y_pred_prophet, 0)
        
        mape_prophet = mean_absolute_percentage_error(test['recaudo'], y_pred_prophet)
        print(f"✅ Prophet MAPE: {mape_prophet:.2f}%")
        
        results['Prophet'] = y_pred_prophet
        joblib.dump(m_prophet, 'models/prophet_model.pkl')
        
    except Exception as e:
        print(f"❌ Error Prophet: {e}")
        results['Prophet'] = np.zeros(len(test))

    # --- XGBOOST ---
    print("\n🌳 Entrenando XGBoost...")
    try:
        # Features numéricos
        cols_feat = [c for c in train.columns if 'lag' in c or 'rolling' in c or 'mes' in c or 'año' in c or 'es_' in c]
        X_train = train[cols_feat]
        y_train = train['recaudo']
        X_test = test[cols_feat]
        
        model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05)
        model_xgb.fit(X_train, y_train)
        
        y_pred_xgb = model_xgb.predict(X_test)
        y_pred_xgb = np.maximum(y_pred_xgb, 0)
        
        mape_xgb = mean_absolute_percentage_error(test['recaudo'], y_pred_xgb)
        print(f"✅ XGBoost MAPE: {mape_xgb:.2f}%")
        
        results['XGBoost'] = y_pred_xgb
        model_xgb.save_model('models/xgboost_model.json')
        
    except Exception as e:
        print(f"❌ Error XGBoost: {e}")
        results['XGBoost'] = np.zeros(len(test))

    # --- LSTM ---
    print("\n🧠 Entrenando LSTM...")
    try:
        # Preparación Univariada Simple
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train[['recaudo']])
        
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data)-seq_length):
                xs.append(data[i:i+seq_length])
                ys.append(data[i+seq_length])
            return np.array(xs), np.array(ys)
            
        SEQ_LENGTH = 3
        X_train_lstm, y_train_lstm = create_sequences(train_scaled, SEQ_LENGTH)
        
        X_train_t = torch.FloatTensor(X_train_lstm)
        y_train_t = torch.FloatTensor(y_train_lstm)
        
        # Modelo
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
            
        # Predicción Recursiva (Walk-forward para Test)
        model_lstm.eval()
        inputs = train_scaled[-SEQ_LENGTH:].tolist() # Iniciar con los últimos de train
        preds_scaled = []
        
        for i in range(len(test)):
            seq = torch.FloatTensor([inputs[-SEQ_LENGTH:]])
            with torch.no_grad():
                pred = model_lstm(seq).item()
            preds_scaled.append(pred)
            inputs.append([pred]) # Agregar predicción para siguiente paso
            
        y_pred_lstm = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        y_pred_lstm = np.maximum(y_pred_lstm, 0) # Clip negativos
        
        mape_lstm = mean_absolute_percentage_error(test['recaudo'], y_pred_lstm)
        print(f"✅ LSTM MAPE: {mape_lstm:.2f}%")
        
        results['LSTM'] = y_pred_lstm
        torch.save(model_lstm.state_dict(), 'models/lstm_model.pth')
        
    except Exception as e:
        print(f"❌ Error LSTM: {e}")
        results['LSTM'] = np.zeros(len(test))

    # --- GUARDAR RESULTADOS ---
    print("\n💾 Guardando resultados...")
    df_res = pd.DataFrame({
        'fecha': test.index,
        'Real': test['recaudo'].values
    })
    
    for m, preds in results.items():
        if len(preds) == len(df_res):
            df_res[m] = preds
        else:
            print(f"⚠️ Longitud incorrecta para {m}: {len(preds)} vs {len(df_res)}")
            
    df_res.to_csv('results/predictions/predicciones_comparativas.csv', index=False)
    print("✅ Proceso completado.")

if __name__ == '__main__':
    run_models()
