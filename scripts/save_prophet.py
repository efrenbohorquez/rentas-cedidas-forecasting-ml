import pandas as pd
import joblib
import os

print("🔮 Entrenando Prophet...")

try:
    from prophet import Prophet
    
    # Cargar datos
    train = pd.read_parquet('data/features/train_mensual.parquet')
    test = pd.read_parquet('data/features/test_mensual.parquet')
    
    # Preparar datos para Prophet
    df_prophet = train[['fecha', 'recaudo']].rename(columns={'fecha': 'ds', 'recaudo': 'y'})
    
    model_prophet = Prophet(yearly_seasonality=True)
    model_prophet.add_country_holidays(country_name='CO')
    model_prophet.fit(df_prophet)
    
    # Guardar modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_prophet, 'models/prophet_model.pkl')
    
    print("✅ Prophet guardado en models/prophet_model.pkl")
    
except Exception as e:
    print(f"❌ Error: {e}")
