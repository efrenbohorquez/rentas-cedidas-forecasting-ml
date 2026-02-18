import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
import config
import utils
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_sequences(data, seq_length, forecast_steps):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_steps + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + forecast_steps), 0]) # Target: Recaudo (col 0)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, output_steps):
    """
    📌 IMPORTANTE (NotebookLM - Video: Redes LSTM para Series Temporales de 'Codificando Bits'):
    Arquitectura LSTM Multistep:
    - Capa 1: LSTM con 64 unidades y return_sequences=True. Esto permite apilar otra capa LSTM,
      capturando patrones secuenciales complejos a largo plazo.
    - Dropout (0.2): Técnica de regularización para prevenir overfitting.
    - Capa 2: LSTM con 32 unidades (sin return_sequences), condensando la información temporal.
    - Capa Salida: Dense layer con 'output_steps' neuronas para predecir múltiples pasos a futuro (Horizonte Semestral).
    """
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dense(output_steps)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn_model(input_shape, output_steps):
    """
    📌 IMPORTANTE (NotebookLM - Video: Redes Convolucionales 1D para Series de Tiempo):
    Arquitectura CNN-1D:
    - Conv1D: Utiliza filtros para extraer características locales (patrones cortos) independientemente de su posición temporal.
      Similar a como las CNNs encuentran bordes en imágenes, aquí encuentran patrones en la secuencia.
    - MaxPooling1D: Reduce la dimensionalidad quedándose con las características más relevantes.
    - Flatten + Dense: Interpreta las características extraídas para realizar la regresión final.
    - Ventaja: Entrenamiento más rápido que LSTM y excelente para detectar patrones locales (picos estacionales).
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(output_steps)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def run_deep_learning_models():
    print("🚀 Iniciando Modelos Deep Learning Avanzados (NotebookLM Insights)...")
    
    # 1. Load Data with New Features (Cyclic)
    df = utils.load_data(config.FULL_DATA_FILE)
    if df is None: return

    # Ensure date format
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').set_index('fecha')
    
    # Features Selection (including new cyclic ones)
    features = ['recaudo', 'sin_mes', 'cos_mes', 'sin_trimestre', 'cos_trimestre', 'recaudo_diff']
    # Ensure they exist (fallback if script 02 wasn't run yet, though it should have matches)
    features = [f for f in features if f in df.columns]
    
    print(f"📊 Features used: {features}")
    
    data = df[features].values
    
    # 2. Scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 3. Sequence Generation
    SEQ_LENGTH = 12 # 1 year context
    FORECAST_STEPS = 6 # Predict next 6 months (Semestral view)
    
    X, y = create_sequences(data_scaled, SEQ_LENGTH, FORECAST_STEPS)
    
    # 4. Split Train/Test (Respecting Global Cutoff)
    # We need to map sequences back to dates to split correctly
    # The target of sequence i corresponds to dates[i + seq_length : i + seq_length + forecast_steps]
    # We align split based on the START of the forecast period.
    
    split_date_obj = pd.Timestamp(config.TRAIN_CUTOFF_DATE)
    dates = df.index
    
    split_idx = -1
    # Safety check: ensure we have enough data
    if len(dates) < SEQ_LENGTH + FORECAST_STEPS:
         print("❌ Error: Not enough data for sequences.")
         return

    for i in range(len(X)):
        # Date corresponding to the first predicted step
        forecast_start_date = dates[i + SEQ_LENGTH]
        if forecast_start_date > split_date_obj:
            split_idx = i
            break
            
    if split_idx == -1:
        print("⚠️ Warning: Could not find split point. Using simple 80/20")
        split_idx = int(len(X) * 0.8)
        
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print(f"🚂 Train Sequences: {X_train.shape}")
    print(f"🧪 Test Sequences:  {X_test.shape}")
    
    # 5. Model Training & Prediction
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # --- LSTM ---
    print("\n🧠 Training LSTM...")
    lstm = build_lstm_model(input_shape, FORECAST_STEPS)
    lstm.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.1)
    lstm_pred_scaled = lstm.predict(X_test)
    
    # --- CNN-1D ---
    print("\n🧬 Training CNN-1D...")
    cnn = build_cnn_model(input_shape, FORECAST_STEPS)
    cnn.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.1)
    cnn_pred_scaled = cnn.predict(X_test)
    
    # 6. Inverse Scaling (Only Target)
    # Target was column 0. We need to create a dummy array to inverse transform.
    def inverse_transform_targets(pred_scaled, scaler, n_features):
        dummy = np.zeros((pred_scaled.shape[0] * pred_scaled.shape[1], n_features))
        # Place predictions in the 0th column (recaudo)
        # Note: pred_scaled is (samples, steps), flattened to (samples*steps, )
        dummy[:, 0] = pred_scaled.flatten()
        inv_dummy = scaler.inverse_transform(dummy)
        # Reshape back to (samples, steps)
        return inv_dummy[:, 0].reshape(pred_scaled.shape)

    lstm_pred = inverse_transform_targets(lstm_pred_scaled, scaler, len(features))
    cnn_pred = inverse_transform_targets(cnn_pred_scaled, scaler, len(features))
    y_true = inverse_transform_targets(y_test, scaler, len(features))
    
    # 7. Evaluation (First Step only for simple metric, or average over horizon)
    # We will save the 1-step ahead prediction for the ensemble alignment
    # But current architecture predicts a block of 6.
    # For ensemble (monthly), we might just accept the first step of each rolling window, 
    # OR we refactor to simple 1-step forecast.
    # Let's use the 1st step of the sequence for simple alignment with monthly test set.
    
    lstm_pred_1step = lstm_pred[:, 0]
    cnn_pred_1step = cnn_pred[:, 0]
    y_true_1step = y_true[:, 0]
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_lstm = np.mean(np.abs((y_true_1step - lstm_pred_1step) / y_true_1step)) * 100
        mape_cnn = np.mean(np.abs((y_true_1step - cnn_pred_1step) / y_true_1step)) * 100
    
    print(f"\n📉 Validation Metrics (1-step forward):")
    print(f"   LSTM MAPE:   {mape_lstm:.2f}%")
    print(f"   CNN-1D MAPE: {mape_cnn:.2f}%")
    
    # 8. Export Predictions for Ensemble
    # We need to align these predictions with dates.
    # The prediction at index i corresponds to date[i + SEQ_LENGTH]
    
    # IMPORTANT: The length of predictions might be less than test if we used sequence logic
    # We must ensure dates match the predictions
    pred_len = len(lstm_pred_1step)
    start_date_idx = split_idx + SEQ_LENGTH
    
    if start_date_idx + pred_len <= len(dates):
        pred_dates = dates[start_date_idx : start_date_idx + pred_len]
    else:
        # Fallback if indices go out of bounds (shouldn't happen with correct logic)
        pred_dates = dates[-pred_len:]
    
    results_df = pd.DataFrame({
        'fecha': pred_dates,
        'DL_LSTM': lstm_pred_1step,
        'DL_CNN': cnn_pred_1step,
        'Real': y_true_1step
    })
    
    out_dir = config.RESULTS_DIR / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'predicciones_dl.xlsx' # Changed to Excel
    results_df.to_excel(out_path, index=False)
    print(f"💾 Guardado en: {out_path}")

if __name__ == '__main__':
    run_deep_learning_models()
