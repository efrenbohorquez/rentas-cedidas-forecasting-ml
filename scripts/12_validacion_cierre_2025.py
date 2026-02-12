import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from datetime import date
import config
import utils
import warnings

warnings.filterwarnings('ignore')

def main():
    print("🚀 Iniciando Validación de Horizonte Final 2025 (Out-of-Sample)...")

    # 1. Cargar Datos
    # Usamos el dataset completo con lags ya generados, pero aplicaremos transformaciones específicas
    try:
        df = utils.load_data(config.FULL_DATA_PARQUET)
    except:
        print("⚠️ No se encontró dataset_completo.parquet, intentando dataset_historico_completo.parquet")
        df = utils.load_data(config.DATA_FEATURES / 'dataset_historico_completo.parquet')
        
    if df is None:
        print("❌ Error: No se pudieron cargar los datos.")
        return

    # Asegurar formato de fecha
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')

    # 2. Preprocesamiento Específico
    print("🛠️ Aplicando Pipeline: Winsorization (5%) -> Log -> Fourier -> RobustScaler")

    # A. Winsorization (5% - 95%)
    lower_limit = df['recaudo'].quantile(0.05)
    upper_limit = df['recaudo'].quantile(0.95)
    df['recaudo_winsor'] = df['recaudo'].clip(lower=lower_limit, upper=upper_limit)

    # B. Transformación Logarítmica
    df['log_recaudo'] = np.log1p(df['recaudo_winsor'])

    # C. Términos de Fourier (Estacionalidad Anual)
    # Periodo = 12 meses
    df['sin_2pi'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['cos_2pi'] = np.cos(2 * np.pi * df['mes'] / 12)

    # Variables Predictoras
    features = [
        'recaudo_lag1', 'recaudo_lag3', 'recaudo_lag6', 'recaudo_lag12',
        'rolling_mean_3m', 'rolling_std_3m',
        'rolling_mean_6m', 'rolling_std_6m',
        'rolling_mean_12m', 'rolling_std_12m',
        'sin_2pi', 'cos_2pi',
        'es_periodo_renta', 'es_pico_fin_año'
    ]
    
    # Filtrar columnas que existen
    features = [f for f in features if f in df.columns]
    target = 'log_recaudo'

    # Eliminar NaNs generados por lags
    df_model = df.dropna(subset=features + [target]).copy()

    # 3. Definición de Split (Train vs Validación)
    # Train: 2020-01-02 hasta 2025-07-31
    # Test: 2025-08-01 hasta 2025-10-31 (Agosto, Septiembre, Octubre 2025)
    
    split_date_train_end = pd.Timestamp('2025-07-31')
    split_date_test_end = pd.Timestamp('2025-10-31')

    train = df_model[df_model['fecha'] <= split_date_train_end].copy()
    test = df_model[(df_model['fecha'] > split_date_train_end) & (df_model['fecha'] <= split_date_test_end)].copy()

    print(f"🚂 Entrenando con datos hasta: {train['fecha'].max().date()}")
    print(f"🧪 Validando en: {test['fecha'].min().date()} - {test['fecha'].max().date()}")

    if test.empty:
        print("❌ Error: No hay datos en el set de prueba (Verificar rango de fechas en dataset).")
        # Imprimir rango disponible para debug
        print(f"ℹ️ Rango disponible en datos: {df_model['fecha'].min().date()} - {df_model['fecha'].max().date()}")
        return

    # D. Escalamiento Robusto (Fit solo en Train)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train[features])
    y_train = train[target]

    X_test = scaler.transform(test[features])
    y_test = test[target] # Log values

    # 4. Entrenamiento (Ridge Regression)
    model = Ridge(alpha=1.0) # Alpha por defecto o optimizado si se desea, usaremos 1.0 como base.
    model.fit(X_train, y_train)

    # 5. Predicción
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log) # Inversa de Log
    y_true = np.expm1(y_test)     # Inversa de Log (o usar columna original 'recaudo')
    
    # Ajuste de consistencia: usar el recaudo REAL original para comparar, no el winsorizado
    # Pero el modelo predice el winsorizado transformado.
    # Para ser justos, comparamos contra el real original.
    y_true_real = test['recaudo'].values 

    # 6. Análisis de Resultados
    results_df = pd.DataFrame({
        'Fecha': test['fecha'].dt.date,
        'Real': y_true_real,
        'Predicho': y_pred,
        'Error_Abs': np.abs(y_true_real - y_pred),
        'Error_Pct': np.abs((y_true_real - y_pred) / y_true_real) * 100
    })

    print("\n📊 Tabla Comparativa (Agosto - Octubre 2025):")
    print(results_df.to_string(index=False))

    # Métricas
    mae = mean_absolute_error(y_true_real, y_pred)
    mape_val = mean_absolute_percentage_error(y_true_real, y_pred) * 100

    print(f"\n📉 Métricas Evaluación (3 meses):")
    print(f"   MAE:  ${mae:,.2f}")
    print(f"   MAPE: {mape_val:.2f}%")

    # Interpretación Automática
    print("\n🧠 Interpretación del Modelo:")
    total_real = results_df['Real'].sum()
    total_pred = results_df['Predicho'].sum()
    diff_total = total_pred - total_real
    
    if diff_total > 0:
        print(f"   El modelo SOBREESTIMÓ el recaudo total del trimestre en un {abs(diff_total/total_real)*100:.2f}%.")
        print("   Posible causa: El modelo asume una recuperación estacional más fuerte de la observada.")
    else:
        print(f"   El modelo SUBESTIMÓ el recaudo total del trimestre en un {abs(diff_total/total_real)*100:.2f}%.")
        print("   Posible causa: Efecto de outliers negativos recientes o cambio de tendencia no capturado a corto plazo.")

    # 7. Visualización (Zoom 2025)
    utils.setup_plot_style()
    
    # Preparar datos para plot (Todo 2024-2025 para contexto)
    plot_start_date = pd.Timestamp('2024-01-01')
    mask_plot = df_model['fecha'] >= plot_start_date
    df_plot = df_model[mask_plot].copy()
    
    plt.figure(figsize=(14, 7))
    
    # Historia
    plt.plot(df_plot['fecha'], df_plot['recaudo'], label='Histórico Real', color='gray', alpha=0.6)
    
    # Entrenamiento (hasta julio 2025)
    train_plot = train[train['fecha'] >= plot_start_date]
    # Re-predecir train para ver ajuste
    y_train_pred_log = model.predict(scaler.transform(train_plot[features]))
    y_train_pred = np.expm1(y_train_pred_log)
    plt.plot(train_plot['fecha'], y_train_pred, label='Ajuste Entrenamiento', color='blue', linestyle='--')

    # Validación (Agosto Octubre)
    plt.plot(results_df['Fecha'], results_df['Predicho'], label='Pronóstico Validación (Out-of-Sample)', color='red', marker='o', linewidth=2)
    plt.plot(results_df['Fecha'], results_df['Real'], label='Dato Real (Validación)', color='black', marker='x', linestyle='None', markersize=8)

    plt.title(f'Validación de Cierre 2025: Ridge Regression + Fourier\nMAPE: {mape_val:.2f}% (Ago-Oct)', fontsize=16)
    plt.axvline(x=split_date_train_end, color='green', linestyle=':', label='Corte Entrenamiento/Validación')
    
    plt.legend()
    plt.ylabel('Recaudo')
    plt.xlabel('Fecha')
    
    utils.save_plot("validacion_cierre_2025_zoom.png", subfolder='validation_2025')
    print("\n✅ Gráfico guardado en results/validation_2025/validacion_cierre_2025_zoom.png")

    # Guardar reporte CSV
    results_path = config.RESULTS_DIR / 'validation_2025' / 'reporte_validacion_cierre.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"💾 Reporte guardado en {results_path}")

if __name__ == "__main__":
    main()
