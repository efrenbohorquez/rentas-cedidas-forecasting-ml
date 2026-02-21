
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- Configuración ---
FILE_PATH = r'd:\RENTAS\data\BaseRentasVF_limpieza21feb.xlsx'
OUTPUT_DIR = r'd:\RENTAS\results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Iniciando análisis de Regresión Lineal...")

# --- 1. Carga y Preprocesamiento ---
print("Cargando datos...")
try:
    df = pd.read_excel(FILE_PATH)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en {FILE_PATH}")
    exit()

# Limpieza básica
df.columns = [c.strip() for c in df.columns]

# Conversión de tipos
# Asumiendo que las columnas se llaman 'FechaRecaudo' y 'ValorRecaudo' como es usual
# Ajustar si los nombres son diferentes
col_fecha = 'FechaRecaudo'
col_valor = 'ValorRecaudo'

if col_fecha not in df.columns or col_valor not in df.columns:
    print(f"Advertencia: Columnas esperadas no encontradas. Columnas disponibles: {df.columns}")
    # Intento de corrección automática si los nombres varían ligeramente
    possible_date_cols = [c for c in df.columns if 'fecha' in c.lower()]
    possible_val_cols = [c for c in df.columns if 'valor' in c.lower() or 'recaudo' in c.lower()]
    if possible_date_cols: col_fecha = possible_date_cols[0]
    if possible_val_cols: col_valor = possible_val_cols[0]

df[col_fecha] = pd.to_datetime(df[col_fecha])
df[col_valor] = pd.to_numeric(df[col_valor], errors='coerce')
df = df.dropna(subset=[col_valor, col_fecha])
df = df.sort_values(by=col_fecha)

print(f"Datos cargados: {len(df)} registros desde {df[col_fecha].min()} hasta {df[col_fecha].max()}")


# --- 2. Ingeniería de Características Diarias ---
df['Mes_Index'] = np.arange(len(df))
df['Mes'] = df[col_fecha].dt.month
df['Lag_1'] = df[col_valor].shift(1)
df_daily_model = df.dropna()

# --- 3. Modelado Diario (Inicial) ---
print("\n--- Modelado Diario (Datos Originales) ---")

# RLS
X_rls = df_daily_model[['Mes_Index']]
y = df_daily_model[col_valor]

# RLM
X_rlm = df_daily_model[['Mes_Index', 'Mes', 'Lag_1']]
X_rlm = pd.get_dummies(X_rlm, columns=['Mes'], drop_first=True)

# Split sin shuffle (series de tiempo)
split_idx = int(len(df_daily_model) * 0.7)
X_train_rls, X_test_rls = X_rls.iloc[:split_idx], X_rls.iloc[split_idx:]
X_train_rlm, X_test_rlm = X_rlm.iloc[:split_idx], X_rlm.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Entrenar y Evaluar
model_rls = LinearRegression()
model_rls.fit(X_train_rls, y_train)
pred_rls = model_rls.predict(X_test_rls)

model_rlm = LinearRegression()
model_rlm.fit(X_train_rlm, y_train)
pred_rlm = model_rlm.predict(X_test_rlm)

print(f"RLS Diario - MAE: {mean_absolute_error(y_test, pred_rls):.2f}, R2: {r2_score(y_test, pred_rls):.4f}")
print(f"RLM Diario - MAE: {mean_absolute_error(y_test, pred_rlm):.2f}, R2: {r2_score(y_test, pred_rlm):.4f}")

# Visualización Diaria (Opcional, comentada para no saturar si son muchos datos)
# plt.figure(figsize=(12, 6))
# plt.plot(y_test.index, y_test, label='Real')
# plt.plot(y_test.index, pred_rls, label='Pred RLS')
# plt.plot(y_test.index, pred_rlm, label='Pred RLM')
# plt.legend()
# plt.title("Predicción Diaria")
# plt.show()


# --- 4. Agregación Mensual ---
print("\n--- Agregación Mensual ---")
df.set_index(col_fecha, inplace=True)
df_monthly = df.resample('M')[col_valor].sum().reset_index()
df_monthly['Mes_Index'] = np.arange(len(df_monthly))
df_monthly['Mes'] = df_monthly[col_fecha].dt.month

# --- 5. Ingeniería de Características Mensual (Avanzada) ---
df_monthly['Lag_1'] = df_monthly[col_valor].shift(1)
df_monthly['Valor_Log'] = np.log1p(df_monthly[col_valor])
df_monthly['Lag_12_Log'] = df_monthly['Valor_Log'].shift(12)

df_model_monthly = df_monthly.dropna().copy()
print(f"Registros mensuales para modelado: {len(df_model_monthly)}")

# --- 6. Modelado Mensual con TimeSeriesSplit ---
print("\n--- Modelado Mensual (Log-Transformado + TimeSeriesSplit) ---")

tscv = TimeSeriesSplit(n_splits=5)

X_cols_simple = ['Mes_Index']
X_cols_multi = ['Mes_Index', 'Lag_1', 'Lag_12_Log', 'Mes'] # 'Lag_1' aqui deberia ser quizas Lag_1_Log? El usuario pidio Lag_1 normal o log?
# El usuario dijo: "Cálculo de Lag_1 (recaudo del mes anterior)." y "Cálculo de Lag_12 (recaudo log-transformado...)"
# Seguiremos la instrucción literal, pero mezclando escalas puede ser raro.
# Mejor usaremos Lag_1 del valor log recaudo para consistencia en RLM Log si el target es log.
# Releyendo: "Cálculo de Lag_1 (recaudo del mes anterior)." -> No especifica log.
# Pero el modelo dice: "Regresión Lineal Múltiple (Log-transformada): Usando Mes_Index, Lag_1, Lag_12 y variables dummy..."
# Si el target es Log, lo ideal es que los lags sean del mismo dominio o transformados similarmente.
# Asumiremos Lag_1 sobre el valor logaritmico tambien para mayor coherencia, o usaremos el Lag_1 nominal.
# El prompt dice: "Cálculo de Lag_1 (recaudo del mes anterior)." (paso 5).
# Y paso 6: "... predecir ValorRecaudo_log".
# Vamos a crear Lag_1_Log tambien para probar, pero usaremos Lag_1 nominal si eso pide el usuario estrictamente.
# Sin embargo, en regresion log-log o log-linear, los lags suelen ser del target transformado.
# Voy a usar Lag_1 sobre el valor LOG para el modelo LOG, ya que tiene mas sentido matematico para series de tiempo (ARIMA en logs usa lags de logs).
# Ajuste: El usuario en paso 5 dice "Cálculo de Lag_1 (recaudo del mes anterior)" SIN especificar log.
# Y "Cálculo de Lag_12 (recaudo log-transformado...)".
# Voy a seguir la instruccion literal para Lag_1 (nominal) y Lag_12 (log).
# Espera, si Lag_1 es nominal (billones) y Target es Log (aprox 20-30), el coeficiente será muy pequeño.
# Mejor añado Lag_1_Log por si acaso o uso Lag_1 nominal. Usaré Lag_1 nominal como pide el texto literal, pero es posible que sea un error de especificación del usuario.
# Revisando el paso 8: "Lag_1 basado en la última predicción". Si predico log, tengo log. Si uso lag nominal, debo revertir log para alimentar el lag.
# OK, usaré Lag_1 del recaudo ORIGINAL.

# Preparacion de datos para RLM log
# Dummies para mes
df_model_monthly = pd.get_dummies(df_model_monthly, columns=['Mes'], prefix='Mes', drop_first=True)
dummy_cols = [c for c in df_model_monthly.columns if c.startswith('Mes_')]
# Nota: Mes_Index tambien empieza por Mes_, cuidado.
dummy_cols = [c for c in df_model_monthly.columns if c.startswith('Mes_') and c != 'Mes_Index']

X_multi = df_model_monthly[['Mes_Index', 'Lag_1', 'Lag_12_Log'] + dummy_cols]
X_simple = df_model_monthly[['Mes_Index']]
y_log = df_model_monthly['Valor_Log']

metrics_rls = []
metrics_rlm = []

# Guardar predicciones del ultimo fold para graficar
last_pred_rls = None
last_pred_rlm = None
last_y_test = None
last_index = None

for train_index, test_index in tscv.split(df_model_monthly):
    X_train_s, X_test_s = X_simple.iloc[train_index], X_simple.iloc[test_index]
    X_train_m, X_test_m = X_multi.iloc[train_index], X_multi.iloc[test_index]
    y_train, y_test = y_log.iloc[train_index], y_log.iloc[test_index]
    
    # RLS Log
    m_simple = LinearRegression()
    m_simple.fit(X_train_s, y_train)
    p_simple_log = m_simple.predict(X_test_s)
    p_simple_real = np.expm1(p_simple_log)
    
    # RLM Log
    m_multi = LinearRegression()
    m_multi.fit(X_train_m, y_train)
    p_multi_log = m_multi.predict(X_test_m)
    p_multi_real = np.expm1(p_multi_log)
    
    y_test_real = np.expm1(y_test)
    
    metrics_rls.append({
        'MAE': mean_absolute_error(y_test_real, p_simple_real),
        'R2': r2_score(y_test_real, p_simple_real)
    })
    
    metrics_rlm.append({
        'MAE': mean_absolute_error(y_test_real, p_multi_real),
        'R2': r2_score(y_test_real, p_multi_real)
    })
    
    last_pred_rls = p_simple_real
    last_pred_rlm = p_multi_real
    last_y_test = y_test_real
    last_index = df_model_monthly.iloc[test_index][col_fecha]

# Promedios
avg_mae_rls = np.mean([m['MAE'] for m in metrics_rls])
avg_r2_rls = np.mean([m['R2'] for m in metrics_rls])
avg_mae_rlm = np.mean([m['MAE'] for m in metrics_rlm])
avg_r2_rlm = np.mean([m['R2'] for m in metrics_rlm])

print(f"RLS Mensual (CV Avg) - MAE: {avg_mae_rls:,.2f}, R2: {avg_r2_rls:.4f}")
print(f"RLM Mensual (CV Avg) - MAE: {avg_mae_rlm:,.2f}, R2: {avg_r2_rlm:.4f}")

# Grafico del ultimo fold
plt.figure(figsize=(10, 6))
plt.plot(last_index, last_y_test, label='Real', marker='o')
plt.plot(last_index, last_pred_rls, label='Pred RLS (Log)', linestyle='--')
plt.plot(last_index, last_pred_rlm, label='Pred RLM (Log)', linestyle='--')
plt.title("Validación Cruzada - Último Fold (Escala Original)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'validacion_cruzada_ultimo_fold.png'))
plt.close() # Para no mostrar si no es interactivo, pero se guarda

# --- 7. Predicción Próximo Mes ---
print("\n--- Predicción para el Próximo Mes ---")
# Reentrenar con TODOS los datos disponibles (df_model_monthly)
final_model = LinearRegression()
final_model.fit(X_multi, y_log)

# Crear features para el proximo mes
last_row = df_model_monthly.iloc[-1]
next_mes_index = last_row['Mes_Index'] + 1
next_date = last_row[col_fecha] + pd.DateOffset(months=1)
next_mes_num = next_date.month

# Lag_1: Recaudo REAL del ultimo mes (no log, segun logica anterior)
next_lag_1 = np.expm1(last_row['Valor_Log']) 

# Lag_12_Log: Recaudo LOG de hace 12 meses
# Buscamos el valor de hace 12 meses en el df
target_date_12_months_ago = last_row[col_fecha] - pd.DateOffset(months=11) # Aprox, o index - 11?
# Mejor usar shift en el dataframe original full para asegurar continuidad
# Pero aqui solo necesito un valor escalar.
# Si el df está ordenado y es consecutivo mensual:
if len(df_model_monthly) >= 12:
    next_lag_12_log = df_model_monthly.iloc[-12]['Valor_Log'] # El valor de hace 12 indices (hace un año si es mensual consecutivo)
else:
    print("No hay suficientes datos historicos para Lag12 del proximo mes.")
    next_lag_12_log = np.nan # Manejar error

# Dummies
# Tengo que recrear la estructura de dummies correcta.
# cols dummies son 'Mes_2', 'Mes_3', ... 'Mes_12' (si drop_first=True y Mes 1 es base)
next_row_dict = {
    'Mes_Index': next_mes_index,
    'Lag_1': next_lag_1,
    'Lag_12_Log': next_lag_12_log
}

# Llenar dummies
for m in range(2, 13):
    col_name = f'Mes_{m}'
    if col_name in dummy_cols:
        next_row_dict[col_name] = 1 if next_mes_num == m else 0

# Convertir a DataFrame con el mismo orden de columnas que X_multi
next_X = pd.DataFrame([next_row_dict])
# Asegurar orden columnas
next_X = next_X[X_multi.columns]

# Prediccion
next_pred_log = final_model.predict(next_X)[0]
next_pred_real = np.expm1(next_pred_log)

print(f"Predicción para {next_date.strftime('%Y-%m')}: ${next_pred_real:,.2f}")

# --- 8. Visualización Final Histórica + Predicción ---
plt.figure(figsize=(12, 6))
# Datos historicos reales
plt.plot(df_monthly[col_fecha], df_monthly[col_valor], label='Histórico Real')
# Prediccion futura
plt.scatter([next_date], [next_pred_real], color='red', label='Predicción Próximo Mes', zorder=5)
plt.title(f"Pronóstico de Recaudo: {next_date.strftime('%Y-%m')}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'pronostico_proximo_mes.png'))
print(f"Gráficos guardados en {OUTPUT_DIR}")

# --- 9. Factores Influyentes ---
# Coeficientes del ultimo modelo
coeffs = pd.DataFrame({
    'Variable': X_multi.columns,
    'Coeficiente': final_model.coef_
})
print("\nFactores más influyentes (Coeficientes del Modelo RLM Log):")
print(coeffs.sort_values(by='Coeficiente', key=abs, ascending=False))

