import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calcular_metricas(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid div by zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return {'MAPE': np.nan, 'RMSE': np.nan, 'MAE': np.nan}
        
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}

def validar_modelos():
    print("🚀 Iniciando Validación Multi-Horizonte (Final)...")
    
    # 1. Cargar predicciones
    ruta_pred = 'results/predictions/predicciones_comparativas.csv'
    if not os.path.exists(ruta_pred):
        print(f"❌ No se encuentra: {ruta_pred}")
        return

    try:
        df = pd.read_csv(ruta_pred)
        df['fecha'] = pd.to_datetime(df['fecha'])
        print(f"📂 Predicciones cargadas (Rango: {df['fecha'].min().date()} - {df['fecha'].max().date()})")
    except Exception as e:
        print(f"❌ Error leyendo CSV: {e}")
        return
    
    # Modelos disponibles
    modelos = [c for c in df.columns if c not in ['fecha', 'Real']]
    
    # Resultados acumulados
    best_models = {}
    reporte_txt = []
    reporte_txt.append("=== REPORTE DE VALIDACIÓN CRUZADA EN EL TIEMPO ===\n")
    
    # 2. Validación Mensual
    print("\n📅 Validación MENSUAL (Operativo)")
    res_mensual = []
    for m in modelos:
        met = calcular_metricas(df['Real'], df[m])
        res_mensual.append({'Modelo': m, **met})
        
    df_res_mensual = pd.DataFrame(res_mensual).sort_values('MAPE')
    best_mensual = df_res_mensual.iloc[0]
    best_models['Mensual'] = best_mensual['Modelo']
    
    reporte_txt.append(f"--- HORIZONTE MENSUAL ---")
    reporte_txt.append(f"Mejor Modelo: {best_mensual['Modelo']} (MAPE: {best_mensual['MAPE']:.2f}%)")
    reporte_txt.append(df_res_mensual.to_string())
    reporte_txt.append("\n")
    print(df_res_mensual)
    
    # 3. Validación Trimestral
    print("\nquarterly Validación TRIMESTRAL (Táctico)")
    df['trimestre'] = df['fecha'].dt.to_period('Q')
    df_q = df.groupby('trimestre')[['Real'] + modelos].sum().reset_index()
    
    res_trimestral = []
    for m in modelos:
        met = calcular_metricas(df_q['Real'], df_q[m])
        res_trimestral.append({'Modelo': m, **met})
        
    df_res_trimestral = pd.DataFrame(res_trimestral).sort_values('MAPE')
    best_trimestre = df_res_trimestral.iloc[0]
    best_models['Trimestral'] = best_trimestre['Modelo']
    
    reporte_txt.append(f"--- HORIZONTE TRIMESTRAL ---")
    reporte_txt.append(f"Mejor Modelo: {best_trimestre['Modelo']} (MAPE: {best_trimestre['MAPE']:.2f}%)")
    reporte_txt.append(df_res_trimestral.to_string())
    reporte_txt.append("\n")
    print(df_res_trimestral)
    
    # 4. Validación Semestral
    print("\nsemesterly Validación SEMESTRAL (Estratégico)")
    # Asignar semestre
    df['semestre'] = np.where(df['fecha'].dt.month <= 6, 1, 2)
    df['periodo_sem'] = df['fecha'].dt.year.astype(str) + '-S' + df['semestre'].astype(str)
    
    df_s = df.groupby('periodo_sem')[['Real'] + modelos].sum().reset_index()
    
    res_semestral = []
    for m in modelos:
        met = calcular_metricas(df_s['Real'], df_s[m])
        res_semestral.append({'Modelo': m, **met})
        
    df_res_semestral = pd.DataFrame(res_semestral).sort_values('MAPE')
    best_semestre = df_res_semestral.iloc[0]
    best_models['Semestral'] = best_semestre['Modelo']
    
    reporte_txt.append(f"--- HORIZONTE SEMESTRAL ---")
    reporte_txt.append(f"Mejor Modelo: {best_semestre['Modelo']} (MAPE: {best_semestre['MAPE']:.2f}%)")
    reporte_txt.append(df_res_semestral.to_string())
    print(df_res_semestral)
    
    # 5. Guardar Reporte Final
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/reporte_final_tesis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(reporte_txt))
    print("\n✅ Reporte guardado en: results/metrics/reporte_final_tesis.txt")
    
    # 6. Generar Gráficos Actualizados
    crear_graficos(df, df_q, df_s, modelos, best_models)

def crear_graficos(df, df_q, df_s, modelos, best_models, train_range="2020-2022", test_range="2023"):
    print("📊 Generando gráficos comparativos...")
    os.makedirs('results/figures', exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4)
    
    # Mensual
    ax = axes[0]
    ax.plot(df['fecha'], df['Real'], 'k-o', label='Real', linewidth=2)
    for m in modelos:
        alpha = 1.0 if m == best_models['Mensual'] else 0.4
        width = 2.0 if m == best_models['Mensual'] else 1.0
        ax.plot(df['fecha'], df[m], label=m, alpha=alpha, linewidth=width, linestyle='--')
    ax.set_title(f'Pronóstico Mensual (Mejor: {best_models["Mensual"]}) | Train: {train_range}, Test: {test_range}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Trimestral
    ax = axes[1]
    x_q = df_q['trimestre'].astype(str)
    ax.plot(x_q, df_q['Real'], 'k-o', label='Real', linewidth=2)
    for m in modelos:
        alpha = 1.0 if m == best_models['Trimestral'] else 0.4
        ax.plot(x_q, df_q[m], label=m, alpha=alpha, linestyle='--')
    ax.set_title(f'Agregado Trimestral (Mejor: {best_models["Trimestral"]}) | Train: {train_range}, Test: {test_range}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Semestral
    ax = axes[2]
    x_s = df_s['periodo_sem']
    ax.bar(x_s, df_s['Real'], alpha=0.3, color='gray', label='Real')
    for m in modelos:
        alpha = 1.0 if m == best_models['Semestral'] else 0.4
        ax.plot(x_s, df_s[m], 'D-', label=m, alpha=alpha)
    ax.set_title(f'Agregado Semestral (Mejor: {best_models["Semestral"]}) | Train: {train_range}, Test: {test_range}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('results/figures/validacion_final_optimizada.png', bbox_inches='tight')
    print("✅ Gráfico guardado: results/figures/validacion_final_optimizada.png")

if __name__ == '__main__':
    validar_modelos()
