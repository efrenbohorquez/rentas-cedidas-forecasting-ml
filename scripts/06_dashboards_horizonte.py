import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
os.makedirs('results/figures/dashboards', exist_ok=True)

def calcular_metricas(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0: return {'MAPE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    sst = np.sum((y_true - np.mean(y_true))**2)
    ssr = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ssr/sst) if sst != 0 else np.nan
    return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def generar_dashboard_individual(modelo, df_pred, df_hist, horizonte_name, filename, train_range, test_range):
    """Genera un dashboard específico para UN modelo y UN horizonte."""
    
    # Métricas
    met = calcular_metricas(df_pred['Real'], df_pred[modelo])
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.15, 1, 0.5], wspace=0.3, hspace=0.4)
    
    # --- TÍTULO ---
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.7, f'📊 Validación Modelo: {modelo} - {horizonte_name}', 
                  fontsize=24, fontweight='bold', ha='center', va='center', color='#2c3e50')
    ax_title.text(0.5, 0.3, f'Entrenamiento: {train_range}  |  Prueba: {test_range}', 
                  fontsize=14, ha='center', va='center', color='#7f8c8d')

    # --- TABLA MÉTRICAS (Izquierda) ---
    ax_table = fig.add_subplot(gs[1, 0])
    ax_table.axis('off')
    # Definir colores según MAPE
    color_score = '#2ecc71' if met['MAPE'] < 15 else '#f1c40f' if met['MAPE'] < 30 else '#e74c3c'
    
    cell_text = [
        [f"{met['MAPE']:.2f}%"],
        [f"{met['RMSE']/1e9:.2f} MM"],
        [f"{met['MAE']/1e9:.2f} MM"],
        [f"{met['R2']:.3f}"]
    ]
    labels = ['MAPE', 'RMSE (Billions)', 'MAE (Billions)', 'R²']
    
    table = ax_table.table(cellText=cell_text, rowLabels=labels, loc='center', cellLoc='center', colWidths=[0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    # Score Card Visual
    ax_table.text(0.5, 0.85, 'Performance', fontsize=16, fontweight='bold', ha='center', transform=ax_table.transAxes)
    circle = plt.Circle((0.5, 0.75), 0.05, color=color_score, transform=ax_table.transAxes)
    ax_table.add_artist(circle)

    # --- SERIE DE TIEMPO (Derecha - Principal) ---
    ax_ts = fig.add_subplot(gs[1, 1:])
    
    # 1. Historia
    if df_hist is not None:
        if horizonte_name == 'Mensual':
            ax_ts.plot(df_hist['fecha'], df_hist['Real']/1e9, color='#95a5a6', label='Histórico (2020-2024)', linewidth=1.5, alpha=0.7)
        else:
            # Agregada
            ax_ts.plot(df_hist['periodo'], df_hist['Real']/1e9, color='#95a5a6', label='Histórico Agregado', linewidth=1.5, marker='.')
            
    # 2. Real vs Pred (Test)
    if horizonte_name == 'Mensual':
        x_test = df_pred['fecha']
        marker = 'o'
    elif 'trimestre' in df_pred.columns:
        x_test = df_pred['trimestre'].astype(str)
        marker = 's'
    else:
        x_test = df_pred['periodo_sem']
        marker = 'D'

    ax_ts.plot(x_test, df_pred['Real']/1e9, color='black', label='Real (Test)', linewidth=2.5, marker=marker)
    ax_ts.plot(x_test, df_pred[modelo]/1e9, color='#3498db', label=f'Predicción {modelo}', linewidth=2.5, linestyle='--', marker=marker)
    
    ax_ts.set_title('Serie Temporal Completa (Historia + Predicción)', fontsize=16, fontweight='bold')
    ax_ts.set_ylabel('Recaudo (Miles de Millones COP)')
    ax_ts.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax_ts.grid(True, alpha=0.3)
    
    # Rotar etiquetas si Son muchas
    if len(x_test) > 10 or (df_hist is not None and len(df_hist) > 20):
        plt.setp(ax_ts.get_xticklabels(), rotation=45, ha='right')

    # --- ERRORES (Abajo) ---
    ax_err = fig.add_subplot(gs[2, :])
    errores = df_pred['Real'] - df_pred[modelo]
    errores_pct = (errores / df_pred['Real']) * 100
    
    colores_err = ['#e74c3c' if abs(e) > 15 else '#2ecc71' for e in errores_pct]
    
    bars = ax_err.bar(x_test, errores_pct, color=colores_err, edgecolor='black', alpha=0.7)
    ax_err.axhline(0, color='black', linewidth=1)
    ax_err.axhline(15, color='gray', linestyle='--', alpha=0.5)
    ax_err.axhline(-15, color='gray', linestyle='--', alpha=0.5)
    
    # Etiquetas de valor
    for bar in bars:
        height = bar.get_height()
        ax_err.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
    ax_err.set_title('Porcentaje de Error por Periodo', fontsize=14, fontweight='bold')
    ax_err.set_ylabel('Error (%)')
    plt.setp(ax_err.get_xticklabels(), rotation=45, ha='right')

    plt.savefig(f'results/figures/dashboards/{filename}', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Generado: {filename}")

def main():
    print("🎨 Generando Tableros Individuales (Por Modelo/Horizonte)...")
    
    # Cargar Predicciones
    df_base = pd.read_csv('results/predictions/predicciones_comparativas.csv')
    df_base['fecha'] = pd.to_datetime(df_base['fecha'])
    
    # Cargar Historia (Contexto 2020+)
    try:
        df_hist_full = pd.read_csv('data/features/dataset_historico_completo.csv')
        df_hist_full['fecha'] = pd.to_datetime(df_hist_full['fecha'])
        df_hist_full = df_hist_full[['fecha', 'recaudo']].rename(columns={'recaudo': 'Real'})
        
        # Calcular Rangos
        train_min = df_hist_full['fecha'].min().strftime('%Y-%m')
        # Asumiendo split en feature engineering
        test_min = df_base['fecha'].min()
        train_max = (test_min - pd.DateOffset(months=1)).strftime('%Y-%m')
        test_max = df_base['fecha'].max().strftime('%Y-%m')
        
        train_range = f"{train_min} a {train_max}"
        test_range = f"{test_min.strftime('%Y-%m')} a {test_max}"
        
        # Separar solo historia pura (hasta inicio de test)
        df_hist_plot = df_hist_full[df_hist_full['fecha'] < test_min].copy()
        
    except Exception as e:
        print(f"⚠️ Error cargando historia: {e}")
        df_hist_plot = None
        train_range = "N/A"
        test_range = "N/A"

    modelos = [c for c in df_base.columns if c not in ['fecha', 'Real']]
    
    # 1. HORIZONTE MENSUAL
    print("\n📅 Procesando Horizonte MENSUAL...")
    for m in modelos:
        generar_dashboard_individual(m, df_base, df_hist_plot, 'Mensual', 
                                     f'dashboard_mensual_{m}.png', train_range, test_range)

    # 2. HORIZONTE TRIMESTRAL
    print("\n📆 Procesando Horizonte TRIMESTRAL...")
    df_q = df_base.copy()
    df_q['trimestre'] = df_q['fecha'].dt.to_period('Q')
    df_q_agg = df_q.groupby('trimestre')[['Real'] + modelos].sum().reset_index()
    
    # Agregar historia trimestral
    if df_hist_plot is not None:
        hist_q = df_hist_plot.copy()
        hist_q['periodo'] = hist_q['fecha'].dt.to_period('Q').astype(str)
        hist_q_agg = hist_q.groupby('periodo')['Real'].sum().reset_index()
    else:
        hist_q_agg = None
        
    for m in modelos:
        generar_dashboard_individual(m, df_q_agg, hist_q_agg, 'Trimestral', 
                                     f'dashboard_trimestral_{m}.png', train_range, test_range)

    # 3. HORIZONTE SEMESTRAL
    print("\n📆 Procesando Horizonte SEMESTRAL...")
    df_s = df_base.copy()
    df_s['semestre'] = np.where(df_s['fecha'].dt.month <= 6, 1, 2)
    df_s['periodo_sem'] = df_s['fecha'].dt.year.astype(str) + '-S' + df_s['semestre'].astype(str)
    df_s_agg = df_s.groupby('periodo_sem')[['Real'] + modelos].sum().reset_index()
    
     # Agregar historia semestral
    if df_hist_plot is not None:
        hist_s = df_hist_plot.copy()
        hist_s['semestre'] = np.where(hist_s['fecha'].dt.month <= 6, 1, 2)
        hist_s['periodo'] = hist_s['fecha'].dt.year.astype(str) + '-S' + hist_s['semestre'].astype(str)
        hist_s_agg = hist_s.groupby('periodo')['Real'].sum().reset_index()
    else:
        hist_s_agg = None

    for m in modelos:
        generar_dashboard_individual(m, df_s_agg, hist_s_agg, 'Semestral', 
                                     f'dashboard_semestral_{m}.png', train_range, test_range)
    
    print("\n✅ Todos los tableros generados en results/figures/dashboards/")

if __name__ == '__main__':
    main()
