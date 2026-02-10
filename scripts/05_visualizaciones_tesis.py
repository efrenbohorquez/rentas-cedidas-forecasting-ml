import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
os.makedirs('results/figures', exist_ok=True)

def generar_visualizaciones():
    print("🎨 Generando visualizaciones para Tesis...")
    
    # Cargar datos
    pred_df = pd.read_csv('results/predictions/predicciones_comparativas.csv')
    pred_df['fecha'] = pd.to_datetime(pred_df['fecha'])
    
    train = pd.read_parquet('data/features/train_mensual.parquet')
    full_df = pd.read_parquet('data/features/dataset_completo.parquet')
    
    modelos = [c for c in pred_df.columns if c not in ['fecha', 'Real']]
    
    # ============================================================
    # FIGURA 1: Serie Temporal Histórica con Tendencia
    # ============================================================
    print("📈 1. Serie Temporal Histórica...")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(full_df['fecha'], full_df['recaudo'], color='steelblue', linewidth=1.5, label='Recaudo Mensual')
    
    # Añadir media móvil
    full_df['MA12'] = full_df['recaudo'].rolling(12).mean()
    ax.plot(full_df['fecha'], full_df['MA12'], color='darkred', linewidth=2, linestyle='--', label='Media Móvil 12M')
    
    ax.axvline(pd.to_datetime('2024-01-01'), color='gray', linestyle=':', linewidth=2, label='Inicio Periodo Test')
    ax.set_title('Serie Temporal de Recaudo (2020-2024)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Recaudo (COP)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/01_serie_temporal_historica.png')
    plt.close()
    
    # ============================================================
    # FIGURA 2: Estacionalidad Mensual (Boxplot)
    # ============================================================
    print("📊 2. Estacionalidad Mensual...")
    fig, ax = plt.subplots(figsize=(12, 6))
    full_df['mes_nombre'] = full_df['fecha'].dt.month_name()
    
    # Orden de meses
    orden_meses = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    sns.boxplot(data=full_df, x='mes_nombre', y='recaudo', order=orden_meses, palette='coolwarm', ax=ax)
    ax.set_title('Distribución de Recaudo por Mes (Estacionalidad)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Recaudo (COP)')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('results/figures/02_estacionalidad_mensual.png')
    plt.close()
    
    # ============================================================
    # FIGURA 3: Análisis de Residuos (Mejor Modelo: SARIMAX)
    # ============================================================
    print("🔍 3. Análisis de Residuos...")
    if 'SARIMAX' in pred_df.columns:
        residuos = pred_df['Real'] - pred_df['SARIMAX']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuos vs Tiempo
        axes[0, 0].plot(pred_df['fecha'], residuos, marker='o', color='steelblue')
        axes[0, 0].axhline(0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuos vs Tiempo')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Error (COP)')
        
        # Histograma de Residuos
        axes[0, 1].hist(residuos, bins=10, edgecolor='black', color='steelblue', alpha=0.7)
        axes[0, 1].set_title('Distribución de Residuos')
        axes[0, 1].set_xlabel('Error')
        axes[0, 1].set_ylabel('Frecuencia')
        
        # Q-Q Plot
        stats.probplot(residuos, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normalidad)')
        
        # Residuos vs Predicho
        axes[1, 1].scatter(pred_df['SARIMAX'], residuos, color='steelblue', alpha=0.7)
        axes[1, 1].axhline(0, color='red', linestyle='--')
        axes[1, 1].set_title('Residuos vs Valor Predicho')
        axes[1, 1].set_xlabel('Predicción')
        axes[1, 1].set_ylabel('Error')
        
        plt.suptitle('Diagnóstico de Residuos - SARIMAX', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/figures/03_analisis_residuos_sarimax.png')
        plt.close()
    
    # ============================================================
    # FIGURA 4: Comparación de Modelos (Barras)
    # ============================================================
    print("📊 4. Ranking de Modelos...")
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    mapes = {m: mape(pred_df['Real'], pred_df[m]) for m in modelos}
    mapes_df = pd.DataFrame(list(mapes.items()), columns=['Modelo', 'MAPE']).sort_values('MAPE')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = ['#2ecc71' if m == 'SARIMAX' else '#3498db' for m in mapes_df['Modelo']]
    bars = ax.barh(mapes_df['Modelo'], mapes_df['MAPE'], color=colores, edgecolor='black')
    
    for bar, val in zip(bars, mapes_df['MAPE']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontweight='bold')
    
    ax.axvline(12, color='red', linestyle='--', linewidth=2, label='Umbral Aceptación (12%)')
    ax.set_title('Ranking de Modelos por MAPE (Horizonte Mensual)', fontsize=14, fontweight='bold')
    ax.set_xlabel('MAPE (%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/figures/04_ranking_modelos.png')
    plt.close()
    
    # ============================================================
    # FIGURA 5: Predicciones vs Real (Detalle)
    # ============================================================
    print("📈 5. Predicciones vs Real...")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(pred_df['fecha'], pred_df['Real']*0.88, pred_df['Real']*1.12, 
                    alpha=0.2, color='gray', label='Banda ±12%')
    ax.plot(pred_df['fecha'], pred_df['Real'], marker='o', markersize=10, 
            color='black', linewidth=2, label='Real')
    
    colores_modelos = {'SARIMAX': '#2ecc71', 'XGBoost': '#e74c3c', 'LSTM': '#9b59b6', 'Prophet': '#3498db'}
    for m in modelos:
        ax.plot(pred_df['fecha'], pred_df[m], marker='s', markersize=6, 
                linestyle='--', color=colores_modelos.get(m, 'gray'), label=m)
    
    # Calcular rango de test
    test_year = pred_df['fecha'].dt.year.mode()[0]
    
    ax.set_title(f'Predicciones vs Valores Reales ({test_year})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Recaudo (COP)')
    ax.legend(loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('results/figures/05_predicciones_vs_real.png')
    plt.close()
    
    # ============================================================
    # FIGURA 6: Heatmap de Errores Mensuales
    # ============================================================
    print("🔥 6. Heatmap de Errores...")
    errores = pred_df.copy()
    for m in modelos:
        errores[f'{m}_error'] = ((errores['Real'] - errores[m]) / errores['Real'] * 100).abs()
    
    errores['mes'] = errores['fecha'].dt.month_name()
    
    error_cols = [f'{m}_error' for m in modelos]
    heatmap_data = errores[['mes'] + error_cols].set_index('mes')
    heatmap_data.columns = modelos
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Error Absoluto (%)'})
    ax.set_title('Heatmap de Errores Porcentuales por Mes y Modelo', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mes')
    ax.set_xlabel('Modelo')
    plt.tight_layout()
    plt.savefig('results/figures/06_heatmap_errores.png')
    plt.close()
    
    # ============================================================
    # FIGURA 7: Importancia de Features (XGBoost)
    # ============================================================
    print("🌳 7. Importancia de Features...")
    try:
        import xgboost as xgb
        model_xgb = xgb.XGBRegressor()
        model_xgb.load_model('models/xgboost_model.json')
        
        features = [c for c in train.columns if c not in ['fecha', 'recaudo']]
        importances = model_xgb.feature_importances_
        
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='forestgreen', edgecolor='black')
        ax.set_title('Top 10 Features más Importantes (XGBoost)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importancia')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig('results/figures/07_feature_importance_xgboost.png')
        plt.close()
    except Exception as e:
        print(f"⚠️ No se pudo generar importancia de features: {e}")
    
    print("\n✅ Visualizaciones generadas en results/figures/")
    print("="*50)
    for f in os.listdir('results/figures'):
        print(f"  📊 {f}")

if __name__ == '__main__':
    generar_visualizaciones()
