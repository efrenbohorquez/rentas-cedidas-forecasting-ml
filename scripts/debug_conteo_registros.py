import pandas as pd
import os

def auditar_registros():
    print("🔍 Auditoría de Registros y Entidades")
    
    ruta = 'data/processed/datos_depurados.parquet'
    if not os.path.exists(ruta):
        print("❌ No se encontró datos_depurados.parquet")
        return

    df = pd.read_parquet(ruta)
    with open('debug_audit.txt', 'w') as f:
        f.write(f"1. Total registros detallados (transacciones): {len(df)}\n")
        
        f.write("\n2. Distribución por Tipo de Entidad:\n")
        f.write(str(df['tipo_entidad'].value_counts()) + "\n")
        
        f.write("\n3. Distribución por Vigencia (Años):\n")
        f.write(str(df['vigencia'].value_counts().sort_index()) + "\n")
        
        # Simular la agregación que hacemos en feature engineering
        df_mun = df[df['tipo_entidad'] == 'Municipal']
        f.write(f"\n4. Registros si filtramos solo MUNICIPAL: {len(df_mun)}\n")
        
        df_agg_mun = df_mun.groupby('fecha')['recaudo'].sum().reset_index()
        f.write(f"5. Meses resultantes (Agregado Municipal): {len(df_agg_mun)}\n")
        
        df_agg_total = df.groupby('fecha')['recaudo'].sum().reset_index()
        f.write(f"6. Meses resultantes (Agregado TOTAL sin filtrar): {len(df_agg_total)}\n")
        
    print("Reporte guardado en debug_audit.txt")

if __name__ == "__main__":
    auditar_registros()
