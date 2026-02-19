
import pandas as pd
import config

try:
    file_path = config.DATA_PROCESSED / "resultados_modelos.xlsx"
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
    else:
        df = pd.read_excel(file_path)
        bimestral_results = df[df['Horizonte'] == 'Bimestral']
        
        if bimestral_results.empty:
            print("⚠️ No results found for 'Bimestral' horizon.")
        else:
            print("\n📊 Resultados del Horizonte Bimestral:")
            print(bimestral_results[['Modelo', 'MAPE']].to_markdown(index=False))
            
            best_model = bimestral_results.loc[bimestral_results['MAPE'].idxmin()]
            print(f"\n🏆 Mejor Modelo Bimestral: {best_model['Modelo']} (MAPE: {best_model['MAPE']:.2%})")

except Exception as e:
    print(f"❌ Error: {e}")
