import pandas as pd
import sys

# Configurar salida
sys.stdout.reconfigure(encoding='utf-8')

try:
    print("Cargando archivo Excel...")
    df = pd.read_excel('BaseRentasCedidas (1).xlsx', nrows=5)
    
    with open('data_inspection.txt', 'w', encoding='utf-8') as f:
        f.write("=== COLUMNAS ===\n")
        f.write("\n".join([f"'{c}'" for c in df.columns]))
        f.write("\n\n=== HEAD (5 filas) ===\n")
        f.write(df.to_string())
        f.write("\n\n=== TIPOS DE DATOS ===\n")
        f.write(str(df.dtypes))

    print("✅ Inspección guardada en data_inspection.txt")

except Exception as e:
    print(f"❌ Error: {str(e)}")
