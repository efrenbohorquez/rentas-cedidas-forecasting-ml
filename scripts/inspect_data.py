import pandas as pd
import sys

import os
from pathlib import Path

# Configurar salida
sys.stdout.reconfigure(encoding='utf-8')

# Importar configuración
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

try:
    from config import DATA_RAW, RAW_EXCEL_FILE
    print(f"Cargando archivo Excel desde: {DATA_RAW / RAW_EXCEL_FILE}")
    df = pd.read_excel(DATA_RAW / RAW_EXCEL_FILE, nrows=5)
    
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
