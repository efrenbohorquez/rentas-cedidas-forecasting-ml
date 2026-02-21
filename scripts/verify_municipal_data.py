import pandas as pd
import sys

import os
from pathlib import Path

# Set output encoding to utf-8 just in case
sys.stdout.reconfigure(encoding='utf-8')

# Importar configuración
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

try:
    from config import DATA_RAW, RAW_EXCEL_FILE
    print(f"Leyendo archivo Excel desde: {DATA_RAW / RAW_EXCEL_FILE}")
    # Read a chunk to be faster, but enough rows to likely find municipalities if they are mixed in
    # The user says the variable contains municipalities. Let's read 2000 rows.
    df = pd.read_excel(DATA_RAW / RAW_EXCEL_FILE, nrows=2000)
    
    print("Limpiando nombres de columnas...")
    # Clean columns as per 01_limpieza_inicial.py
    df.columns = [c.strip().replace(' ', '') for c in df.columns]
    
    target_col = 'NombreBeneficiarioAportante'
    
    if target_col not in df.columns:
        print(f"ERROR: Columna '{target_col}' no encontrada incluso después de limpieza.")
        print("Columnas disponibles:", df.columns.tolist())
    else:
        print(f"Columna '{target_col}' encontrada. Buscando municipios...")
        unique_vals = df[target_col].dropna().unique()
        
        # Look for "MUNICIPIO" (case insensitive)
        municipios = [v for v in unique_vals if 'MUNICIPIO' in str(v).upper()]
        departamentos = [v for v in unique_vals if 'DEPARTAMENTO' in str(v).upper()]
        
        print(f"Total valores únicos en muestra: {len(unique_vals)}")
        print(f"Municipios encontrados: {len(municipios)}")
        if municipios:
            print("Ejemplos de Municipios:", municipios[:5])
        else:
            print("No se encontraron municipios en las primeras 2000 filas.")
            
        print(f"Departamentos encontrados: {len(departamentos)}")
        if departamentos:
            print("Ejemplos de Departamentos:", departamentos[:5])

except Exception as e:
    print(f"Error: {e}")
