import json

nb_path = r'd:\RENTAS\notebooks\01_EDA_Depuracion.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Encontrar y corregir todas las celdas con parquet
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'parquet' in src:
        print(f'Celda {i} con parquet:')
        print(src)
        nb['cells'][i]['source'] = [
            "import os\n",
            "# Guardar copia de datos originales en CSV\n",
            "output_dir = r'D:\\RENTAS\\data\\raw'\n",
            "os.makedirs(output_dir, exist_ok=True)\n",
            "output_path = os.path.join(output_dir, 'datos_originales.csv')\n",
            "df_raw.to_csv(output_path, index=False, encoding='utf-8-sig')\n",
            "print(f'✅ Datos originales guardados en: {output_path}')\n",
            "print(f'   Registros: {len(df_raw):,} | Columnas: {df_raw.shape[1]}')\n",
            "\n",
            "print('\\n📋 SIGUIENTE PASO:')\n",
            "print('   Abrir cuaderno: 02_Feature_Engineering.ipynb')\n"
        ]
        print(f'  → Corregida\n')

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('✅ Guardado OK')
