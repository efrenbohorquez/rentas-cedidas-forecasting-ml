import pandas as pd
import os

ruta_excel = 'BaseRentasCedidas (1).xlsx'
if not os.path.exists(ruta_excel):
    print(f"❌ Archivo no encontrado: {ruta_excel}")
else:
    print(f"📂 Leyendo archivo: {ruta_excel}...")
    try:
        df = pd.read_excel(ruta_excel)
        print("Columnas:", df.columns.tolist())
        
        # Limpiar columnas
        df.columns = [c.strip().replace(' ', '') for c in df.columns]
        
        if 'Vigencia' in df.columns:
            print("\n📅 Conteo por Vigencia:")
            print(df['Vigencia'].value_counts().sort_index())
        else:
            print("❌ Columna 'Vigencia' no encontrada")
            
        if 'MesNombreCalendario' in df.columns:
            print("\n🗓️ Meses únicos encontrados:")
            print(df['MesNombreCalendario'].unique())
            
            # Chequear nulos en mapeo
            mapa_meses = {
                'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
                'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
            }
            df['mes_check'] = df['MesNombreCalendario'].map(mapa_meses)
            nulos = df[df['mes_check'].isna()]['MesNombreCalendario'].unique()
            if len(nulos) > 0:
                print(f"\n⚠️ MESES QUE FALLARÍAN EL MAPEO: {nulos}")
        else:
            print("❌ Columna 'MesNombreCalendario' no encontrada")

    except Exception as e:
        print(f"❌ Error leyendo Excel: {e}")
