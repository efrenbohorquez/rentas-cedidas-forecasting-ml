import pandas as pd
import numpy as np
import os

def limpiar_datos():
    print("🚀 Iniciando limpieza de datos (Versión Municipal + Neteo)...")
    
    # 1. Cargar datos
    ruta_excel = 'BaseRentasCedidas (1).xlsx'
    if not os.path.exists(ruta_excel):
        print(f"❌ Archivo no encontrado: {ruta_excel}")
        return
        
    print(f"📂 Leyendo archivo: {ruta_excel}...")
    try:
        df = pd.read_excel(ruta_excel)
    except Exception as e:
        print(f"❌ Error leyendo Excel: {e}")
        return

    # 2. Limpieza de columnas
    print("🧹 Limpiando nombres de columnas...")
    # Handle normal spaces and EM SPACE (\u2003)
    df.columns = [c.strip().replace(' ', '').replace(u'\u2003', '') for c in df.columns]
    
    cols_necesarias = {
        'Vigencia': 'vigencia',
        'MesNombreCalendario': 'mes_nombre',
        'ValorRecaudo': 'recaudo',
        'NombreBeneficiarioAportante': 'entidad',
        'Nombre_SubGrupo_Aportante': 'fuente',
        'FechaRecaudo': 'fecha_transaccion'
    }
    
    missing = [c for c in cols_necesarias.keys() if c not in df.columns]
    if missing:
        print(f"❌ Faltan columnas clave: {missing}")
        print(f"   Columnas encontradas: {df.columns.tolist()}")
        return

    df = df[list(cols_necesarias.keys())].rename(columns=cols_necesarias)
    
    # 3. Conversiones Básicas
    df['recaudo'] = pd.to_numeric(df['recaudo'], errors='coerce').fillna(0)
    
    # 4. Estandarización de Fechas (Vigencia-Mes)
    # Mapeo robusto (Case Insensitive, Strip)
    mapa_meses = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
        'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    
    # Normalizar mes_nombre
    df['mes_nombre_norm'] = df['mes_nombre'].astype(str).str.upper().str.strip()
    df['mes_num'] = df['mes_nombre_norm'].map(mapa_meses)
    
    # Reportar meses no mapeados
    unmapped = df[df['mes_num'].isna()]['mes_nombre_norm'].unique()
    if len(unmapped) > 0:
        print(f"⚠️ MESES NO RECONOCIDOS (Serán eliminados): {unmapped}")
    
    df = df.dropna(subset=['mes_num'])
    df['mes_num'] = df['mes_num'].astype(int)
    
    # Crear columna Fecha estandarizada (Día 1 del mes)
    df['fecha'] = pd.to_datetime(dict(year=df['vigencia'], month=df['mes_num'], day=1))
    
    # 5. Identificación de Nivel (Municipal vs Departamental)
    print("🔍 Clasificando entidades (Municipal vs Departamental)...")
    def clasificar_entidad(nombre):
        nombre = str(nombre).upper()
        if any(x in nombre for x in ['MUNICIPIO', 'ALCALDÍA', 'DISTRITO', 'ALCALDIA']):
            return 'Municipal'
        elif any(x in nombre for x in ['DEPARTAMENTO', 'GOBERNACIÓN', 'GOBERNACION']):
            return 'Departamental'
        else:
            return 'Otro/Nacional'

    df['tipo_entidad'] = df['entidad'].apply(clasificar_entidad)
    print(f"   Distribución por tipo:\n{df['tipo_entidad'].value_counts()}")

    # 6. Neteo Mensual (Manejo de Negativos)
    # En lugar de eliminar negativos, agrupamos por Fecha + Entidad + Fuente
    # Esto suma positivos y negativos del mismo periodo.
    print("➕➖ Realizando neteo mensual de transacciones...")
    
    grupos = ['fecha', 'vigencia', 'mes_num', 'entidad', 'tipo_entidad', 'fuente']
    df_neto = df.groupby(grupos)['recaudo'].sum().reset_index()
    
    # Verificar si quedaron netos negativos (reales devoluciones netas)
    n_negativos = (df_neto['recaudo'] < 0).sum()
    if n_negativos > 0:
        print(f"⚠️ Se encontraron {n_negativos} registros con recaudo neto negativo. Se ajustarán a 0 o mantendrán según lógica de negocio.")
        # Opcional: Ajustar a 0 o dejarlos. Para series de tiempo, 0 es mejor que negativo si es ingreso.
        # Asumiremos 0 para evitar problemas con logaritmos o modelos que asumen > 0.
        df_neto.loc[df_neto['recaudo'] < 0, 'recaudo'] = 0
        
    # Validar integridad
    print(f"   Registros originales: {len(df)}")
    print(f"   Registros neteados: {len(df_neto)}")
    
    # Asegurar tipos para Parquet
    df_neto['entidad'] = df_neto['entidad'].astype(str)
    df_neto['fuente'] = df_neto['fuente'].astype(str)
    df_neto['tipo_entidad'] = df_neto['tipo_entidad'].astype(str)
    
    # 7. Guardar
    os.makedirs('data/processed', exist_ok=True)
    ruta_salida = 'data/processed/datos_depurados.parquet'
    df_neto.to_parquet(ruta_salida, index=False)
    
    print(f"✅ Datos depurados guardados en: {ruta_salida}")
    print(df_neto.head())
    print(df_neto.info())

if __name__ == '__main__':
    limpiar_datos()
