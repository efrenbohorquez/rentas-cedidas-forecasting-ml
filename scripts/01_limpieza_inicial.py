import pandas as pd
import numpy as np
import config
import utils

def limpiar_datos():
    print("🚀 Iniciando limpieza de datos (Refactorizado)...")
    
    # 1. Cargar datos
    ruta_excel = config.BASE_DIR / config.RAW_EXCEL_FILE
    
    print(f"📂 Buscando archivo en: {ruta_excel}...")
    df = utils.load_data(ruta_excel, file_type='excel')
    
    if df is None:
        print("❌ No se pudo cargar el archivo Excel.")
        return

    # 2. Limpieza de columnas
    print("🧹 Limpiando nombres de columnas...")
    import re
    # Remove non-alphanumeric (except underscores) and trim
    new_cols = [re.sub(r'[^\w\s]', '', c).strip().replace(' ', '').replace(' ', '') for c in df.columns]
    # Fallback/Additional cleanup
    new_cols = [c.replace(' ', '').replace(u'\u2003', '').strip() for c in new_cols]
    
    df.columns = new_cols
    
    # Handle duplicates if any
    if len(df.columns) != len(set(df.columns)):
        print("⚠️ Advertencia: Columnas duplicadas detectadas tras limpieza. Renombrando...")
        df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)
    
    print(f"   Columnas limpias (primeras 10): {df.columns.tolist()[:10]}")

    # Adapt to available columns
    map_candidates = {
        'ValorRecaudo': 'recaudo',
        'NombreBeneficiarioAportante': 'entidad',
        'Nombre_SubGrupo_Aportante': 'fuente',
        'FechaRecaudo': 'fecha_transaccion',
        'Vigencia': 'vigencia',
        'MesNombreCalendario': 'mes_nombre'
    }
    
    # Filter mapping
    cols_to_rename = {k: v for k, v in map_candidates.items() if k in df.columns}
    print(f"   Renombrando: {cols_to_rename}")
    df = df.rename(columns=cols_to_rename)
    
    # Check critical columns
    if 'recaudo' not in df.columns or 'fecha_transaccion' not in df.columns:
        print(f"❌ Faltan columnas CRÍTICAS (recaudo o fecha_transaccion).")
        print(f"   Disponibles: {df.columns.tolist()}")
        # Imprimir candidatos cercanos para debug
        print(f"   Posibles candidatos 'recaudo': {[c for c in df.columns if 'recaudo' in c.lower()]}")
        print(f"   Posibles candidatos 'fecha': {[c for c in df.columns if 'fecha' in c.lower()]}")
        return

    # Derive Vigencia/Mes if missing
    if 'vigencia' not in df.columns or 'mes_nombre' not in df.columns:
        print("⚠️ 'Vigencia' o 'Mes' no encontrados. Derivando de 'fecha_transaccion'...")
        try:
            df['fecha_dt'] = pd.to_datetime(df['fecha_transaccion'], errors='coerce')
        except Exception as e:
            print(f"❌ Error convirtiendo fecha: {e}")
            return
            
        if 'vigencia' not in df.columns:
            df['vigencia'] = df['fecha_dt'].dt.year
            
        if 'mes_nombre' not in df.columns:
            # Create Spanish month names map
            meses_es = {1: 'ENERO', 2: 'FEBRERO', 3: 'MARZO', 4: 'ABRIL', 5: 'MAYO', 6: 'JUNIO',
                        7: 'JULIO', 8: 'AGOSTO', 9: 'SEPTIEMBRE', 10: 'OCTUBRE', 11: 'NOVIEMBRE', 12: 'DICIEMBRE'}
            df['mes_nombre'] = df['fecha_dt'].dt.month.map(meses_es)
            
    # Keep only necessary columns
    cols_finales = ['vigencia', 'mes_nombre', 'recaudo', 'entidad', 'fuente', 'fecha_transaccion']
    
    # Check if all exist
    missing_final = [c for c in cols_finales if c not in df.columns]
    if missing_final:
        print(f"❌ Error: Faltan columnas finales derivadas: {missing_final}")
        return
        
    df = df[cols_finales]
    
    # 3. Conversiones Básicas
    print("🔄 Convirtiendo tipos...")
    try:
        df['recaudo'] = pd.to_numeric(df['recaudo'], errors='coerce').fillna(0)
    except Exception as e:
        print(f"❌ Error convirtiendo recaudo: {e}")
        return
    
    # 4. Estandarización de Fechas (Vigencia-Mes)
    print("📅 Estandarizando fechas...")
    df['mes_nombre_norm'] = df['mes_nombre'].astype(str).str.upper().str.strip()
    df['mes_num'] = df['mes_nombre_norm'].map(config.MESES_MAP)
    
    # Reportar meses no mapeados y limpiar
    # Drop rows with invalid dates/months early
    df = df.dropna(subset=['mes_num', 'vigencia'])
    df['mes_num'] = df['mes_num'].astype(int)
    df['vigencia'] = df['vigencia'].astype(int)
    
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
    print("➕➖ Realizando neteo mensual de transacciones...")
    
    grupos = ['fecha', 'vigencia', 'mes_num', 'entidad', 'tipo_entidad', 'fuente']
    df_neto = df.groupby(grupos)['recaudo'].sum().reset_index()
    
    # Verificar si quedaron netos negativos
    n_negativos = (df_neto['recaudo'] < 0).sum()
    if n_negativos > 0:
        print(f"⚠️ Se encontraron {n_negativos} registros con recaudo neto negativo. Ajustando a 0.")
        df_neto.loc[df_neto['recaudo'] < 0, 'recaudo'] = 0
    
    # --- Interpolación de Valores Faltantes ---
    # 📌 TÉCNICA DE MEJORA (NotebookLM - Video: Limpieza de Datos en Series Temporales):
    # En lugar de eliminar filas con datos faltantes o llenar con ceros (lo cual sesga el modelo),
    # utilizamos interpolación lineal temporal. Esto preserva la tendencia local y la continuidad
    # de la serie temporal, vital para modelos como LSTM que dependen de secuencias continuas.
    
    # --- Interpolación de Valores Faltantes ---
    # 📌 TÉCNICA DE MEJORA (NotebookLM - Video: Limpieza de Datos en Series Temporales):
    # En lugar de eliminar filas con datos faltantes o llenar con ceros (lo cual sesga el modelo),
    # utilizamos interpolación lineal temporal. Esto preserva la tendencia local y la continuidad
    # de la serie temporal, vital para modelos como LSTM que dependen de secuencias continuas.
    
    print("✨ Aplicando interpolación temporal (Referencia: NotebookLM - Manejo de Missing Values)...")
    
    # Asegurar ordenamiento por fecha para que la interpolación temporal tenga sentido
    df_neto = df_neto.sort_values(by=['entidad', 'fuente', 'fecha'])

    # 1. Reemplazar 0s con NaN para permitir interpolación correcta (asumiendo que 0 es 'dato faltante' en este contexto)
    # NOTA: Si 0 es un valor válido de recaudo real, esto debería ajustarse. Asumimos aquí que 0 = sin registro.
    df_neto['recaudo'] = df_neto['recaudo'].replace(0, np.nan)
    
    # 2. Interpolar linealmente dentro de cada grupo (Entidad, Fuente)
    # Función robusta para aplicar interpolación
    def interpolate_group(group):
        # Si todo es NaN, retornar 0s
        if group.isna().all():
            return group.fillna(0)
        # Si hay suficientes datos, interpolar
        try:
            return group.interpolate(method='linear').fillna(0) # fillna(0) para bordes iniciales/finales
        except Exception:
            return group.fillna(0) # Fallback

    try:
        # Usamos transform para mantener el índice original alineado
        print("   ⏳ Ejecutando interpolación por grupos (esto puede tardar unos segundos)...")
        df_neto['recaudo'] = df_neto.groupby(['entidad', 'fuente'])['recaudo'].transform(interpolate_group)
        print("   ✅ Interpolación completada.")
    except Exception as e:
        print(f"❌ Error crítico en interpolación, volviendo a llenar con 0: {e}")
        df_neto['recaudo'] = df_neto['recaudo'].fillna(0)
    
    # Restaurar NaNs restantes a 0 (por si acaso)
    df_neto['recaudo'] = df_neto['recaudo'].fillna(0)

    print(f"   Registros originales: {len(df)}")
    print(f"   Registros neteados e interpolados: {len(df_neto)}")
    
    # Asegurar tipos para Excel
    df_neto['entidad'] = df_neto['entidad'].astype(str)
    df_neto['fuente'] = df_neto['fuente'].astype(str)
    df_neto['tipo_entidad'] = df_neto['tipo_entidad'].astype(str)
    
    # Convertir timestamps a date para Excel limpio
    try:
        df_neto['fecha'] = pd.to_datetime(df_neto['fecha']).dt.date
    except AttributeError:
        pass
    
    # 7. Guardar
    utils.save_data(df_neto, config.CLEANED_DATA_FILE)
    
    print(f"✅ Limpieza completada (Excel generado en {config.CLEANED_DATA_FILE}).")

if __name__ == '__main__':
    limpiar_datos()
