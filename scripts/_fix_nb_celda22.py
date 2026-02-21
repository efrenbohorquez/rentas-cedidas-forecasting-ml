import json

nb_path = r'd:\RENTAS\notebooks\01_EDA_Depuracion.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── CELDA 22: reemplazar uso de columna_recaudo por la variable 'target' ya definida
nb['cells'][22]['source'] = [
    "# ── 9. Detección de Outliers con Isolation Forest ─────────────────────────────\n",
    "# Reutilizamos 'target' y 's' definidos en la sección 7\n",
    "recaudo_valido = df_raw[target].replace([float('inf'), float('-inf')], float('nan')).dropna()\n",
    "\n",
    "if len(recaudo_valido) > 100:\n",
    "    print('🔍 Detectando outliers con Isolation Forest...')\n",
    "\n",
    "    X = recaudo_valido.values.reshape(-1, 1)\n",
    "    iso_forest = IsolationForest(contamination=0.05, random_state=42)\n",
    "    outlier_labels = iso_forest.fit_predict(X)\n",
    "\n",
    "    n_outliers = (outlier_labels == -1).sum()\n",
    "    print(f'\\n⚠️  Outliers detectados: {n_outliers:,} ({n_outliers/len(recaudo_valido)*100:.2f}%)')\n",
    "\n",
    "    outliers = recaudo_valido[outlier_labels == -1]\n",
    "    print('\\nTop 10 outliers más extremos:')\n",
    "    print(outliers.nlargest(10).apply(lambda x: f'${x:,.2f}'))\n",
    "else:\n",
    "    print('⚠️  Datos insuficientes para Isolation Forest')\n"
]

# ── CELDA 26: reemplazar uso de columna_recaudo por 'target'
nb['cells'][26]['source'] = [
    "print('=' * 80)\n",
    "print('RESUMEN EJECUTIVO - DIAGNÓSTICO DE DATOS')\n",
    "print('=' * 80)\n",
    "\n",
    "print('\\n📊 DIMENSIONES:')\n",
    "print(f'  Registros totales: {len(df_raw):,}')\n",
    "print(f'  Columnas: {df_raw.shape[1]}')\n",
    "\n",
    "recaudo_valido = df_raw[target].replace([float('inf'), float('-inf')], float('nan')).dropna()\n",
    "\n",
    "print('\\n⚠️  PROBLEMAS DETECTADOS:')\n",
    "negativos = (recaudo_valido < 0).sum()\n",
    "if negativos > 0:\n",
    "    print(f'  ❌ Valores negativos: {negativos:,} registros')\n",
    "else:\n",
    "    print('  ✅ Sin valores negativos')\n",
    "\n",
    "extremos = (recaudo_valido > 8e8).sum()\n",
    "if extremos > 0:\n",
    "    print(f'  🚨 Outliers extremos (>$800M): {extremos:,} registros')\n",
    "\n",
    "cerca_cero = ((recaudo_valido >= 0) & (recaudo_valido < 1000)).sum()\n",
    "if cerca_cero > 10000:\n",
    "    print(f'  📉 Transacciones en escala cero (<$1,000): {cerca_cero:,} registros')\n",
    "\n",
    "print('\\n✅ ACCIONES REQUERIDAS:')\n",
    "print('  1. Auditar y corregir valores negativos')\n",
    "print('  2. Aplicar Winsorization/Capping a outliers extremos')\n",
    "print('  3. Filtrar o normalizar transacciones de bajo valor')\n",
    "print('  4. Validar integridad temporal')\n",
    "print('  5. Preparar agregaciones por horizonte temporal')\n",
    "\n",
    "print('\\n' + '=' * 80)\n"
]

# ── CELDA 28: cambiar parquet → CSV
nb['cells'][28]['source'] = [
    "import os\n",
    "# Guardar copia de datos originales en CSV (sin parquet)\n",
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

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('✅ Notebook corregido:')
print('   Celda 22 → usa "target" en lugar de "columna_recaudo"')
print('   Celda 26 → usa "target" en lugar de "columna_recaudo"')
print('   Celda 28 → exporta a CSV en lugar de parquet')
