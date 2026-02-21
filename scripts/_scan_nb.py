import json
nb_path = r'd:\RENTAS\notebooks\01_EDA_Depuracion.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'columna_recaudo' in src:
        print(f'CELDA {i}: {src[:200]}')
print('Scan completo.')
