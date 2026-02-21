import os, json

PROYECTO = r'd:\RENTAS'
IGNORE = {'.git', '__pycache__', '.venv', 'node_modules', '.gemini'}
MALOS = [
    'BaseRentasCedidas (1)',
    'BaseRentasCedidas.xlsx',
    r'Music\TABLERO',
    '../data/BaseRentas',
    './data/BaseRentas',
]

problemas = []
revisados = 0

SELF = os.path.abspath(__file__)

for root, dirs, files in os.walk(PROYECTO):
    dirs[:] = [d for d in dirs if d not in IGNORE]
    for fname in files:
        ext = fname.lower().split('.')[-1]
        if ext not in ('py', 'ipynb', 'md', 'txt', 'json', 'yaml', 'yml'):
            continue
        fpath = os.path.join(root, fname)
        if os.path.abspath(fpath) == SELF:
            continue
        revisados += 1
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for malo in MALOS:
                if malo in content:
                    for i, line in enumerate(content.split('\n'), 1):
                        if malo in line:
                            rel = os.path.relpath(fpath, PROYECTO)
                            problemas.append(f'{rel}:{i}: {line.strip()[:80]}')
                    break
        except:
            pass

print(f'Archivos revisados: {revisados}')
if problemas:
    print(f'PENDIENTES ({len(problemas)}):')
    for p in problemas:
        print(f'  {p}')
else:
    print('RESULTADO: LIMPIO - Sin referencias antiguas en ningun archivo')
    print('Ruta activa: D:\\RENTAS\\data\\BaseRentasVF_limpieza21feb.xlsx')
