
import pandas as pd
import config
from pathlib import Path

files = [
    config.TRAIN_DATA_MENSUAL,
    config.TRAIN_DATA_TRIMESTRAL,
    config.TRAIN_DATA_SEMESTRAL
]

for f in files:
    print(f"Checking {f.name}...")
    if f.exists():
        try:
            df = pd.read_excel(f)
            print(f"  Columns: {df.columns.tolist()}")
            if 'fecha' in df.columns:
                print("  ✅ 'fecha' column found.")
            else:
                print("  ❌ 'fecha' column MISSING.")
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
    else:
        print("  ⚠️ File does not exist.")
