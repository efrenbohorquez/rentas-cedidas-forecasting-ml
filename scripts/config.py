from pathlib import Path

# --- DIRECTORIES ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "BS"  # Assuming input Excel is here or root
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_FEATURES = BASE_DIR / "data" / "features"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for d in [DATA_PROCESSED, DATA_FEATURES, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- FILES ---
RAW_EXCEL_FILE = "BaseRentasCedidas (1).xlsx" # Path relative to root or absolute if needed
CLEANED_DATA_PARQUET = DATA_PROCESSED / "datos_depurados.parquet"
CLEANED_DATA_CSV = DATA_PROCESSED / "datos_depurados.csv"

TRAIN_DATA_PARQUET = DATA_FEATURES / "train_mensual.parquet"
TEST_DATA_PARQUET = DATA_FEATURES / "test_mensual.parquet"
FULL_DATA_PARQUET = DATA_FEATURES / "dataset_completo.parquet"

# --- DATES ---
TRAIN_START_YEAR = 2020
TRAIN_END_YEAR = 2025
TEST_YEAR = 2026

# --- VISUALIZATION ---
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
FIG_SIZE_DEFAULT = (12, 6)
PALETTE_DEFAULT = 'viridis'

# --- MAPPING ---
MESES_MAP = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
    'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}
