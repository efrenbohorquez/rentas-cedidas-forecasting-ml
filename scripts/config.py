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
# --- FILES ---
RAW_EXCEL_FILE = "BaseRentasCedidas (1).xlsx" # Path relative to root or absolute if needed
CLEANED_DATA_FILE = DATA_PROCESSED / "datos_depurados.xlsx" # Changed to .xlsx
CLEANED_DATA_PARQUET = CLEANED_DATA_FILE # Alias for compatibility
CLEANED_DATA_CSV = DATA_PROCESSED / "datos_depurados.csv"

TRAIN_DATA_FILE = DATA_FEATURES / "train_mensual.xlsx"
TEST_DATA_FILE = DATA_FEATURES / "test_mensual.xlsx"
FULL_DATA_FILE = DATA_FEATURES / "dataset_completo.xlsx"
FULL_HISTORIC_DATA_FILE = DATA_FEATURES / 'dataset_historico_completo.xlsx'

# Aliases for compatibility with existing scripts
TRAIN_DATA_PARQUET = TRAIN_DATA_FILE
TEST_DATA_PARQUET = TEST_DATA_FILE
FULL_DATA_PARQUET = FULL_DATA_FILE

# --- DATES ---
# --- DATES (Global Split) ---
TRAIN_START_YEAR = 2020
TRAIN_CUTOFF_DATE = '2025-09-30' # Train ends here (End of Sept)
TEST_START_DATE = '2025-10-01'   # Test starts here (Start of Oct)
TEST_END_DATE = '2025-12-31'     # Test ends here (End of Dec)

# Deprecated but kept for compatibility
TRAIN_END_YEAR = 2025
TEST_YEAR = 2025

# --- VISUALIZATION ---
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
FIG_SIZE_DEFAULT = (12, 6)
PALETTE_DEFAULT = 'viridis'

# --- MAPPING ---
MESES_MAP = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
    'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}
