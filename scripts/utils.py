import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import config

# --- I/O ---
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import warnings
def load_data(path, file_type='csv'):
    """Loads data from a given path with error handling."""
    path = Path(path)
    if not path.exists():
        # Try to find with other extensions if exact match fails
        if path.suffix == '.xlsx' and path.with_suffix('.csv').exists():
            path = path.with_suffix('.csv')
            file_type = 'csv'
        elif path.suffix == '.csv' and path.with_suffix('.xlsx').exists():
            path = path.with_suffix('.xlsx')
            file_type = 'excel'
        else:
             raise FileNotFoundError(f"❌ File not found: {path}")
    
    try:
        # Auto-detect based on extension if not explicit
        if path.suffix == '.xlsx':
            file_type = 'excel'
        elif path.suffix == '.csv':
            file_type = 'csv'

        if file_type == 'csv':
            df = pd.read_csv(path)
        elif file_type == 'excel':
            df = pd.read_excel(path)
        else:
            raise ValueError(f"❌ Unsupported file type: {file_type}")
        print(f"📂 Loaded {path.name}: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def save_data(df, path, save_csv=False):
    """Saves DataFrame to CSV (default) or Excel based on path suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.xlsx':
        df.to_excel(path, index=False)
        print(f"💾 Saved Excel: {path}")
    elif path.suffix == '.csv':
        df.to_csv(path, index=False)
        print(f"💾 Saved CSV: {path}")
    else:
        # Default fallback: save as CSV
        csv_path = path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved CSV: {csv_path}")
    
    # Optionally save additional CSV copy
    if save_csv and path.suffix != '.csv':
        csv_path = path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved CSV: {csv_path}")

# --- METRICS ---
def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_metrics(y_true, y_pred):
    """Returns a dictionary of metrics."""
    return {
        'MAPE': mape(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

# --- PLOTTING ---
def setup_plot_style():
    """Configures global plot styles."""
    plt.style.use(config.PLOT_STYLE)
    sns.set_palette(config.PALETTE_DEFAULT)
    plt.rcParams['figure.figsize'] = config.FIG_SIZE_DEFAULT
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def save_plot(filename, subfolder='figures'):
    """Saves the current figure to the results directory."""
    out_dir = config.RESULTS_DIR / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"🖼️ Plot saved: {path}")
    plt.close()

# --- OPTIMIZATION & VALIDATION ---

def time_series_cv(df, n_splits=5):
    """Generates train/test indices for Walk-Forward Validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(df)

def optimize_sarima(series, p_range=(0, 2), d_range=(0, 1), q_range=(0, 2), seasonal_period=12):
    """Grid Search for SARIMA parameters based on AIC."""
    print(f"⚙️ Optimizing SARIMA (S={seasonal_period})...")
    best_aic = float('inf')
    best_params = None
    best_model = None

    # Generate all combinations of p, d, q
    pdq = list(product(range(p_range[0], p_range[1]+1), 
                       range(d_range[0], d_range[1]+1), 
                       range(q_range[0], q_range[1]+1)))
    
    # Seasonal PDQ (assumed simple for speed: P=1, D=1, Q=1)
    seasonal_pdq = [(1, 1, 1, seasonal_period)]
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(series,
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (param, param_seasonal)
                    best_model = results
            except:
                continue
                
    print(f"✅ Best SARIMA: {best_params} - AIC: {best_aic:.2f}")
    return best_model, best_params

def optimize_xgboost(X, y, cv_splits=3):
    """Grid Search for XGBoost hyperparameters."""
    print("⚙️ Optimizing XGBoost...")
    
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0]
    }
    
    # Adjust n_splits if data is too small
    if len(X) < cv_splits * 2:
        cv_splits = max(2, len(X) // 3)
        print(f"⚠️ Data too small. Reducing CV splits to {cv_splits}")
        
    try:
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)
        
        grid_search = GridSearchCV(estimator=xgb_model,
                                   param_grid=param_grid,
                                   cv=tscv,
                                   scoring='neg_mean_absolute_percentage_error',
                                   verbose=1)
        
        grid_search.fit(X, y)
        print(f"✅ Best XGBoost Params: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_
    except Exception as e:
        print(f"❌ Grid Search Failed: {e}. Using default params.")
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X, y)
        return model, {'n_estimators': 100}

def walk_forward_cv(model_fn, df, n_splits=5, target_col='recaudo'):
    """
    Performs walk-forward validation.
    model_fn: function that takes (train_data, test_data) and returns predictions for test_data (array-like).
    df: pandas DataFrame containing the data.
    target_col: name of the target column.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    print(f"🔄 Starting Walk-Forward CV ({n_splits} splits)...")
    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        
        try:
            y_pred = model_fn(train, test)
            score = mape(test[target_col], y_pred)
            scores.append(score)
            print(f"   🔹 Split {i+1}: MAPE = {score:.2f}%")
        except Exception as e:
            print(f"   ❌ Split {i+1} Error: {e}")
            
    mean_score = np.mean(scores) if scores else np.inf
    print(f"✅ Average CV MAPE: {mean_score:.2f}%")
    return mean_score
