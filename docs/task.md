# Audit Codebase for "Rentas Cedidas" Suitability

- [x] Analyze existing data structure (columns, types) <!-- id: 0 -->
    - [x] Read `data_inspection.txt` to understand available columns <!-- id: 1 -->
    - [x] Verify column mapping in `01_limpieza_inicial.py` against actual data <!-- id: 2 -->
- [x] Evaluate Data Cleaning Logic <!-- id: 3 -->
    - [x] Assess handling of negative values (deletion vs. adjustment) <!-- id: 4 -->
    - [x] Verify date handling (Cash basis vs. Accrual basis - FechaRecaudo vs Vigencia/Mes) <!-- id: 5 -->
    - [x] Check for granularity (Municipal level support) <!-- id: 6 -->
- [x] Refactor Data Cleaning (`01_limpieza_inicial.py`) <!-- id: 15 -->
    - [x] Implement Municipal vs Departmental filtering <!-- id: 16 -->
    - [x] Implement monthly netting for negative values <!-- id: 17 -->
    - [x] Standardize date handling to Vigencia-Mes <!-- id: 18 -->
- [x] Implement Box-Jenkins Analysis (`02b_analisis_box_jenkins.py`) <!-- id: 19 -->
    - [x] Create script for stationarity (ADF, KPSS) and seasonality (ACF, PACF) checks <!-- id: 20 -->
    - [x] Analyze all three horizons (Monthly, Quarterly, Semesterly) <!-- id: 21 -->
- [x] Refactor Feature Engineering (`02_feature_engineering.py`) <!-- id: 22 -->
    - [x] Add municipal filtering option <!-- id: 23 -->
    - [x] Strict 2020-2025 Train / 2026 Test split <!-- id: 24 -->
- [x] Refactor Modeling (`03_modelos.py`) <!-- id: 25 -->
    - [x] Train on 2020-2025 only <!-- id: 26 -->
    - [x] Predict for 2026 <!-- id: 27 -->
- [x] Refactor Validation (`04_validacion_final.py`) <!-- id: 28 -->
    - [x] Multi-horizon validation (Monthly, Quarterly, Semesterly) against 2026 data (Demo on 2023) <!-- id: 29 -->
- [x] Refactor Dashboards (`06_dashboards_horizonte.py`) <!-- id: 30 -->
    - [x] Correct Data Range (2020-2025) <!-- id: 31 -->
    - [x] Generate Individual Dashboards per Model/Horizon (12 Total) <!-- id: 32 -->

# Thesis Documentation Support
- [x] Draft Guide for Findings (Hallazgos) <!-- id: 33 -->
- [x] Draft Guide for Conclusions and Recommendations <!-- id: 34 -->
- [x] Draft Guide for Methodology (CRISP-DM / OSEMN) <!-- id: 35 -->
- [x] Draft Executive Summary (Abstract) <!-- id: 36 -->
- [x] Draft Advanced Recommendations & 10 Approaches

# Advanced Diagnostics
- [x] Create specialized diagnostic module (`07_diagnostico_arima_sarima.py`)
    - [x] Analyze Monthly, Bi-monthly, Quarterly horizons
    - [x] Compare ARIMA vs SARIMA (AIC, RMSE, Visual Fit)
    - [x] Generate comparative plots

# Final Report & Data Export
- [x] Create reporting module (`08_reporte_final.py`)
    - [x] Generate Data Quality Report (Descriptive Stats of Cleaned Data)
    - [x] Summarize Time Series Diagnostics (Best Models Table)
    - [x] Summarize Time Series Diagnostics (Best Models Table)
    - [x] Export Modeling Datasets to CSV (Monthly, Bi-monthly, Quarterly)

# Municipal Analysis Module
- [x] Create `09_analisis_municipal.py`
    - [x] Descriptive: Pareto Analysis (Top Contributors, Revenue Concentration)
    - [x] Inferential: Correlation Analysis (Are municipalities synchronized?)
    - [x] Predictive: Generate Forecasts for Top 5 Municipalities (ARIMA/SARIMA)
    - [x] Predictive: Generate Forecasts for Top 5 Municipalities (ARIMA/SARIMA)
    - [x] Reporting: Export Municipal Stats CSV and Forecast Plots

# Final Thesis Compilation
- [x] Create `tesis_estructura_final.md` (Master Outline)
    - [x] Map Code/Reports to Chapters (Intro, Methods, Results, Discussion)
    - [x] Verify content completeness (Introduction, Problem Statement, Hypothesis)
    - [x] Link Final Artifacts (Charts, Tables, CSVs) to Sections

# Data Unification
- [x] Convert all Datasets to CSV (`01_limpieza`, `02_features`, `03_modelos`)
    - [x] `datos_depurados.csv`
    - [x] `train_mensual.csv`, `test_mensual.csv`
    - [x] `dataset_completo.csv`, `dataset_historico_completo.csv`

# Project Finalization
- [x] Update Repository (Git Add/Commit/Push)
- [x] Verify Remote Sync

# Phase 2: Refactoring and Enhancement
- [x] Create `plan_mejora_tesis.md` (Strategic Roadmap)
- [x] Refactoring: Create Shared Utils (`utils.py`)
    - [x] Extract Data Loading Logic
    - [x] Extract Metrics/Scoring Logic
    - [x] Extract Plotting Styles
- [x] Advanced Analysis: Exploratory Module (`10_analisis_avanzado.py`)
- [x] Report Update: Include Advanced Findings in Thesis Docs

# Phase 4: Hybrid/Ensemble and Optimization (Thesis Enhancements)
- [x] Implement Walk-Forward Validation in `utils.py`
- [x] Implement Hyperparameter Optimization (Grid/Random Search)
    - [x] Optimize SARIMAX (AIC/BIC selection)
    - [x] Optimize XGBoost (GridSearchCV)
- [x] Create Ensemble Module (`11_modelos_ensemble.py`)
    - [x] Training of Base Models (Level 0) with Optimal Params
    - [x] Stacking Implementation (Level 1 Meta-model)
    - [x] Evaluation: Ensemble vs Single Models
- [x] Update Thesis Documentation with Hybrid Approach Results

# Phase 5: Final Review and Code Freeze
- [x] Full Pipeline Verification (`main.py`)
- [x] Documentation Consistency Check (Hallazgos, Conclusiones, Walkthrough)
- [x] Final Codebase Cleanup
- [x] Generate Final Project Report (User Handoff)

# Phase 6: Post-Handoff Requests
- [x] Create Validation Script `12_validacion_cierre_2025.py`
    - [x] Implement Fourier & RobustScaler
    - [x] Out-of-Sample Validation (Aug-Oct 2025)
    - [x] Generate Comparative Report & Plots

- [x] Global Configuration Update (Split Dates)
    - [x] Update `config.py` with `TRAIN_CUTOFF_DATE = '2025-07-31'`
    - [x] Update `02_feature_engineering.py` to use date-based splitting
    - [x] Regenerate Train/Test Datasets







