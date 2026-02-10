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
- [x] Draft Advanced Recommendations & 10 Approaches <!-- id: 37 -->
