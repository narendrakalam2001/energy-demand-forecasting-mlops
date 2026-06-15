# ============================================================
# CONFIGURATION — Energy Demand Forecasting System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE  = 42
N_JOBS        = 1

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS             = 5
RANDOM_SEARCH_ITERS  = 5
TIME_SERIES_CV_FOLDS = 3   # TimeSeriesSplit folds

# ── Feature selection ────────────────────────────────────────
SELECT_K = 20

# ── Outlier clipping ─────────────────────────────────────────
CLIP_FOLD = 1.5

# ── Target column ────────────────────────────────────────────
TARGET_COL = "total load actual"   # actual grid load in MW

# ── Forecast horizon ─────────────────────────────────────────
FORECAST_HORIZON_H = 24   # 24-hour ahead forecast

# ── Demand alert bands (MW) — Spain grid reference values ────
DEMAND_BANDS = {
    "LOW":    (0,     25000),
    "NORMAL": (25000, 40000),
    "HIGH":   (40000, 55000),
    "PEAK":   (55000, 999999),
}

# ── Business rule thresholds ─────────────────────────────────
PEAK_DEMAND_THRESHOLD   = 42000   # MW — trigger peak alert
LOW_DEMAND_THRESHOLD    = 20000   # MW — trigger low-demand alert
MAX_FORECAST_ERROR_PCT  = 15.0    # % — max acceptable MAPE before alert
PRICE_SPIKE_THRESHOLD   = 100.0   # EUR/MWh — price spike rule

# ── PSI drift thresholds ─────────────────────────────────────
PSI_MODERATE = 0.10
PSI_HIGH     = 0.20

# ── Forecast score alert ─────────────────────────────────────
MAPE_ALERT_THRESHOLD  = 10.0   # % — alert if MAPE > 10%
RMSE_ALERT_THRESHOLD  = 3000.0 # MW — alert if RMSE > 3000 MW

# ── Challenger promotion gates ────────────────────────────────
MIN_RMSE_IMPROVEMENT    = 50.0    # MW — challenger must beat champion by >= 50 MW RMSE
MIN_R2_THRESHOLD        = 0.85    # R² >= 0.85 required
MAX_GENERALIZATION_GAP  = 0.10    # train-test RMSE gap <= 10%

# ── Lag features ─────────────────────────────────────────────
LAG_HOURS   = [1, 2, 3, 6, 12, 24, 48, 168]   # 168 = 1 week
ROLL_WINDOWS = [6, 12, 24, 48, 168]            # rolling mean/std windows

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR   = "energy_models"
METRICS_LOG = "energy_models/metrics_log.csv"
TESTS_DIR   = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)