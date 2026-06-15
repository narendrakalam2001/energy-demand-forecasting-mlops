# ============================================================
# FORECAST ENGINE — Energy Demand Forecasting
# ============================================================
# Fintech-grade → Grid-grade 4-tier decision system:
#   LOW     → low demand, reduce generation
#   NORMAL  → baseline, maintain generation
#   HIGH    → high demand, activate reserves
#   PEAK    → critical peak, emergency procurement
#
# Decision hierarchy:
#   1. Hard grid rules (override ML)
#   2. ML model forecast + demand bands
#
# Real-world analog: NLDC (National Load Dispatch Centre)
# India uses similar rule + ML hybrid systems.
# ============================================================

import pandas as pd
import numpy as np
import logging

from src.config import (
    DEMAND_BANDS,
    PEAK_DEMAND_THRESHOLD,
    LOW_DEMAND_THRESHOLD,
    PRICE_SPIKE_THRESHOLD,
    MAPE_ALERT_THRESHOLD
)

logger = logging.getLogger(__name__)


# ============================================================
# DEMAND BAND — forecast value → LOW/NORMAL/HIGH/PEAK
# ============================================================

def get_demand_band(forecast_mw: float) -> str:
    for band, (low, high) in DEMAND_BANDS.items():
        if low <= forecast_mw < high:
            return band
    return "PEAK"


# ============================================================
# FORECAST ENGINE — row-level decisions
# ============================================================

def forecast_engine(
    forecast_df: pd.DataFrame,
    forecasts:   np.ndarray,
    actuals:     np.ndarray = None
) -> list:
    """
    For each forecasted hour → returns dispatch decision string.

    Rule priority:
      1. Price spike + peak demand → EMERGENCY_PROCUREMENT
      2. Forecast >= 42000 MW      → ACTIVATE_RESERVE
      3. Forecast <= 20000 MW      → REDUCE_GENERATION
      4. Forecast >= 35000 MW      → MONITOR_HIGH
      5. else                      → NORMAL_OPERATION
    """
    decisions = []

    for idx, (_, row) in enumerate(forecast_df.iterrows()):
        fc = forecasts[idx]

        # ── Hard rule: price spike + high demand → emergency ──
        price = row.get("price actual", 0)
        if price > PRICE_SPIKE_THRESHOLD and fc >= PEAK_DEMAND_THRESHOLD * 0.9:
            decisions.append("EMERGENCY_PROCUREMENT")
            continue

        # ── ML-based band decisions ───────────────────────────
        if fc >= PEAK_DEMAND_THRESHOLD:
            decisions.append("ACTIVATE_RESERVE")
        elif fc <= LOW_DEMAND_THRESHOLD:
            decisions.append("REDUCE_GENERATION")
        elif fc >= PEAK_DEMAND_THRESHOLD * 0.85:
            decisions.append("MONITOR_HIGH")
        else:
            decisions.append("NORMAL_OPERATION")

    return decisions


# ============================================================
# SCORE FORECAST — single hour (for API)
# ============================================================

def score_forecast(
    row:            dict,
    forecast_mw:    float,
    actual_mw:      float = None
) -> dict:
    """
    Returns structured forecast output for a single hour.
    Used by FastAPI prediction endpoint.
    """
    demand_band    = get_demand_band(forecast_mw)
    rule_triggered = None

    price = row.get("price_actual", row.get("price actual", 0)) or 0

    # ── Rule-based overrides ──────────────────────────────────
    if price > PRICE_SPIKE_THRESHOLD and forecast_mw >= PEAK_DEMAND_THRESHOLD * 0.9:
        dispatch_action = "EMERGENCY_PROCUREMENT"
        rule_triggered  = "PRICE_SPIKE_AND_PEAK"

    elif forecast_mw >= PEAK_DEMAND_THRESHOLD:
        dispatch_action = "ACTIVATE_RESERVE"
        rule_triggered  = "PEAK_DEMAND_THRESHOLD"

    elif forecast_mw <= LOW_DEMAND_THRESHOLD:
        dispatch_action = "REDUCE_GENERATION"
        rule_triggered  = "LOW_DEMAND_THRESHOLD"

    elif forecast_mw >= PEAK_DEMAND_THRESHOLD * 0.85:
        dispatch_action = "MONITOR_HIGH"
        rule_triggered  = None

    else:
        dispatch_action = "NORMAL_OPERATION"

    result = {
        "forecast_mw":    round(float(forecast_mw), 2),
        "demand_band":    demand_band,
        "dispatch_action":dispatch_action,
        "rule_triggered": rule_triggered,
    }

    # ── If actual provided (post-hoc eval) ───────────────────
    if actual_mw is not None:
        error_mw  = abs(actual_mw - forecast_mw)
        error_pct = error_mw / (actual_mw + 1e-9) * 100
        result["actual_mw"]  = round(float(actual_mw), 2)
        result["error_mw"]   = round(float(error_mw),  2)
        result["error_pct"]  = round(float(error_pct), 4)

    return result


# ============================================================
# FORECAST MONITOR SCORES — for dashboard
# ============================================================

def build_monitor_scores(
    forecasts:   np.ndarray,
    actuals:     np.ndarray,
    timestamps:  pd.Series,
    decisions:   list
) -> pd.DataFrame:
    """
    Builds monitor_scores.csv equivalent for energy forecasting.
    Used by Streamlit dashboard.
    """
    errors    = np.abs(np.asarray(actuals) - np.asarray(forecasts))
    error_pct = errors / (np.asarray(actuals) + 1e-9) * 100

    df = pd.DataFrame({
        "timestamp":  timestamps.values if hasattr(timestamps, "values") else timestamps,
        "forecast":   np.round(forecasts, 2),
        "actual":     np.round(actuals,   2),
        "error_mw":   np.round(errors,    2),
        "error_pct":  np.round(error_pct, 4),
        "demand_band":[get_demand_band(f) for f in forecasts],
        "decision":   decisions,
    })

    return df
