# ============================================================
# PREDICTION SERVICE — Energy Demand Forecasting
# ============================================================

import pandas as pd
import numpy as np
import logging

from src.forecast_engine import score_forecast
from src.data_loader     import add_time_features

logger = logging.getLogger(__name__)


# ============================================================
# PREPARE FEATURES — for API inference
# ============================================================

def prepare_features(input_data: dict) -> pd.DataFrame:
    """
    Takes raw API input dict → returns feature DataFrame.
    Mirrors the columns expected by the trained pipeline.
    All lag/rolling features must be passed in by caller
    (they are not computable at inference time without history).
    """
    df = pd.DataFrame([input_data])

    # ── Cyclical encoding (must match training) ───────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)

    # ── Peak/off-peak flag ────────────────────────────────────
    df["is_morning_peak"]  = df["hour"].between(7,  10).astype(int)
    df["is_evening_peak"]  = df["hour"].between(18, 22).astype(int)
    df["is_overnight"]     = (~df["hour"].between(6, 22)).astype(int)

    # ── Season ────────────────────────────────────────────────
    season_map = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}
    df["season"] = df["month"].map(season_map)

    # ── Temperature features ──────────────────────────────────
    if "temp_celsius" in df.columns and df["temp_celsius"].notna().any():
        df["cooling_degree"] = np.maximum(0, df["temp_celsius"] - 22)
        df["heating_degree"] = np.maximum(0, 15 - df["temp_celsius"])
        df["temp_x_hour"]    = df["temp_celsius"] * df["hour"]
        df["temp_x_peak"]    = df["temp_celsius"] * df["is_evening_peak"]

    return df


# ============================================================
# PREDICT — single hour
# ============================================================

def predict_demand(model, input_data: dict) -> dict:
    """
    Full prediction flow for one hour:
      1. Feature engineering
      2. Model forecast (MW)
      3. Forecast engine scoring (rules + bands)

    Returns structured forecast output dict.
    """
    df = prepare_features(input_data)

    try:
        forecast_mw = float(model.predict(df)[0])
        # Clamp negative forecasts
        forecast_mw = max(0.0, forecast_mw)
    except Exception as e:
        logger.error("model.predict failed: %s", e)
        forecast_mw = input_data.get("demand_lag_24h", 30000.0)  # fallback: same hour yesterday

    row    = df.iloc[0].to_dict()
    result = score_forecast(row, forecast_mw)

    logger.info(
        "Forecast  |  mw=%.2f  band=%s  action=%s",
        result["forecast_mw"],
        result["demand_band"],
        result["dispatch_action"]
    )

    return result
