# ============================================================
# DATA LOADER + FEATURE ENGINEERING — Energy Demand Forecasting
# ============================================================
# Spain ESIOS grid data: hourly energy generation, load,
# renewable mix, prices, and weather (2015–2018).
#
# Feature engineering covers:
#   - Time/calendar features (hour, dow, month, season)
#   - Lag features (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)
#   - Rolling stats (mean, std) over multiple windows
#   - Renewable penetration ratio
#   - Price-demand interaction
#   - Temperature × demand interaction
# ============================================================

import numpy as np
import pandas as pd
import logging

from src.config import (
    TARGET_COL, LAG_HOURS, ROLL_WINDOWS,
    PEAK_DEMAND_THRESHOLD, LOW_DEMAND_THRESHOLD
)

logger = logging.getLogger(__name__)

REQUIRED_ENERGY_COLS = ["time", "total load actual"]
REQUIRED_WEATHER_COLS = ["dt_iso", "temp"]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the energy generation/load dataframe."""

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ── Parse datetime ────────────────────────────────────────
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df          = df.dropna(subset=["time"])
        df          = df.sort_values("time").reset_index(drop=True)
    else:
        raise ValueError("Energy dataset must contain 'time' column")

    # ── Check required columns ────────────────────────────────
    missing = [c for c in REQUIRED_ENERGY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Drop rows where target is null ────────────────────────
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with null target", dropped)

    # ── Sanity: target must be positive ───────────────────────
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)

    # ── Deduplication ─────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["time"], ignore_index=True)
    if len(df) < before:
        logger.info("Dropped %d duplicate timestamps", before - len(df))

    # ── Minimum size check ────────────────────────────────────
    if len(df) < 1000:
        raise ValueError("Dataset too small (< 1000 rows)")

    logger.info("Energy data validated  |  shape=%s  |  range=%s to %s",
                df.shape,
                df["time"].min().date(),
                df["time"].max().date())

    return df


def validate_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the weather dataframe."""

    df.columns = df.columns.str.strip().str.lower()

    if "dt_iso" in df.columns:
        df["dt_iso"] = pd.to_datetime(df["dt_iso"], utc=True, errors="coerce")
        df = df.dropna(subset=["dt_iso"])
        df = df.sort_values("dt_iso").reset_index(drop=True)

    missing = [c for c in REQUIRED_WEATHER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required weather columns: {missing}")

    logger.info("Weather data validated  |  shape=%s", df.shape)
    return df


# ============================================================
# MERGE ENERGY + WEATHER
# ============================================================

def merge_energy_weather(
    energy_df:  pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge hourly energy data with weather data on timestamp.
    Weather is aggregated per hour (mean across all city stations).
    """

    # ── Aggregate weather: mean per hour (multi-city dataset) ─
    weather_agg = (
        weather_df
        .assign(hour=weather_df["dt_iso"].dt.floor("h"))
        .groupby("hour")
        .agg({
            "temp":         "mean",
            "temp_min":     "mean" if "temp_min"     in weather_df.columns else lambda x: np.nan,
            "temp_max":     "mean" if "temp_max"     in weather_df.columns else lambda x: np.nan,
            "humidity":     "mean" if "humidity"     in weather_df.columns else lambda x: np.nan,
            "wind_speed":   "mean" if "wind_speed"   in weather_df.columns else lambda x: np.nan,
            "clouds_all":   "mean" if "clouds_all"   in weather_df.columns else lambda x: np.nan,
            "pressure":     "mean" if "pressure"     in weather_df.columns else lambda x: np.nan,
        })
        .reset_index()
    )

    # Drop columns that ended up all NaN (missing in dataset)
    weather_agg = weather_agg.dropna(axis=1, how="all")

    # ── Merge ─────────────────────────────────────────────────
    energy_df["_merge_key"] = energy_df["time"].dt.floor("h")
    merged = energy_df.merge(
        weather_agg.rename(columns={"hour": "_merge_key"}),
        on="_merge_key",
        how="left"
    ).drop(columns=["_merge_key"])

    logger.info("Merged energy + weather  |  shape=%s", merged.shape)

    return merged


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """
    Calendar and cyclical time features — critical for
    energy demand which has strong daily/weekly seasonality.
    """
    df = df.copy()
    t  = df[time_col]

    # ── Calendar ──────────────────────────────────────────────
    df["hour"]        = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek         # 0=Mon, 6=Sun
    df["month"]       = t.dt.month
    df["quarter"]     = t.dt.quarter
    df["day_of_year"] = t.dt.dayofyear
    df["week_of_year"]= t.dt.isocalendar().week.astype(int)
    df["year"]        = t.dt.year

    # ── Boolean flags ─────────────────────────────────────────
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_morning_peak"]  = df["hour"].between(7,  10).astype(int)   # 7–10 AM
    df["is_evening_peak"]  = df["hour"].between(18, 22).astype(int)   # 6–10 PM
    df["is_overnight"]     = (~df["hour"].between(6, 22)).astype(int)

    # ── Season (Spain: Mediterranean climate) ─────────────────
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0,   # Winter
        3: 1,  4: 1, 5: 1,   # Spring
        6: 2,  7: 2, 8: 2,   # Summer
        9: 3,  10: 3, 11: 3  # Autumn
    })

    # ── Cyclical encoding (sin/cos) — avoids discontinuity ────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)

    return df


def add_lag_features(
    df:         pd.DataFrame,
    target_col: str       = TARGET_COL,
    lag_hours:  list      = LAG_HOURS
) -> pd.DataFrame:
    """
    Lag features — past demand at t-1h, t-24h, t-168h etc.
    Critical for time-series: model learns from recent history.

    WARNING: Must be called after sorting by time.
    NaN rows from lags are dropped later in training_pipeline.
    """
    df = df.copy()
    for lag in lag_hours:
        df[f"demand_lag_{lag}h"] = df[target_col].shift(lag)
        logger.debug("Lag feature added: demand_lag_%dh", lag)
    return df


def add_rolling_features(
    df:          pd.DataFrame,
    target_col:  str  = TARGET_COL,
    windows:     list = ROLL_WINDOWS
) -> pd.DataFrame:
    """
    Rolling mean and std over multiple windows.
    Captures trend and volatility of recent demand.
    """
    df = df.copy()
    for w in windows:
        df[f"demand_roll_mean_{w}h"] = df[target_col].shift(1).rolling(w).mean()
        df[f"demand_roll_std_{w}h"]  = df[target_col].shift(1).rolling(w).std()
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering — calls all sub-routines.
    Returns fully engineered DataFrame ready for modelling.
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # ── Renewable penetration ratio ───────────────────────────
    renewable_cols = [c for c in df.columns if any(
        x in c for x in ["solar", "wind", "hydro", "biomass", "geothermal", "wave"]
    )]
    if renewable_cols:
        df["renewable_total"]       = df[renewable_cols].fillna(0).sum(axis=1)
        total_gen_cols = [c for c in df.columns if "generation" in c and c not in renewable_cols]
        if total_gen_cols:
            total_gen = df[total_gen_cols].fillna(0).sum(axis=1) + df["renewable_total"]
            df["renewable_penetration"] = df["renewable_total"] / (total_gen + 1)
        else:
            df["renewable_penetration"] = df["renewable_total"] / (df[TARGET_COL] + 1)

    # ── Price interaction ─────────────────────────────────────
    if "price actual" in df.columns:
        df["price_demand_ratio"] = df["price actual"] / (df[TARGET_COL] + 1)
        df["price_lag_1h"]       = df["price actual"].shift(1)
        df["price_change_1h"]    = df["price actual"].diff(1)

    # ── Temperature × demand ──────────────────────────────────
    if "temp" in df.columns:
        # Convert Kelvin to Celsius (Kaggle dataset has Kelvin)
        if df["temp"].median() > 200:
            df["temp_celsius"] = df["temp"] - 273.15
        else:
            df["temp_celsius"] = df["temp"]

        df["temp_x_hour"]       = df["temp_celsius"] * df["hour"]
        df["temp_x_peak"]       = df["temp_celsius"] * df["is_evening_peak"]
        df["cooling_degree"]    = np.maximum(0, df["temp_celsius"] - 22)   # cooling demand
        df["heating_degree"]    = np.maximum(0, 15 - df["temp_celsius"])   # heating demand

    # ── Demand flag features ──────────────────────────────────
    df["is_peak_demand"] = (df[TARGET_COL] >= PEAK_DEMAND_THRESHOLD).astype(int)
    df["is_low_demand"]  = (df[TARGET_COL] <= LOW_DEMAND_THRESHOLD).astype(int)

    # ── Demand delta (rate of change) ─────────────────────────
    df["demand_delta_1h"] = df[TARGET_COL].diff(1)
    df["demand_delta_24h"]= df[TARGET_COL].diff(24)

    logger.info("Feature engineering done  |  total columns=%d", df.shape[1])

    return df


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(
    df:        pd.DataFrame,
    target:    str = TARGET_COL,
    threshold: int = 10
):
    """
    Auto-detect ordinal / continuous / binary columns.
    Excludes target column and datetime columns.
    """
    ordinal_cols    = []
    continuous_cols = []
    binary_cols     = []

    exclude = {target, "time", "dt_iso", "_merge_key"}

    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == "object" or str(df[col].dtype) == "datetime64[ns, UTC]":
            continue

        n_unique = df[col].nunique(dropna=False)

        if np.issubdtype(df[col].dtype, np.number):
            if n_unique == 2:
                binary_cols.append(col)
            elif 3 <= n_unique <= threshold:
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)

    logger.info("Feature types  |  ordinal=%d  continuous=%d  binary=%d",
                len(ordinal_cols), len(continuous_cols), len(binary_cols))

    return ordinal_cols, continuous_cols, binary_cols


# ============================================================
# DEMAND BAND
# ============================================================

def get_demand_band(value_mw: float) -> str:
    from src.config import DEMAND_BANDS
    for band, (low, high) in DEMAND_BANDS.items():
        if low <= value_mw < high:
            return band
    return "PEAK"
