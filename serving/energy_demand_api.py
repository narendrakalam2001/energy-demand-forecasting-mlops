# ============================================================
# ENERGY DEMAND FORECAST API — FastAPI Serving
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import logging
import time
import os
import json

from src.model_loader     import load_latest_model
from services.prediction_service import predict_demand

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Energy Demand Forecasting API",
    description="24-hour ahead electricity demand forecasting for Spain grid (ESIOS data)",
    version="1.0.0"
)

# ── Load model on startup ─────────────────────────────────────
try:
    model = load_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model = None


# ============================================================
# INPUT SCHEMA
# ============================================================

class HourlyInput(BaseModel):
    # Time features
    hour:         int   = Field(..., ge=0, le=23,   description="Hour of day (0-23)")
    day_of_week:  int   = Field(..., ge=0, le=6,    description="Day of week (0=Mon, 6=Sun)")
    month:        int   = Field(..., ge=1, le=12,   description="Month (1-12)")
    is_weekend:   int   = Field(0,  ge=0, le=1,    description="1 if weekend")

    # Lag demand features (MW) — required for model inference
    demand_lag_1h:   float = Field(..., description="Demand 1 hour ago (MW)")
    demand_lag_24h:  float = Field(..., description="Demand 24 hours ago (MW)")
    demand_lag_168h: float = Field(..., description="Demand 168 hours ago / same hour last week (MW)")

    # Rolling features
    demand_roll_mean_24h: float = Field(..., description="Rolling mean demand last 24h (MW)")
    demand_roll_std_24h:  float = Field(..., description="Rolling std demand last 24h (MW)")

    # Weather (optional)
    temp_celsius:    Optional[float] = Field(None, description="Temperature in Celsius")
    humidity:        Optional[float] = Field(None, description="Relative humidity (%)")
    wind_speed:      Optional[float] = Field(None, description="Wind speed (m/s)")

    # Price (optional)
    price_actual:    Optional[float] = Field(None, description="Electricity price EUR/MWh")

    # Generation mix (optional)
    renewable_penetration: Optional[float] = Field(None, description="Renewable share (0-1)")


class BatchInput(BaseModel):
    hours: List[HourlyInput] = Field(..., description="List of hourly inputs for batch forecast")


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Energy Demand Forecasting API is live ⚡",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict"
    }


@app.get("/health")
def health():
    return {
        "status":       "running",
        "model_loaded": model is not None
    }


@app.get("/model_info")
def model_info():
    registry_path = "energy_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            reg = json.load(f)
        card_path = reg.get("model_card_path", "")
        if card_path and os.path.exists(card_path):
            with open(card_path) as f:
                return json.load(f)
        return reg
    return {"error": "Model registry not found"}


@app.post("/predict")
def predict(input_data: HourlyInput):
    """
    Single-hour demand forecast.
    Returns: forecast_mw, demand_band, dispatch_action, rule_triggered.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    start      = time.time()
    input_dict = input_data.dict()

    result = predict_demand(model, input_dict)

    # ── Log prediction ────────────────────────────────────────
    log_record = {
        "timestamp":      time.time(),
        "hour":           input_dict["hour"],
        "day_of_week":    input_dict["day_of_week"],
        "month":          input_dict["month"],
        "demand_lag_1h":  input_dict["demand_lag_1h"],
        "forecast_mw":    result["forecast_mw"],
        "demand_band":    result["demand_band"],
        "dispatch_action":result["dispatch_action"],
        "rule_triggered": result.get("rule_triggered"),
    }

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["latency_seconds"] = round(time.time() - start, 4)
    return result


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    """
    Batch forecast for multiple hours.
    Returns list of forecast results.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start   = time.time()
    results = []

    for hour_input in batch.hours:
        result = predict_demand(model, hour_input.dict())
        results.append(result)

    return {
        "count":           len(results),
        "forecasts":       results,
        "latency_seconds": round(time.time() - start, 4)
    }
