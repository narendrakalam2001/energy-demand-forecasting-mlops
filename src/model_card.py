# ============================================================
# MODEL CARD — Energy Demand Forecasting
# ============================================================
# Builds and saves a structured model card — Google Model Cards
# standard adapted for time-series regression.
# ============================================================

import os
import json
import time
import logging

from typing import Optional

logger = logging.getLogger(__name__)


def build_model_card(
    selected_name:   str,
    train_fit_size:  int,
    cal_size:        int,
    test_size:       int,
    train_date_range:tuple,   # (start_str, end_str)
    test_date_range: tuple,
    metrics:         dict,   # rmse, mae, mape, smape, r2, cvrmse
    cost_result:     dict,
    decision_counts: dict,
    feature_order:   list,
    selector_k:      int,
    version:         str  = "v1",
    fi_dict:         Optional[dict] = None,
    shap_dict:       Optional[dict] = None,
    demand_band_acc: Optional[dict] = None,
) -> dict:
    """Assembles the full model card dictionary."""

    card = {
        "model_version": version,
        "model_name":    selected_name,
        "trained_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "project":       "Energy Demand Forecasting System",
        "domain":        "Energy / Utilities",
        "target":        "total load actual (MW)",

        "dataset": {
            "train_fit_size":   train_fit_size,
            "calibration_size": cal_size,
            "test_size":        test_size,
            "train_date_range": list(train_date_range),
            "test_date_range":  list(test_date_range),
        },

        "metrics": {
            "rmse":     round(float(metrics.get("rmse",   0)), 4),
            "mae":      round(float(metrics.get("mae",    0)), 4),
            "mape":     round(float(metrics.get("mape",   0)), 4),
            "smape":    round(float(metrics.get("smape",  0)), 4),
            "r2":       round(float(metrics.get("r2",     0)), 4),
            "cvrmse":   round(float(metrics.get("cvrmse", 0)), 4),
        },

        "cost_evaluation": cost_result,
        "dispatch_decisions": decision_counts,

        "pipeline_config": {
            "feature_order": feature_order,
            "selector_k":    selector_k,
        },
    }

    if demand_band_acc is not None:
        card["demand_band_accuracy"] = demand_band_acc

    if fi_dict is not None:
        card["feature_importances"] = fi_dict

    if shap_dict is not None:
        card["shap_top_features"] = shap_dict

    return card


def save_model_card(card: dict, model_dir: str, selected_name: str, version: str = "v1") -> str:
    os.makedirs(model_dir, exist_ok=True)
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info("Model card saved → %s", card_path)
    return card_path


def load_model_card(model_dir: str, selected_name: str, version: str = "v1") -> dict:
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")
    if not os.path.exists(card_path):
        raise FileNotFoundError(f"Model card not found: {card_path}")
    with open(card_path) as f:
        card = json.load(f)
    logger.info("Model card loaded ← %s", card_path)
    return card
