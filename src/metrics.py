# ============================================================
# METRICS — Energy Demand Forecasting
# ============================================================
# Regression + time-series evaluation metrics:
#   RMSE, MAE, MAPE, sMAPE, R², CVRMSE
#   PSI (Population Stability Index) — feature drift
#   Demand band accuracy
#   Cost-sensitive evaluation (MW forecast error cost)
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ============================================================
# CORE REGRESSION METRICS
# ============================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — penalizes large errors."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error — interpretable in MW."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """
    Mean Absolute Percentage Error (%).
    Standard KPI for energy demand forecasting at Tata Power, NTPC etc.
    Industry target: MAPE < 3% (excellent), < 5% (good), < 10% (acceptable).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """
    Symmetric MAPE — bounded, handles near-zero actuals better.
    Range: 0–200%.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(
        np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
        ) * 100
    )


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination.
    R² = 1.0 → perfect; R² = 0 → predicting mean; R² < 0 → worse than mean.
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    ss_res  = np.sum((y_true - y_pred) ** 2)
    ss_tot  = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))


def cvrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Variation of RMSE (%).
    CVRMSE = RMSE / mean(y_true) × 100.
    Energy industry standard: CVRMSE < 15% is acceptable.
    """
    mean_actual = np.mean(np.asarray(y_true, dtype=float))
    return float(rmse(y_true, y_pred) / (mean_actual + 1e-9) * 100)


def compute_all_metrics(y_true, y_pred) -> dict:
    """Compute all regression metrics in one call."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result = {
        "rmse":   round(rmse(y_true,  y_pred), 4),
        "mae":    round(mae(y_true,   y_pred), 4),
        "mape":   round(mape(y_true,  y_pred), 4),
        "smape":  round(smape(y_true, y_pred), 4),
        "r2":     round(r2_score(y_true, y_pred), 4),
        "cvrmse": round(cvrmse(y_true, y_pred), 4),
    }

    logger.info(
        "Metrics  |  RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%  R²=%.4f  CVRMSE=%.2f%%",
        result["rmse"], result["mae"], result["mape"],
        result["r2"], result["cvrmse"]
    )

    return result


# ============================================================
# PSI — Population Stability Index
# ============================================================

def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.1   → stable (no retrain needed)
    PSI 0.1–0.2 → moderate drift (monitor)
    PSI > 0.2   → major drift (retrain recommended)

    CRITICAL: Bins BOTH arrays using edges from expected only.
    Common bug: ranking both independently → PSI always ~0.
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        quantiles  = np.linspace(0, 100, buckets + 1)
        bin_edges  = np.percentile(expected, quantiles)
        bin_edges  = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        psi_value = float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))
        return psi_value

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# DRIFT REPORT — feature mean shift
# ============================================================

def simple_drift_report(
    X_ref: pd.DataFrame,
    X_new: pd.DataFrame,
    top_n: int = 10
) -> pd.Series:
    diffs = (X_ref.mean() - X_new.mean()).abs()
    rel   = (diffs / (X_ref.std().replace(0, 1))).sort_values(ascending=False)
    return rel.head(top_n)


# ============================================================
# COST-SENSITIVE EVALUATION — Energy domain
# ============================================================

def cost_sensitive_evaluation(
    y_true,
    y_pred,
    imbalance_cost_per_mwh: float = 85.0,    # EUR/MWh — Spain imbalance cost (2018 avg)
    peak_error_multiplier:  float = 2.5,     # peak hours cost 2.5× more
    peak_threshold_mw:      float = 40000.0  # MW — peak demand threshold
) -> dict:
    """
    Grid imbalance cost model for energy demand forecasting.

    In real power grids (NTPC, Tata Power, Adani):
      - Under-forecast → grid buys expensive spot power → imbalance cost
      - Over-forecast  → surplus generation → curtailment cost
    Both are proportional to forecast error in MWh.

    Cost = |error_MWh| × imbalance_cost_per_MWh
    Peak hours multiply cost by peak_error_multiplier.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors_mwh  = np.abs(y_true - y_pred)           # hourly error in MW (treated as MWh)
    is_peak     = y_true >= peak_threshold_mw
    is_under    = y_pred < y_true

    base_cost   = errors_mwh * imbalance_cost_per_mwh
    peak_cost   = base_cost * peak_error_multiplier
    cost_vector = np.where(is_peak, peak_cost, base_cost)

    under_cost  = cost_vector[is_under].sum()
    over_cost   = cost_vector[~is_under].sum()
    total_cost  = cost_vector.sum()

    result = {
        "total_forecast_error_mw":   round(float(errors_mwh.sum()),  2),
        "mean_error_mw":             round(float(errors_mwh.mean()), 2),
        "peak_hours_count":          int(is_peak.sum()),
        "under_forecast_count":      int(is_under.sum()),
        "over_forecast_count":       int((~is_under).sum()),
        "estimated_under_cost_eur":  round(float(under_cost),  2),
        "estimated_over_cost_eur":   round(float(over_cost),   2),
        "total_estimated_cost_eur":  round(float(total_cost),  2),
    }

    logger.info(
        "Cost eval  |  total_error=%.0fMW  total_cost=€%.0f  "
        "under=€%.0f  over=€%.0f",
        result["total_forecast_error_mw"],
        result["total_estimated_cost_eur"],
        result["estimated_under_cost_eur"],
        result["estimated_over_cost_eur"],
    )

    return result


# ============================================================
# DEMAND BAND ACCURACY
# ============================================================

def demand_band_accuracy(y_true, y_pred) -> dict:
    """
    Measures what % of hours the model correctly identifies
    the demand band (LOW / NORMAL / HIGH / PEAK).
    Used by grid operators to verify operational readiness.
    """
    from src.data_loader import get_demand_band

    bands_true = [get_demand_band(v) for v in y_true]
    bands_pred = [get_demand_band(v) for v in y_pred]

    correct = sum(bt == bp for bt, bp in zip(bands_true, bands_pred))
    total   = len(bands_true)

    accuracy = round(correct / total * 100, 2) if total > 0 else 0.0

    result = {
        "band_accuracy_pct": accuracy,
        "total_hours":       total,
        "correct_hours":     correct,
    }

    logger.info("Demand band accuracy: %.2f%%  (%d/%d)", accuracy, correct, total)
    return result
