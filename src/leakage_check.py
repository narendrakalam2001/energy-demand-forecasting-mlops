# ============================================================
# LEAKAGE CHECK — Energy Demand Forecasting
# ============================================================
# Time-series specific leakage checks:
#   1. Exact match — feature identical to target
#   2. Near-perfect correlation — corr >= threshold with target
#   3. Future leakage — features that contain future target info
#      (e.g., 'total load forecast' included when predicting
#       'total load actual')
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Columns known to be leaky for energy demand forecasting
# (direct proxies of actual load — provided by the grid operator
#  only after-the-fact, not available at forecast time)
KNOWN_LEAKY_COLS = [
    "total load forecast",    # TSO forecast of actual load → direct proxy
    "price day ahead",        # day-ahead price uses actual load in clearing
    "generation fossil hard coal",  # real-time generation follows real-time load
]


def detect_leakage(
    X_train:         pd.DataFrame,
    y_train:         pd.Series,
    threshold_corr:  float = 0.97,
    known_leaky:     list  = None
) -> list:
    """
    Runs leakage heuristics on training set.

    Args:
        X_train        : feature DataFrame (after chronological split)
        y_train        : target Series (demand in MW)
        threshold_corr : correlation threshold for flagging
        known_leaky    : list of column names known to be leaky

    Returns:
        List of warning strings. Empty list = no leakage detected.
    """
    warnings = []

    if known_leaky is None:
        known_leaky = KNOWN_LEAKY_COLS

    for col in X_train.columns:
        try:
            # ── Check 0: known leaky columns ─────────────────
            if col in known_leaky:
                warnings.append(
                    f"[LEAKAGE] '{col}' is a known future-leaky feature "
                    f"(TSO/operator data not available at forecast time) → remove"
                )
                continue

            # ── Check 1: exact match with target ─────────────
            if X_train[col].equals(y_train.astype(X_train[col].dtype)):
                warnings.append(
                    f"[LEAKAGE] '{col}' is identical to target → remove this feature"
                )
                continue

            # ── Check 2: near-perfect correlation ────────────
            if np.issubdtype(X_train[col].dtype, np.number):
                valid_mask = X_train[col].notna() & y_train.notna()
                if valid_mask.sum() < 10:
                    continue

                corr = abs(
                    np.corrcoef(
                        X_train.loc[valid_mask, col].values,
                        y_train[valid_mask].values
                    )[0, 1]
                )
                if corr >= threshold_corr:
                    warnings.append(
                        f"[LEAKAGE] '{col}' has corr={corr:.4f} with target "
                        f"(>= {threshold_corr}) → possible leakage"
                    )

        except Exception as e:
            logger.warning("Leakage check failed for column '%s': %s", col, e)

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Leakage check passed — no obvious leakage detected")

    return warnings


def check_future_lag_leakage(feature_cols: list, target_col: str) -> list:
    """
    Checks that no lag-0 (current-time) target value is used as feature.
    Lag-0 = direct copy of target = perfect leakage.
    """
    warnings = []
    suspicious = [c for c in feature_cols if target_col in c and "lag_0" in c]
    for col in suspicious:
        warnings.append(
            f"[LEAKAGE] '{col}' appears to be lag-0 of target → remove (perfect leak)"
        )
    return warnings
