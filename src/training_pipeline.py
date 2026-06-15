# ============================================================
# TRAINING PIPELINE — Energy Demand Forecasting
# ============================================================
# 24-step production pipeline:
#   1.  Load energy + weather data
#   2.  Validate data
#   3.  Merge energy + weather
#   4.  Feature engineering (time, lag, rolling, interactions)
#   5.  Drop rows with NaN from lags
#   6.  Feature type detection
#   7.  Time-series chronological split (no shuffle)
#   8.  Leakage detection
#   9.  Build preprocessors (scaled + unscaled)
#   10. Tune scaled models (TimeSeriesSplit CV)
#   11. Tune unscaled models
#   12. Train MLP separately
#   13. Evaluate all models
#   14. Select best model
#   15. Forecast + metrics on test set
#   16. Forecast engine decisions
#   17. Cost-sensitive evaluation (grid imbalance cost)
#   18. Repeated CV stability check
#   19. Feature importance (permutation)
#   20. SHAP explainability
#   21. PSI drift monitoring (train vs test features)
#   22. Build monitor scores for dashboard
#   23. Model card + save
#   24. Challenger comparison + MLflow
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.config          import RANDOM_STATE, N_JOBS, MODEL_DIR, SELECT_K, TARGET_COL, TIME_SERIES_CV_FOLDS

# ── Plots output directory ────────────────────────────────────
PLOTS_DIR = os.path.join("docs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
from src.data_loader     import (validate_energy_data, validate_weather_data,
                                  merge_energy_weather, add_engineered_features,
                                  detect_feature_types, get_demand_band)
from src.preprocessing   import build_preprocessors, safe_k, time_series_split
from src.model_tuning    import scaled_models, unscaled_models, tune_models, train_mlp_pipeline
from src.evaluation      import (evaluate_models, select_best_model,
                                  compute_feature_importance, compute_shap,
                                  save_model_and_card, mlflow_log_run)
from src.leakage_check   import detect_leakage, check_future_lag_leakage
from src.model_card      import build_model_card, save_model_card
from src.model_loader    import run_challenger_comparison
from src.metrics         import (compute_all_metrics, psi, cost_sensitive_evaluation,
                                  demand_band_accuracy)
from src.forecast_engine import forecast_engine, build_monitor_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training():

    # ─────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────────────────
    # Dataset: Spain Energy Grid (Kaggle)
    # https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
    # Download and update paths below:

    ENERGY_PATH  = r"D:\Data Science Datasets\energy data set\energy_dataset.csv"
    WEATHER_PATH = r"D:\Data Science Datasets\energy data set\weather_features.csv"

    logger.info("Loading energy data from: %s", ENERGY_PATH)
    energy_df  = pd.read_csv(ENERGY_PATH)

    logger.info("Loading weather data from: %s", WEATHER_PATH)
    weather_df = pd.read_csv(WEATHER_PATH)

    # ─────────────────────────────────────────────────────────
    # 2. VALIDATE
    # ─────────────────────────────────────────────────────────
    energy_df  = validate_energy_data(energy_df)
    weather_df = validate_weather_data(weather_df)

    # ─────────────────────────────────────────────────────────
    # 3. MERGE
    # ─────────────────────────────────────────────────────────
    df = merge_energy_weather(energy_df, weather_df)

    # ─────────────────────────────────────────────────────────
    # 4. FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────
    df = add_engineered_features(df)

    # ─────────────────────────────────────────────────────────
    # 5. DROP NaN ROWS (from lag features)
    # ─────────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Drop lag NaN rows — first 168 rows will have NaN lags
    lag_cols = [c for c in df.columns if "lag" in c or "roll" in c]
    df = df.dropna(subset=lag_cols[:1] if lag_cols else []).reset_index(drop=True)
    logger.info("Dropped %d NaN rows from lag features. Remaining: %d", before - len(df), len(df))

    # ─────────────────────────────────────────────────────────
    # 6. FEATURE TYPE DETECTION
    # ─────────────────────────────────────────────────────────
    # Exclude datetime + target + columns known leaky/to drop
    drop_always = ["time", "total load forecast"]
    df_model = df.drop(columns=[c for c in drop_always if c in df.columns], errors="ignore")

    ord_cols, cont_cols, bin_cols = detect_feature_types(df_model, target=TARGET_COL)

    X = df_model.drop(columns=[TARGET_COL])
    y = df_model[TARGET_COL]

    logger.info("Dataset  |  shape=%s  |  target_mean=%.2f MW  target_std=%.2f MW",
                df_model.shape, y.mean(), y.std())

    # ─────────────────────────────────────────────────────────
    # 7. CHRONOLOGICAL SPLIT (no shuffle — time series!)
    # ─────────────────────────────────────────────────────────
    n      = len(X)
    n_test = int(n * 0.15)
    n_cal  = int(n * 0.10)
    n_fit  = n - n_test - n_cal

    X_train_fit = X.iloc[:n_fit].copy()
    X_cal       = X.iloc[n_fit: n_fit + n_cal].copy()
    X_test      = X.iloc[n_fit + n_cal:].copy()
    y_train_fit = y.iloc[:n_fit].copy()
    y_cal       = y.iloc[n_fit: n_fit + n_cal].copy()
    y_test      = y.iloc[n_fit + n_cal:].copy()

    # Keep timestamps for monitor scores
    timestamps_test = (
        df_model["time"].iloc[n_fit + n_cal:]
        if "time" in df_model.columns
        else pd.Series(range(len(X_test)))
    )

    logger.info("Split  |  train_fit=%d  cal=%d  test=%d", len(X_train_fit), len(X_cal), len(X_test))

    # ─────────────────────────────────────────────────────────
    # 8. LEAKAGE DETECTION
    # ─────────────────────────────────────────────────────────
    leak_warnings = detect_leakage(X_train_fit, y_train_fit)
    lag_warnings  = check_future_lag_leakage(X.columns.tolist(), TARGET_COL)

    if leak_warnings or lag_warnings:
        for w in leak_warnings + lag_warnings:
            logger.warning("LEAKAGE: %s", w)
    else:
        logger.info("No obvious leakage detected")

    # ─────────────────────────────────────────────────────────
    # 9. BUILD PREPROCESSORS
    # ─────────────────────────────────────────────────────────
    pre_scaled, pre_unscaled, feature_order = build_preprocessors(
        ord_cols, cont_cols, bin_cols, X_train_fit
    )

    k_safe = safe_k(SELECT_K, pre_scaled, X_train_fit)

    # ─────────────────────────────────────────────────────────
    # 10. TUNE SCALED MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning scaled models ...")
    scaled_pipelines = tune_models(
        scaled_models, pre_scaled, X_train_fit, y_train_fit, selector_k=k_safe
    )

    # ─────────────────────────────────────────────────────────
    # 11. TUNE UNSCALED MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning unscaled models ...")
    unscaled_pipelines = tune_models(
        unscaled_models, pre_unscaled, X_train_fit, y_train_fit, selector_k=k_safe
    )

    # ─────────────────────────────────────────────────────────
    # 12. TRAIN NEURAL NETWORK
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    mlp_pipe = train_mlp_pipeline(X_train_fit, y_train_fit, pre_scaled)

    all_pipelines = {**scaled_pipelines, **unscaled_pipelines, "NeuralNet": mlp_pipe}

    # ─────────────────────────────────────────────────────────
    # 13. EVALUATE ALL MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating all models ...")
    summary = evaluate_models(all_pipelines, X_train_fit, y_train_fit, X_test, y_test)

    print("\n" + "=" * 60)
    print("ALL MODELS SUMMARY (sorted by RMSE)")
    print("=" * 60)
    print(summary.to_string())

    # ─────────────────────────────────────────────────────────
    # 14. SELECT BEST MODEL
    # ─────────────────────────────────────────────────────────
    selected_name, selected_pipe = select_best_model(
        summary, all_pipelines, scaled_pipelines, unscaled_pipelines
    )

    # ─────────────────────────────────────────────────────────
    # 15. FINAL METRICS ON TEST SET
    # ─────────────────────────────────────────────────────────
    y_pred_sel   = selected_pipe.predict(X_test)
    y_pred_train = selected_pipe.predict(X_train_fit)

    metrics = compute_all_metrics(y_test.values, y_pred_sel)

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {selected_name}")
    print("=" * 60)
    for k, v in metrics.items():
        unit = " MW" if k in ("rmse", "mae") else ("%" if k in ("mape", "smape", "cvrmse") else "")
        print(f"  {k.upper():<10}: {v:.4f}{unit}")

    # ── Forecast vs Actual plot ───────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    plot_n = min(500, len(y_test))
    axes[0].plot(y_test.values[:plot_n],  label="Actual",   alpha=0.8)
    axes[0].plot(y_pred_sel[:plot_n],     label="Forecast", alpha=0.8)
    axes[0].set_title(f"{selected_name} — Forecast vs Actual (first {plot_n} hours)")
    axes[0].set_ylabel("Demand (MW)")
    axes[0].legend()

    residuals = y_test.values - y_pred_sel
    axes[1].plot(residuals[:plot_n], color="orange", alpha=0.7)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residuals (Actual - Forecast)")
    axes[1].set_ylabel("Error (MW)")

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "forecast_vs_actual.png"))
    plt.show()

    # ── Scatter plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test.values[:plot_n], y_pred_sel[:plot_n], alpha=0.3, s=10)
    mn = min(y_test.min(), y_pred_sel.min())
    mx = max(y_test.max(), y_pred_sel.max())
    ax.plot([mn, mx], [mn, mx], "r--", label="Perfect forecast")
    ax.set_xlabel("Actual Demand (MW)")
    ax.set_ylabel("Forecasted Demand (MW)")
    ax.set_title(f"{selected_name} — Scatter")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "forecast_scatter.png"))
    plt.show()

    # ─────────────────────────────────────────────────────────
    # 16. FORECAST ENGINE DECISIONS
    # ─────────────────────────────────────────────────────────
    decisions = forecast_engine(X_test, y_pred_sel)
    decision_counts = pd.Series(decisions).value_counts()

    print("\nFORECAST ENGINE DECISIONS")
    print(decision_counts)

    # ─────────────────────────────────────────────────────────
    # 17. COST-SENSITIVE EVALUATION
    # ─────────────────────────────────────────────────────────
    cost_result = cost_sensitive_evaluation(y_test.values, y_pred_sel)
    print("\nGRID IMBALANCE COST EVALUATION")
    for k, v in cost_result.items():
        print(f"  {k}: {v}")

    # ── Demand band accuracy ──────────────────────────────────
    band_acc = demand_band_accuracy(y_test.values, y_pred_sel)
    print(f"\nDemand Band Accuracy: {band_acc['band_accuracy_pct']:.2f}%")

    # ─────────────────────────────────────────────────────────
    # 18. REPEATED TS CV STABILITY
    # ─────────────────────────────────────────────────────────
    try:
        tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_FOLDS)
        rep_scores = cross_val_score(
            selected_pipe, X_train_fit, y_train_fit,
            scoring = "neg_root_mean_squared_error",
            cv      = tscv,
            n_jobs  = 1
        )
        logger.info("TS CV  |  mean_RMSE=%.2f  std=%.2f", -rep_scores.mean(), rep_scores.std())
    except Exception as e:
        logger.warning("TS CV stability check failed: %s", e)

    # ─────────────────────────────────────────────────────────
    # 19. FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────
    fi = compute_feature_importance(selected_pipe, X_train_fit, y_train_fit)
    if fi is not None:
        print("\nTOP FEATURE IMPORTANCES")
        print(fi.head(15))

    # ─────────────────────────────────────────────────────────
    # 20. SHAP EXPLAINABILITY
    # ─────────────────────────────────────────────────────────
    shap_result = compute_shap(selected_pipe, X_train_fit, X_test.head(500))

    # ─────────────────────────────────────────────────────────
    # 21. PSI DRIFT MONITORING
    # ─────────────────────────────────────────────────────────
    psi_scores = {}
    for col in cont_cols:
        if col in X_train_fit.columns and col in X_test.columns:
            psi_scores[col] = psi(X_train_fit[col].values, X_test[col].values)

    psi_df = (
        pd.Series(psi_scores)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "drift_score"})
    )
    psi_df.columns = ["feature", "drift_score"]
    psi_df.to_csv(os.path.join(MODEL_DIR, "feature_drift_report.csv"), index=False)

    print("\nTOP PSI (train vs test features)")
    print(psi_df.head(10).to_string())

    # ─────────────────────────────────────────────────────────
    # 22. MONITOR SCORES
    # ─────────────────────────────────────────────────────────
    monitor_df = build_monitor_scores(
        forecasts  = y_pred_sel,
        actuals    = y_test.values,
        timestamps = timestamps_test,
        decisions  = decisions,
    )
    monitor_df.to_csv(os.path.join(MODEL_DIR, "monitor_scores.csv"), index=False)
    logger.info("Monitor scores saved")

    # ─────────────────────────────────────────────────────────
    # 23. MODEL CARD + SAVE MODEL
    # ─────────────────────────────────────────────────────────
    train_gap = float(abs(
        float(np.sqrt(np.mean((y_train_fit.values - y_pred_train) ** 2)))
        - metrics["rmse"]
    ) / (metrics["rmse"] + 1e-9))

    model_card = build_model_card(
        selected_name    = selected_name,
        train_fit_size   = len(X_train_fit),
        cal_size         = len(X_cal),
        test_size        = len(X_test),
        train_date_range = (
            str(df_model["time"].iloc[0].date()) if "time" in df_model.columns else "N/A",
            str(df_model["time"].iloc[n_fit - 1].date()) if "time" in df_model.columns else "N/A"
        ),
        test_date_range  = (
            str(df_model["time"].iloc[n_fit + n_cal].date()) if "time" in df_model.columns else "N/A",
            str(df_model["time"].iloc[-1].date()) if "time" in df_model.columns else "N/A"
        ),
        metrics          = metrics,
        cost_result      = cost_result,
        decision_counts  = decision_counts.to_dict(),
        feature_order    = feature_order,
        selector_k       = k_safe,
        fi_dict          = fi.head(20).to_dict() if fi is not None else None,
        shap_dict        = shap_result.get("shap_top", {}) if shap_result else None,
        demand_band_acc  = band_acc,
    )

    # FIX: capture card_path → pass to save_model_and_card so registry stores it directly
    card_path  = save_model_card(model_card, MODEL_DIR, selected_name)
    model_path = save_model_and_card(
        selected_name, selected_pipe, model_card,
        model_card_path=card_path   # FIX: no brittle string parsing at load time
    )

    # ─────────────────────────────────────────────────────────
    # 24. CHALLENGER + MLFLOW
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running Champion vs Challenger comparison ...")

    challenger_result = run_challenger_comparison(
        challenger_name       = selected_name,
        challenger_rmse       = metrics["rmse"],
        challenger_r2         = metrics["r2"],
        challenger_gap        = train_gap,
        challenger_model_path = model_path,
        challenger_card_path  = card_path,   # FIX: direct path — no string parsing
    )

    print("\n" + "=" * 60)
    print(f"CHALLENGER RESULT: {challenger_result['decision']}")
    print(f"Reason: {challenger_result['reason']}")
    print("=" * 60)

    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{selected_name}"
    mlflow_log_run(run_name, selected_name, selected_pipe, model_card, X_train_sample=X_train_fit)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE  |  Best model: {selected_name}")
    print(f"  RMSE:   {metrics['rmse']:.2f} MW")
    print(f"  MAPE:   {metrics['mape']:.2f}%")
    print(f"  R²:     {metrics['r2']:.4f}")
    print(f"Challenger: {challenger_result['decision']}")
    print("=" * 60)

    return selected_name, selected_pipe, model_card


if __name__ == "__main__":
    start = time.time()
    name, model, card = run_training()
    logger.info("Finished in %.1fs", time.time() - start)