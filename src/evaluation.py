# ============================================================
# EVALUATION — Energy Demand Forecasting
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from typing import Dict, Optional, Tuple
from sklearn.pipeline          import Pipeline
from sklearn.model_selection   import TimeSeriesSplit, cross_val_score
from sklearn.inspection        import permutation_importance

from src.config  import RANDOM_STATE, N_JOBS, TIME_SERIES_CV_FOLDS, MODEL_DIR
from src.metrics import compute_all_metrics, r2_score as _r2

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_models(
    pipelines: Dict,
    X_train:   pd.DataFrame,
    y_train:   pd.Series,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
) -> pd.DataFrame:
    """
    Evaluates all pipelines on test set.
    Uses TimeSeriesSplit for CV (no shuffle).
    Returns summary DataFrame sorted by RMSE ascending.
    """
    rows = []
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_FOLDS)

    for name, pipe in pipelines.items():
        logger.info("Evaluating: %s", name)

        try:
            y_pred_train = pipe.predict(X_train)
            train_rmse   = float(np.sqrt(np.mean((y_train.values - y_pred_train) ** 2)))
        except Exception:
            train_rmse = None

        try:
            y_pred = pipe.predict(X_test)
        except Exception as e:
            logger.error("Predict failed for %s: %s", name, e)
            continue

        m = compute_all_metrics(y_test.values, y_pred)

        # ── CV RMSE ───────────────────────────────────────────
        try:
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                scoring = "neg_root_mean_squared_error",
                cv      = tscv,
                n_jobs  = 1
            )
            cv_mean = float(-cv_scores.mean())
            cv_std  = float(cv_scores.std())
        except Exception:
            cv_mean = None
            cv_std  = None

        gap = abs((train_rmse or 0) - m["rmse"]) / (m["rmse"] + 1e-9) if train_rmse else None

        rows.append({
            "model":          name,
            "train_rmse":     round(train_rmse, 2) if train_rmse else None,
            "test_rmse":      m["rmse"],
            "train_test_gap": round(gap, 4) if gap is not None else None,
            "cv_mean_rmse":   round(cv_mean, 2) if cv_mean else None,
            "cv_std_rmse":    round(cv_std, 2)  if cv_std  else None,
            "mae":            m["mae"],
            "mape":           m["mape"],
            "smape":          m["smape"],
            "r2":             m["r2"],
            "cvrmse":         m["cvrmse"],
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values("test_rmse", ascending=True)
        .reset_index(drop=True)
    )

    summary.to_csv(os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False)
    logger.info("Model summary saved to model_experiment_results.csv")

    return summary


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    summary:        pd.DataFrame,
    pipelines:      Dict,
    scaled_pipes:   Dict,
    unscaled_pipes: Dict,
) -> Tuple[str, Pipeline]:
    """
    Progressive filter — selects best generalizing model.
    Primary: lowest test RMSE
    Guard: R² >= threshold, CV std reasonable, gap <= 10%
    """

    def _filter(df, max_gap, max_cv_std, min_r2):
        mask = (
            df["train_test_gap"].notna() &
            df["cv_std_rmse"].notna() &
            (df["train_test_gap"] <= max_gap) &
            (df["cv_std_rmse"]   <= max_cv_std) &
            (df["r2"]            >= min_r2)
        )
        return df[mask].copy()

    thresholds = [
        (0.05, 500,   0.90),
        (0.08, 800,   0.85),
        (0.10, 1500,  0.80),
        (0.20, 99999, 0.00),
    ]

    candidates = pd.DataFrame()
    for gg, cvs, min_r2 in thresholds:
        candidates = _filter(summary, gg, cvs, min_r2)
        if not candidates.empty:
            logger.info("Candidates found with gap<=%.2f cv_std<=%.0f r2>=%.2f", gg, cvs, min_r2)
            break

    if candidates.empty:
        selected_name = summary.iloc[0]["model"]
    else:
        candidates    = candidates.sort_values(
            ["test_rmse", "r2", "cv_std_rmse"], ascending=[True, False, True]
        ).reset_index(drop=True)
        selected_name = candidates.iloc[0]["model"]

    logger.info("Selected model: %s", selected_name)

    selected_pipe = scaled_pipes.get(selected_name) or unscaled_pipes.get(selected_name)
    if selected_pipe is None and selected_name in pipelines:
        selected_pipe = pipelines[selected_name]
    if selected_pipe is None:
        raise RuntimeError(f"Selected model '{selected_name}' not found in any pipeline dict")

    return selected_name, selected_pipe


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(pipe: Pipeline, X_train, y_train) -> Optional[pd.Series]:
    """Permutation importance — model-agnostic."""
    try:
        result = permutation_importance(
            pipe, X_train, y_train,
            n_repeats      = 5,
            random_state   = RANDOM_STATE,
            scoring        = "neg_root_mean_squared_error",
            n_jobs         = N_JOBS
        )
        fi = pd.Series(
            result.importances_mean,
            index=X_train.columns
        ).sort_values(ascending=False)
        return fi
    except Exception as e:
        logger.warning("Permutation importance failed: %s", e)
        return None


def _get_feature_names(pipe, X_sample):
    try:
        pre       = pipe.named_steps["preprocessor"]
        raw_names = list(pre.get_feature_names_out())
        clean_names = []
        for n in raw_names:
            clean_names.append(n.split("__", 1)[1] if "__" in n else n)
        sel = pipe.named_steps.get("selector", None)
        if sel is not None:
            mask        = sel.get_support()
            clean_names = [n for n, m in zip(clean_names, mask) if m]
        return clean_names
    except Exception as e:
        logger.warning("Could not extract feature names: %s", e)
        return None


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def compute_shap(pipe: Pipeline, X_train, X_explain, max_explain: int = 500) -> Optional[dict]:

    if not SHAP_AVAILABLE:
        logger.info("SHAP not installed — skipping")
        return None

    # Non-Pipeline estimators (e.g. the SARIMA challenger, which is a plain
    # sklearn-compatible wrapper around statsmodels, not a
    # preprocessor→selector→classifier Pipeline) don't have named_steps and
    # aren't tree/kernel-explainable the same way. Skip cleanly rather than
    # crashing training if SARIMA ever wins the champion gate.
    if not hasattr(pipe, "named_steps"):
        logger.info("SHAP skipped — '%s' is not a sklearn Pipeline (no named_steps)",
                    type(pipe).__name__)
        return None

    # Pipeline step is named "classifier" (just a key — works for any estimator type)
    clf = pipe.named_steps["classifier"]
    pre = pipe.named_steps["preprocessor"]
    sel = pipe.named_steps.get("selector", None)

    try:
        X_pre = pre.transform(X_train)
        X_ex  = sel.transform(pre.transform(X_explain)) if sel is not None else pre.transform(X_explain)
        X_ex  = np.asarray(X_ex)[:max_explain]

        names = _get_feature_names(pipe, X_train)

        tree_types = (
            "RandomForestRegressor", "ExtraTreesRegressor",
            "GradientBoostingRegressor", "XGBRegressor",
            "LGBMRegressor", "CatBoostRegressor",
            "HistGradientBoostingRegressor", "DecisionTreeRegressor",
            "AdaBoostRegressor"
        )

        if type(clf).__name__ in tree_types:
            explainer = shap.TreeExplainer(clf)
            try:
                vals = np.array(explainer(X_ex).values)
            except Exception:
                vals = np.array(explainer.shap_values(X_ex))
        else:
            X_tr    = sel.transform(X_pre) if sel is not None else X_pre
            bg      = shap.sample(X_tr, min(200, len(X_tr)))
            explainer = shap.KernelExplainer(clf.predict, bg)
            vals      = np.array(explainer.shap_values(X_ex, nsamples=100))

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        mean_abs   = np.abs(vals).mean(axis=0)
        feat_names = names[:mean_abs.shape[0]] if names else [f"f{i}" for i in range(mean_abs.shape[0])]
        fi_series  = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)

        logger.info("SHAP done  |  explainer=%s", type(explainer).__name__)
        return {"shap_top": fi_series.head(20).to_dict(), "explainer": type(explainer).__name__}

    except Exception as e:
        logger.exception("SHAP failed: %s", e)
        return None


# ============================================================
# SAVE MODEL
# ============================================================

def save_model_and_card(
    selected_name: str,
    pipe:          Pipeline,
    model_card:    dict,
    version:       str = "v1",
    model_card_path: str = ""   # FIX: accept card path, store directly in registry
) -> str:
    """
    Saves model .joblib and updates registry.

    FIX vs original: model_card_path is now accepted and written directly
    into latest_model.json — eliminates brittle filename string-parsing to
    reconstruct the card path.
    """
    model_path = os.path.join(MODEL_DIR, f"energy_model_{selected_name}_{version}.joblib")
    joblib.dump(pipe, model_path)

    # Registry stores card path directly — no string parsing needed at load time
    registry = {
        "model_name":      os.path.basename(model_path),
        "model_card_path": model_card_path or "",
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("Model saved   → %s", model_path)
    return model_path


# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_run(
    run_name:       str,
    selected_name:  str,
    pipe:           Pipeline,
    model_card:     dict,
    X_train_sample: pd.DataFrame = None
):
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not installed — skipping")
        return

    try:
        # FIX: newer MLflow versions put the plain filesystem tracking
        # backend ("./mlruns") into maintenance mode and refuse to log new
        # runs against it (raises MlflowException). Point at a local sqlite
        # DB instead — still fully local, no server needed, just a
        # supported backend. Safe to call every run; MLflow reuses the
        # existing mlflow.db file if present.
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("energy_demand_experiments")

        metrics_dict = model_card.get("metrics", model_card)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", selected_name)
            mlflow.log_param("selector_k", model_card.get("pipeline_config", {}).get("selector_k", "?"))

            mlflow.log_metric("test_rmse",  float(metrics_dict.get("rmse",   0)))
            mlflow.log_metric("test_mae",   float(metrics_dict.get("mae",    0)))
            mlflow.log_metric("test_mape",  float(metrics_dict.get("mape",   0)))
            mlflow.log_metric("test_r2",    float(metrics_dict.get("r2",     0)))
            mlflow.log_metric("test_cvrmse",float(metrics_dict.get("cvrmse", 0)))

            try:
                log_kwargs = {
                    "sk_model":         pipe,
                    "name":             "model",
                    "pip_requirements": [
                        "scikit-learn", "lightgbm", "xgboost",
                        "pandas", "numpy",
                    ],
                }
                if X_train_sample is not None:
                    log_kwargs["input_example"] = X_train_sample.iloc[:5].astype(float)
                mlflow.sklearn.log_model(**log_kwargs)
            except Exception as e:
                logger.warning("mlflow log_model failed: %s", e)

        logger.info("MLflow run logged: %s", run_name)

    except Exception as e:
        logger.exception("MLflow logging failed: %s", e)