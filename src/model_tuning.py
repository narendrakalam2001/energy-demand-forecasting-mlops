# ============================================================
# MODEL TUNING — Energy Demand Forecasting
# ============================================================
# Regression models evaluated:
#   Scaled:   Ridge, Lasso, ElasticNet, SVR, KNN, MLP
#   Unscaled: RandomForest, ExtraTrees, GradientBoosting,
#             XGBoost, LightGBM, CatBoost, HistGradientBoosting
#
# TimeSeriesSplit CV — respects temporal ordering.
# No data shuffling. No SMOTE (regression task).
# ============================================================

import numpy as np
import logging

from typing import Dict, List, Tuple

from sklearn.base              import BaseEstimator
from sklearn.compose           import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection   import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline          import Pipeline

from sklearn.linear_model  import Ridge, Lasso, ElasticNet
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.svm           import SVR
from sklearn.ensemble      import (RandomForestRegressor, GradientBoostingRegressor,
                                   ExtraTreesRegressor, HistGradientBoostingRegressor,
                                   AdaBoostRegressor)
from sklearn.tree          import DecisionTreeRegressor
from xgboost               import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    LGBMRegressor  = None
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostRegressor  = None
    CATBOOST_AVAILABLE = False

from src.config        import RANDOM_STATE, N_JOBS, RANDOM_SEARCH_ITERS, SELECT_K, TIME_SERIES_CV_FOLDS
from src.preprocessing import safe_k

logger = logging.getLogger(__name__)


# ============================================================
# HELPER
# ============================================================

def _compute_n_iter(param_dist: dict, budget: int) -> int:
    if not param_dist:
        return 1
    prod = 1
    for v in param_dist.values():
        try:
            prod *= len(v)
        except TypeError:
            prod *= budget
    return min(budget, max(1, prod))


# ============================================================
# SCALED MODELS — need StandardScaler
# ============================================================

scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "Ridge": (
        Ridge(random_state=RANDOM_STATE),
        {
            "classifier__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        }
    ),

    "Lasso": (
        Lasso(random_state=RANDOM_STATE, max_iter=5000),
        {
            "classifier__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        }
    ),

    "ElasticNet": (
        ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
        {
            "classifier__alpha":   [0.001, 0.01, 0.1, 1.0],
            "classifier__l1_ratio":[0.1, 0.3, 0.5, 0.7, 0.9],
        }
    ),

    "KNN": (
        KNeighborsRegressor(n_jobs=N_JOBS),
        {
            "classifier__n_neighbors": [5, 10, 20],
            "classifier__weights":     ["uniform", "distance"],
        }
    ),
}

# SVR too slow for large time-series data; skip unless needed
# "SVR": (SVR(), {"classifier__C": [0.1, 1, 10], "classifier__epsilon": [0.01, 0.1]})


# ============================================================
# UNSCALED MODELS — tree-based
# ============================================================

unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "DecisionTree": (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {
            "classifier__max_depth":        [10, 20, 30, None],
            "classifier__min_samples_leaf": [1, 2, 5],
        }
    ),

    "RandomForest": (
        RandomForestRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":     [100, 200],
            "classifier__max_depth":        [None, 20],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__max_features":     ["sqrt", 0.5],
        }
    ),

    "ExtraTrees": (
        ExtraTreesRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth":    [None, 20],
            "classifier__max_features": ["sqrt", 0.5],
        }
    ),

    "GradientBoosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1, 0.15],
            "classifier__max_depth":     [3, 4, 5],
            "classifier__subsample":     [0.8, 1.0],
        }
    ),

    "HistGBM": (
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "classifier__max_iter":      [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [None, 10, 20],
            "classifier__l2_regularization": [0.0, 0.1, 1.0],
        }
    ),

    "XGBoost": (
        XGBRegressor(
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0
        ),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1, 0.15],
            "classifier__max_depth":     [4, 6, 8],
            "classifier__subsample":     [0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 1.0],
        }
    ),
}

if LGBM_AVAILABLE:
    unscaled_models["LightGBM"] = (
        LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=N_JOBS),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [-1, 10, 20],
            "classifier__num_leaves":    [31, 63, 127],
            "classifier__subsample":     [0.8, 1.0],
        }
    )

if CATBOOST_AVAILABLE:
    unscaled_models["CatBoost"] = (
        CatBoostRegressor(verbose=0, random_state=RANDOM_STATE),
        {
            "classifier__iterations":    [100, 200, 300],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__depth":         [4, 6, 8],
        }
    )


# ============================================================
# TUNE MODELS — TimeSeriesSplit CV
# ============================================================

def tune_models(
    models:       Dict[str, Tuple[BaseEstimator, dict]],
    preprocessor: ColumnTransformer,
    X_train:      "pd.DataFrame",
    y_train:      "pd.Series",
    selector_k:   int = SELECT_K
) -> Dict[str, Pipeline]:
    """
    For each model:
      preprocessor → SelectKBest(f_regression) → regressor

    Tuned with RandomizedSearchCV using TimeSeriesSplit.
    TimeSeriesSplit is critical — preserves temporal ordering.

    Returns dict of {model_name: best_pipeline}.
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    final_pipelines: Dict[str, SkPipeline] = {}

    k_safe = safe_k(selector_k, preprocessor, X_train)
    logger.info("Selector k set to %d (requested %d)", k_safe, selector_k)

    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_FOLDS)

    for name, (reg, param_dist) in models.items():

        logger.info("Tuning: %s", name)

        pipe = SkPipeline([
            ("preprocessor", preprocessor),
            ("selector",     SelectKBest(f_regression, k=k_safe)),
            ("classifier",   reg),
        ])

        n_iter = _compute_n_iter(param_dist, RANDOM_SEARCH_ITERS)

        search = RandomizedSearchCV(
            pipe,
            param_distributions = param_dist,
            n_iter              = n_iter,
            scoring             = "neg_root_mean_squared_error",
            cv                  = tscv,
            n_jobs              = N_JOBS,
            random_state        = RANDOM_STATE,
            verbose             = 0
        )

        try:
            search.fit(X_train, y_train)
            logger.info("%s  best RMSE=%.2f", name, -search.best_score_)
            final_pipelines[name] = search.best_estimator_
        except Exception as e:
            logger.error("Tuning failed for %s: %s", name, e)

    return final_pipelines


# ============================================================
# NEURAL NETWORK MLP — trained separately
# ============================================================

def train_mlp_pipeline(
    X_train:      "pd.DataFrame",
    y_train:      "pd.Series",
    preprocessor: ColumnTransformer
) -> Pipeline:
    """
    MLP Regressor trained outside CV search (too slow for TS CV).
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline       import Pipeline as SkPipeline

    logger.info("Training Neural Network (MLP Regressor) ...")

    pipe = SkPipeline([
        ("preprocessor", preprocessor),
        ("classifier",   MLPRegressor(
            hidden_layer_sizes  = (256, 128, 64),
            activation          = "relu",
            solver              = "adam",
            alpha               = 0.001,
            batch_size          = 512,
            learning_rate       = "adaptive",
            max_iter            = 100,
            early_stopping      = True,
            validation_fraction = 0.1,
            n_iter_no_change    = 10,
            random_state        = RANDOM_STATE
        ))
    ])

    pipe.fit(X_train, y_train)
    logger.info("MLP training done")
    return pipe