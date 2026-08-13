# ============================================================
# SARIMA / SARIMAX CHALLENGER — Energy Demand Forecasting
# ============================================================
# Classical statistical time-series baseline, run alongside the
# 13-model ML zoo (Ridge ... CatBoost, MLP) as an additional
# challenger inside the SAME champion-vs-challenger pipeline.
#
# WHY THIS EXISTS
#   Every one of the 13 existing models is a tabular regressor
#   (X → y): scaled linear, tree ensemble, or MLP. None of them
#   is a genuine seasonal time-series model. In demand forecasting
#   work, a SARIMA/SARIMAX baseline is the standard sanity check —
#   "does the fancy stuff actually beat classical time series?"
#   Its absence here was a real methodology gap, not just theory.
#
# WHY A SMALL EXOG SET (not the full 60+ engineered columns)
#   SARIMAX's own AR/MA/seasonal terms already model
#   autocorrelation. Feeding it demand_lag_1h, demand_lag_24h,
#   demand_roll_mean_24h etc. on top of that causes severe
#   multicollinearity with its own state and makes MLE fitting
#   unstable and slow. Real practice: SARIMA baselines get a
#   small, slowly-varying exog set (temperature + calendar
#   signals); the ML models get the full engineered feature bank.
#   See SARIMA_EXOG_COLS in src/config.py.
#
# HONEST SERVING LIMITATION (documented, not hidden)
#   SARIMAX forecasts by continuing from the end of its fitted
#   series — a natural fit for offline batch evaluation (predict
#   the whole contiguous X_test block in one call, done here).
#   It does NOT map cleanly onto the existing stateless single-hour
#   /predict REST endpoint, which expects an arbitrary caller-
#   supplied hour with its own lag features. If SARIMA ever wins
#   the champion gate, serving needs a stateful/batch forecast path
#   instead of (or alongside) the current endpoint — see README,
#   "SARIMA Serving Caveat".
# ============================================================

import logging
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

from src.config import (
    SARIMA_ORDER, SARIMA_SEASONAL_ORDER, SARIMA_MAXITER, SARIMA_EXOG_COLS
)

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except Exception:
    SARIMAX = None
    STATSMODELS_AVAILABLE = False


# ============================================================
# SARIMAX — sklearn-compatible wrapper
# ============================================================

class SARIMAXRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps statsmodels SARIMAX in the sklearn estimator API
    (fit/predict/score + get_params/set_params via BaseEstimator)
    so it can sit in the same `pipelines` dict as the ML regressors
    and pass through evaluate_models(), select_best_model(), and
    run_challenger_comparison() completely unmodified.

    fit(X, y):
        X must contain SARIMA_EXOG_COLS (extra columns ignored).
        Fits SARIMAX(y, exog=X[exog_cols], order=..., seasonal_order=...)

    predict(X):
        Two modes, chosen automatically by len(X):
          - len(X) == len(training series) → in-sample fitted values
            (get_prediction). This is what evaluate_models() calls to
            compute train_rmse — a true out-of-sample continuation
            forecast of that same length would be both extremely slow
            and statistically meaningless (SARIMA forecast uncertainty
            balloons far out).
          - otherwise → out-of-sample forecast continuing from the end
            of the fitted series for len(X) steps (get_forecast). This
            is the path used for X_test in evaluate_models() and for
            each held-out fold in cross_val_score().
    """

    def __init__(self, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
                 exog_cols=None, maxiter=SARIMA_MAXITER,
                 enforce_stationarity=False, enforce_invertibility=False):
        self.order                = order
        self.seasonal_order       = seasonal_order
        self.exog_cols            = exog_cols
        self.maxiter              = maxiter
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

    # ── helpers ─────────────────────────────────────────────
    def _get_exog_cols(self, X):
        cols = self.exog_cols if self.exog_cols is not None else SARIMA_EXOG_COLS
        return [c for c in cols if c in X.columns]

    def _exog_matrix(self, X, cols):
        if not cols:
            return None
        return X[cols].astype(float).fillna(0.0).values

    # ── sklearn API ─────────────────────────────────────────
    def fit(self, X, y):
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for SARIMAXRegressor "
                "(pip install statsmodels)"
            )

        y = np.asarray(y, dtype=float)
        self._fit_exog_cols_ = self._get_exog_cols(X)
        exog = self._exog_matrix(X, self._fit_exog_cols_)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                exog                  = exog,
                order                 = self.order,
                seasonal_order        = self.seasonal_order,
                enforce_stationarity  = self.enforce_stationarity,
                enforce_invertibility = self.enforce_invertibility,
            )
            self.results_ = model.fit(disp=False, maxiter=self.maxiter)

        self.n_train_ = len(y)
        logger.info(
            "SARIMAX fit done  |  order=%s  seasonal_order=%s  n=%d  AIC=%.1f",
            self.order, self.seasonal_order, self.n_train_, self.results_.aic
        )
        return self

    def predict(self, X):
        if not hasattr(self, "results_"):
            raise RuntimeError("SARIMAXRegressor.predict called before fit()")

        steps     = len(X)
        exog_cols = getattr(self, "_fit_exog_cols_", self._get_exog_cols(X))
        exog      = self._exog_matrix(X, exog_cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if steps == self.n_train_:
                # In-sample fit quality (train_rmse diagnostic) — see docstring.
                pred = self.results_.get_prediction(start=0, end=steps - 1, exog=exog)
            else:
                # Out-of-sample continuation forecast (test set / CV fold).
                pred = self.results_.get_forecast(steps=steps, exog=exog)

        preds = np.asarray(pred.predicted_mean, dtype=float)
        return np.maximum(preds, 0.0)   # demand can't be negative

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


# ============================================================
# BUILDER — used by training_pipeline.py
# ============================================================

def build_sarima_challenger(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    exog_cols: list = None,
) -> SARIMAXRegressor:
    """
    Fits and returns the SARIMA challenger, ready to drop straight
    into the same `all_pipelines` dict as the ML models in
    training_pipeline.run_training().

    Caller (training_pipeline) wraps this in try/except: if
    statsmodels isn't installed, or MLE fails to converge on a
    given dataset, training continues without SARIMA rather than
    crashing the whole run — same pattern already used for the
    optional LightGBM/CatBoost imports in model_tuning.py.
    """
    reg = SARIMAXRegressor(exog_cols=exog_cols)
    reg.fit(X_train, y_train)
    return reg
