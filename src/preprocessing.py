# ============================================================
# PREPROCESSING — Energy Demand Forecasting
# ============================================================

import numpy as np
import pandas as pd
import logging
import warnings

from typing import List, Tuple

from sklearn.base              import BaseEstimator, TransformerMixin
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.impute            import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

from src.config import CLIP_FOLD, SELECT_K

logger = logging.getLogger(__name__)


# ============================================================
# CLIPPER
# ============================================================

class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, fold: float = CLIP_FOLD):
        self.fold = fold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        q1  = np.nanquantile(X, 0.25, axis=0)
        q3  = np.nanquantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.fold * iqr
        self.upper_ = q3 + self.fold * iqr
        eps = 1e-9
        self.upper_ = np.where(self.upper_ == self.lower_, self.upper_ + eps, self.upper_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# NaN FILLER  ← ROOT CAUSE FIX
# ============================================================

class NaNFiller(BaseEstimator, TransformerMixin):
    """
    Fills NaN with per-column median. If median itself is NaN
    (all-NaN column in CV fold), fills with fill_value=0.

    WHY THIS EXISTS — two separate NaN sources in this dataset:

    Source 1 — continuous/skewed cols (e.g. generation solar at night):
      These columns are partially NaN. SimpleImputer(median) skips
      all-NaN folds silently → NaN survives → SelectKBest crashes.
      NaNFiller handles this explicitly.

    Source 2 — "binary" cols (e.g. generation nuclear, generation biomass):
      detect_feature_types() uses nunique(dropna=False). A column with
      only {0.0, NaN} has nunique=2 → classified as binary → previously
      passed through as "passthrough" → NaN passed directly to SelectKBest.
      FIX: bin_cols now go through NaNFiller pipeline, not "passthrough".

    fill_value=0 is the correct domain value: generation columns that are
    NaN in the dataset represent zero generation (not reported by operator).
    """

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.medians_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if not mask.any():
                continue
            median_val = self.medians_[i]
            fill = self.fill_value if np.isnan(median_val) else median_val
            X[mask, i] = fill
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# SAFE POWER TRANSFORMER
# ============================================================

class SafePowerTransformer(BaseEstimator, TransformerMixin):
    """
    Yeo-Johnson per column — skips zero-variance cols after clipping.
    Prevents scipy BracketError on generation columns (solar at night etc.)
    """
    MIN_VAR = 1e-6

    def __init__(self, method: str = "yeo-johnson", standardize: bool = False):
        self.method      = method
        self.standardize = standardize

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        self.transformers_  = []
        for i in range(X.shape[1]):
            col       = X[:, i]
            col_clean = col[~np.isnan(col)]
            if len(col_clean) > 1 and np.var(col_clean) > self.MIN_VAR:
                pt = PowerTransformer(method=self.method, standardize=self.standardize)
                try:
                    pt.fit(col_clean.reshape(-1, 1))
                    self.transformers_.append(pt)
                except Exception as e:
                    logger.warning("SafePowerTransformer col %d failed: %s → passthrough", i, e)
                    self.transformers_.append(None)
            else:
                self.transformers_.append(None)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        result = np.empty_like(X)
        for i, pt in enumerate(self.transformers_):
            col = X[:, i].reshape(-1, 1)
            if pt is not None:
                try:
                    result[:, i] = pt.transform(col).flatten()
                except Exception:
                    result[:, i] = X[:, i]
            else:
                result[:, i] = X[:, i]
        return result

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# SAFE K
# ============================================================

def safe_k(requested_k: int, preprocessor: ColumnTransformer, X_sample: pd.DataFrame) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = preprocessor.fit(X_sample)
    try:
        feat_names = fitted.get_feature_names_out()
    except Exception:
        arr = fitted.transform(X_sample)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        feat_names = [f"f{i}" for i in range(arr.shape[1])]
    k = min(requested_k, max(1, len(feat_names)))
    logger.info("safe_k: requested=%d  available=%d  using=%d", requested_k, len(feat_names), k)
    return k


# ============================================================
# PREPROCESSOR BUILDER
# ============================================================

def build_preprocessors(
    ord_cols:  List[str],
    cont_cols: List[str],
    bin_cols:  List[str],
    X_train:   pd.DataFrame,
    clip_fold: float = CLIP_FOLD
) -> Tuple[ColumnTransformer, ColumnTransformer, List[str]]:
    """
    Returns:
        preprocessor_scaled   — for linear/distance-based models
        preprocessor_unscaled — for tree models
        feature_order

    NaN STRATEGY (two sources fixed):
      1. Skewed/non-skewed cols: Clipper → NaNFiller → [Power] → Scale
         NaNFiller replaces passthrough-NaN issue from SimpleImputer.

      2. Binary cols: NaNFiller (fill_value=0) instead of "passthrough".
         CRITICAL FIX: detect_feature_types classifies {0.0, NaN} columns
         as binary (nunique(dropna=False)=2). "passthrough" kept their NaN
         which crashed SelectKBest. Now they go through NaNFiller.

      3. Ordinal cols: SimpleImputer(most_frequent) — these are integer
         calendar features, always have observed values, safe to use.
    """

    ord_cols  = [c for c in ord_cols  if c in X_train.columns]
    cont_cols = [c for c in cont_cols if c in X_train.columns]
    bin_cols  = [c for c in bin_cols  if c in X_train.columns]

    skewed = []
    for c in cont_cols:
        col_data = X_train[c].dropna()
        if len(col_data) > 0:
            try:
                if abs(float(col_data.skew())) > 0.8:
                    skewed.append(c)
            except Exception:
                pass
    non_skewed = [c for c in cont_cols if c not in skewed]

    logger.info("Skewed cols (%d): %s", len(skewed), skewed[:10])
    logger.info("Non-skewed cols (%d): %s", len(non_skewed), non_skewed[:10])

    def _ord_pipeline():
        return Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ord",    OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])

    def _bin_pipeline():
        # FIX: NaNFiller instead of "passthrough"
        # {0.0, NaN} binary cols must have NaN filled before SelectKBest
        return Pipeline([
            ("fill", NaNFiller(fill_value=0.0)),
        ])

    # ── Scaled pipelines ──────────────────────────────────────
    scaled_parts = []
    if ord_cols:
        scaled_parts.append(("ord", _ord_pipeline(), ord_cols))
    if skewed:
        scaled_parts.append(("skewed", Pipeline([
            ("clip",  Clipper(fold=clip_fold)),
            ("fill",  NaNFiller()),
            ("power", SafePowerTransformer(method="yeo-johnson", standardize=False)),
            ("scale", StandardScaler()),
        ]), skewed))
    if non_skewed:
        scaled_parts.append(("non_skew", Pipeline([
            ("clip",  Clipper(fold=clip_fold)),
            ("fill",  NaNFiller()),
            ("scale", StandardScaler()),
        ]), non_skewed))
    if bin_cols:
        scaled_parts.append(("bin", _bin_pipeline(), bin_cols))

    # ── Unscaled pipelines (trees) ────────────────────────────
    unscaled_parts = []
    if ord_cols:
        unscaled_parts.append(("ord", _ord_pipeline(), ord_cols))
    if skewed:
        unscaled_parts.append(("skewed", Pipeline([
            ("clip", Clipper(fold=clip_fold)),
            ("fill", NaNFiller()),
        ]), skewed))
    if non_skewed:
        unscaled_parts.append(("non_skew", Pipeline([
            ("clip", Clipper(fold=clip_fold)),
            ("fill", NaNFiller()),
        ]), non_skewed))
    if bin_cols:
        unscaled_parts.append(("bin", _bin_pipeline(), bin_cols))

    preprocessor_scaled = ColumnTransformer(
        transformers=scaled_parts,
        remainder="drop"
    )
    preprocessor_unscaled = ColumnTransformer(
        transformers=unscaled_parts,
        remainder="drop"
    )

    feature_order = ord_cols + skewed + non_skewed + bin_cols

    logger.info(
        "Preprocessors built  |  ord=%d  skewed=%d  non_skewed=%d  bin=%d  total=%d",
        len(ord_cols), len(skewed), len(non_skewed), len(bin_cols), len(feature_order)
    )

    return preprocessor_scaled, preprocessor_unscaled, feature_order


# ============================================================
# TIME-SERIES SPLIT
# ============================================================

def time_series_split(
    df:        pd.DataFrame,
    time_col:  str   = "time",
    test_frac: float = 0.15,
    cal_frac:  float = 0.10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n      = len(df)
    n_test = int(n * test_frac)
    n_cal  = int(n * cal_frac)
    n_fit  = n - n_test - n_cal

    train_fit = df.iloc[:n_fit].copy()
    cal       = df.iloc[n_fit: n_fit + n_cal].copy()
    test      = df.iloc[n_fit + n_cal:].copy()

    logger.info(
        "Time-series split  |  train_fit=%d [%s → %s]  cal=%d  test=%d [%s → %s]",
        len(train_fit),
        train_fit[time_col].min().date(),
        train_fit[time_col].max().date(),
        len(cal),
        len(test),
        test[time_col].min().date(),
        test[time_col].max().date()
    )
    return train_fit, cal, test