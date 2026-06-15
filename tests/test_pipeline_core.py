# ============================================================
# PYTEST UNIT TESTS — Energy Demand Forecasting
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#            pytest tests/ -v --cov=src --cov-report=term-missing
#
# Tests cover:
#   Clipper, build_preprocessors, detect_feature_types,
#   detect_leakage, add_time_features, add_lag_features,
#   rmse, mae, mape, smape, r2_score, cvrmse, psi,
#   cost_sensitive_evaluation, demand_band_accuracy,
#   get_demand_band, score_forecast, config thresholds,
#   SafePowerTransformer, dual imputer NaN fix
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd

from src.preprocessing   import Clipper, SafePowerTransformer, build_preprocessors
from src.data_loader     import (detect_feature_types, add_time_features,
                                  add_lag_features, get_demand_band)
from src.leakage_check   import detect_leakage, check_future_lag_leakage
from src.metrics         import (rmse, mae, mape, smape, r2_score, cvrmse,
                                  psi, cost_sensitive_evaluation, demand_band_accuracy)
from src.forecast_engine import get_demand_band as engine_get_band, score_forecast
from src.config          import (
    DEMAND_BANDS, PSI_MODERATE, PSI_HIGH,
    MIN_RMSE_IMPROVEMENT, MIN_R2_THRESHOLD, MAX_GENERALIZATION_GAP,
    PEAK_DEMAND_THRESHOLD, LOW_DEMAND_THRESHOLD, PRICE_SPIKE_THRESHOLD
)


# ============================================================
# CLIPPER TESTS  (6 tests)
# ============================================================

class TestClipper:

    def test_fit_transform_shape(self):
        """Output shape must match input shape."""
        X = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape == X.shape

    def test_clips_outliers(self):
        """Extreme values must be clipped."""
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).max() < 9999.0

    def test_no_change_on_normal_data(self):
        """Values within IQR range must not be changed."""
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        np.testing.assert_array_almost_equal(X, clip.transform(X), decimal=3)

    def test_1d_input(self):
        """1D array should be handled without error."""
        X = np.array([1.0, 2.0, 3.0, 1000.0])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape[0] == 4

    def test_get_feature_names_out(self):
        """get_feature_names_out must return array of correct length."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        clip = Clipper()
        clip.fit(X)
        names = clip.get_feature_names_out(["a", "b"])
        assert list(names) == ["a", "b"]

    def test_fit_on_train_applied_to_test(self):
        """Clipper fitted on train must use training bounds on test."""
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        X_test  = np.array([[100.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X_train)
        clipped = clip.transform(X_test)
        assert clipped[0, 0] < 100.0


# ============================================================
# SAFE POWER TRANSFORMER TESTS  (3 tests)
# ============================================================

class TestSafePowerTransformer:

    def test_fit_transform_no_crash_on_normal_data(self):
        """Must not raise on normal-variance data."""
        X = np.random.uniform(20000, 45000, (100, 3))
        spt = SafePowerTransformer()
        spt.fit(X)
        out = spt.transform(X)
        assert out.shape == X.shape

    def test_zero_variance_column_passthrough(self):
        """All-zero column (zero variance) must not crash — passthrough."""
        X = np.ones((50, 2))   # both cols zero variance
        spt = SafePowerTransformer()
        spt.fit(X)
        out = spt.transform(X)
        assert not np.isnan(out).any()

    def test_get_feature_names_out(self):
        X = np.random.uniform(0, 1, (30, 2))
        spt = SafePowerTransformer()
        spt.fit(X)
        names = spt.get_feature_names_out(["a", "b"])
        assert list(names) == ["a", "b"]


# ============================================================
# PREPROCESSOR TESTS  (5 tests)
# ============================================================

class TestBuildPreprocessors:

    def _sample_df(self):
        return pd.DataFrame({
            "hour":        list(range(24)) * 5,
            "demand_lag":  np.random.uniform(20000, 45000, 120),
            "temp":        np.random.uniform(5, 35, 120),
            "is_weekend":  [0, 1] * 60,
            "season":      [0, 1, 2, 3] * 30,
        })

    def test_returns_three_outputs(self):
        """build_preprocessors must return (pre_scaled, pre_unscaled, feat_order)."""
        df = self._sample_df()
        result = build_preprocessors(["season"], ["demand_lag", "temp"], ["is_weekend"], df)
        assert len(result) == 3

    def test_feature_order_coverage(self):
        """feature_order must contain all input columns."""
        df = self._sample_df()
        ord_cols  = ["season"]
        cont_cols = ["demand_lag", "temp"]
        bin_cols  = ["is_weekend"]
        _, _, feat_order = build_preprocessors(ord_cols, cont_cols, bin_cols, df)
        for col in ord_cols + cont_cols + bin_cols:
            assert col in feat_order

    def test_scaled_preprocessor_no_nan(self):
        """Scaled preprocessor must produce output without NaN."""
        df = self._sample_df()
        pre_scaled, _, _ = build_preprocessors(["season"], ["demand_lag", "temp"], ["is_weekend"], df)
        out = pre_scaled.fit_transform(df)
        assert not np.isnan(out).any()

    def test_unscaled_preprocessor_no_nan(self):
        """Unscaled preprocessor must produce output without NaN."""
        df = self._sample_df()
        _, pre_unscaled, _ = build_preprocessors(["season"], ["demand_lag", "temp"], ["is_weekend"], df)
        out = pre_unscaled.fit_transform(df)
        assert not np.isnan(out).any()

    def test_all_nan_column_no_crash(self):
        """
        Dual-imputer fix: all-NaN column must not crash preprocessor.
        This is the exact scenario from the production crash (solar generation
        columns are all-NaN during certain CV folds at night).
        """
        df = self._sample_df()
        df["all_nan_col"] = np.nan   # simulate solar=NaN-at-night column
        cont_cols = ["demand_lag", "temp", "all_nan_col"]
        pre_scaled, _, _ = build_preprocessors([], cont_cols, [], df)
        out = pre_scaled.fit_transform(df)
        assert not np.isnan(out).any(), "All-NaN column must not leave NaN in output"


# ============================================================
# TIME FEATURE TESTS  (4 tests)
# ============================================================

class TestAddTimeFeatures:

    def _sample_df(self):
        times = pd.date_range("2017-01-01", periods=48, freq="h", tz="UTC")
        return pd.DataFrame({"time": times, "total load actual": np.random.uniform(25000, 40000, 48)})

    def test_hour_column_added(self):
        df = add_time_features(self._sample_df())
        assert "hour" in df.columns
        assert df["hour"].between(0, 23).all()

    def test_cyclical_encoding_range(self):
        """Sin/cos features must be in [-1, 1]."""
        df = add_time_features(self._sample_df())
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_is_weekend_correct(self):
        df = add_time_features(self._sample_df())
        first_row = df.iloc[0]
        assert first_row["is_weekend"] in (0, 1)

    def test_season_range(self):
        df = add_time_features(self._sample_df())
        assert df["season"].between(0, 3).all()


# ============================================================
# LAG FEATURE TESTS  (3 tests)
# ============================================================

class TestAddLagFeatures:

    def _sample_df(self, n=200):
        return pd.DataFrame({
            "total load actual": np.random.uniform(25000, 40000, n)
        })

    def test_lag_columns_created(self):
        df = add_lag_features(self._sample_df())
        assert "demand_lag_1h" in df.columns
        assert "demand_lag_24h" in df.columns
        assert "demand_lag_168h" in df.columns

    def test_lag_1h_correct_value(self):
        """Lag 1h at row i must equal row i-1 actual."""
        df = add_lag_features(self._sample_df(50))
        actual_at_10  = df["total load actual"].iloc[10]
        lag1_at_11    = df["demand_lag_1h"].iloc[11]
        assert abs(actual_at_10 - lag1_at_11) < 1e-6

    def test_first_rows_are_nan(self):
        """First row of lag_1h must be NaN (nothing before it)."""
        df = add_lag_features(self._sample_df())
        assert pd.isna(df["demand_lag_1h"].iloc[0])


# ============================================================
# FEATURE TYPE DETECTION TESTS  (3 tests)
# ============================================================

class TestDetectFeatureTypes:

    def test_binary_detected(self):
        df = pd.DataFrame({
            "is_weekend":       [0, 1, 0, 1, 0, 1],
            "total load actual":[25000, 30000, 28000, 32000, 27000, 31000]
        })
        _, _, bin_cols = detect_feature_types(df)
        assert "is_weekend" in bin_cols

    def test_target_excluded(self):
        df = pd.DataFrame({
            "demand": np.random.uniform(20000, 45000, 50),
            "total load actual": np.random.uniform(20000, 45000, 50)
        })
        ord_cols, cont_cols, bin_cols = detect_feature_types(df)
        assert "total load actual" not in ord_cols + cont_cols + bin_cols

    def test_continuous_detected(self):
        df = pd.DataFrame({
            "demand_lag_1h":    np.random.uniform(20000, 45000, 100),
            "total load actual":np.random.uniform(20000, 45000, 100)
        })
        _, cont_cols, _ = detect_feature_types(df)
        assert "demand_lag_1h" in cont_cols


# ============================================================
# LEAKAGE CHECK TESTS  (4 tests)
# ============================================================

class TestDetectLeakage:

    def test_catches_identical_feature(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y = pd.Series(      [1.0, 2.0, 3.0, 4.0, 5.0])
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) > 0

    def test_no_false_positives_on_clean_data(self):
        np.random.seed(42)
        X = pd.DataFrame({"hour": np.random.randint(0, 24, 200).astype(float)})
        y = pd.Series(np.random.uniform(20000, 45000, 200))
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) == 0

    def test_catches_known_leaky_columns(self):
        """total load forecast is a known leaky col — must be flagged."""
        X = pd.DataFrame({"total load forecast": np.random.uniform(25000, 40000, 100)})
        y = pd.Series(np.random.uniform(25000, 40000, 100))
        warnings = detect_leakage(X, y)
        assert any("total load forecast" in w for w in warnings)

    def test_empty_dataframe_no_crash(self):
        X = pd.DataFrame()
        y = pd.Series([25000.0, 30000.0, 28000.0])
        warnings = detect_leakage(X, y)
        assert isinstance(warnings, list)


# ============================================================
# REGRESSION METRIC TESTS  (8 tests)
# ============================================================

class TestRegressionMetrics:

    def _perfect(self):
        y = np.array([25000.0, 30000.0, 35000.0, 40000.0])
        return y, y.copy()

    def test_rmse_perfect(self):
        y, p = self._perfect()
        assert rmse(y, p) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_nonzero(self):
        y = np.array([30000.0, 35000.0])
        p = np.array([31000.0, 34000.0])
        assert rmse(y, p) == pytest.approx(1000.0, rel=1e-4)

    def test_mae_perfect(self):
        y, p = self._perfect()
        assert mae(y, p) == pytest.approx(0.0, abs=1e-6)

    def test_mape_range(self):
        y = np.array([30000.0, 35000.0, 28000.0])
        p = np.array([31000.0, 34000.0, 29000.0])
        m = mape(y, p)
        assert isinstance(m, float)
        assert 0.0 <= m <= 100.0

    def test_smape_symmetric(self):
        """sMAPE must be symmetric: swap y and p → same value."""
        y = np.array([30000.0, 35000.0])
        p = np.array([28000.0, 37000.0])
        assert abs(smape(y, p) - smape(p, y)) < 1e-4

    def test_r2_perfect(self):
        y, p = self._perfect()
        assert r2_score(y, p) == pytest.approx(1.0, abs=1e-6)

    def test_r2_mean_predictor(self):
        """Predicting mean → R² == 0."""
        y = np.array([25000.0, 30000.0, 35000.0, 40000.0])
        p = np.full_like(y, y.mean())
        assert r2_score(y, p) == pytest.approx(0.0, abs=1e-4)

    def test_cvrmse_units(self):
        """CVRMSE = RMSE / mean × 100. Must be reasonable %."""
        y = np.array([30000.0, 35000.0, 32000.0])
        p = np.array([31000.0, 34000.0, 33000.0])
        cv = cvrmse(y, p)
        assert 0 < cv < 50


# ============================================================
# PSI TESTS  (3 tests)
# ============================================================

class TestPSI:

    def test_identical_distributions(self):
        x = np.random.normal(30000, 5000, 500)
        assert psi(x, x) < 0.05

    def test_shifted_distribution_higher_psi(self):
        rng = np.random.RandomState(42)
        ref = rng.normal(30000, 5000, 1000)
        new = rng.normal(38000, 5000, 1000)
        assert psi(ref, new) > psi(ref, ref)

    def test_disjoint_distribution_high_psi(self):
        rng = np.random.RandomState(0)
        ref = rng.normal(25000, 2000, 500)
        new = rng.normal(45000, 2000, 500)
        score = psi(ref, new)
        assert score > 0.20, f"Disjoint demand distributions should have PSI >> 0.20, got {score:.4f}"


# ============================================================
# COST EVALUATION TESTS  (3 tests)
# ============================================================

class TestCostEvaluation:

    def test_returns_dict_with_required_keys(self):
        y = np.array([30000.0, 35000.0, 40000.0, 45000.0])
        p = np.array([31000.0, 34000.0, 39000.0, 46000.0])
        result = cost_sensitive_evaluation(y, p)
        for key in ("total_forecast_error_mw", "total_estimated_cost_eur",
                    "under_forecast_count", "over_forecast_count"):
            assert key in result

    def test_perfect_forecast_zero_cost(self):
        y = np.array([30000.0, 35000.0, 40000.0])
        result = cost_sensitive_evaluation(y, y)
        assert result["total_estimated_cost_eur"] == pytest.approx(0.0, abs=1e-3)

    def test_peak_hours_cost_more(self):
        """
        Same absolute error during peak hours must cost more than off-peak.
        FIX: Use equal-sized errors, only vary whether y_true is above peak threshold.
        Original test used 'error' array as y_pred — giving a 3 MW error for peak
        (|45000-42000|) vs 22 MW for offpeak (|20000-42000|), so offpeak was always
        more expensive regardless of the peak multiplier. Correct test: same error MW.
        """
        error_mw = 1000.0

        # Peak hour: actual is above peak threshold (40000 MW)
        y_peak_actual = np.array([45000.0])
        y_peak_pred   = np.array([45000.0 - error_mw])   # under-forecast by 1000 MW

        # Off-peak hour: actual is below peak threshold
        y_off_actual  = np.array([25000.0])
        y_off_pred    = np.array([25000.0 - error_mw])   # same 1000 MW error

        cost_peak    = cost_sensitive_evaluation(y_peak_actual, y_peak_pred)
        cost_offpeak = cost_sensitive_evaluation(y_off_actual,  y_off_pred)

        assert cost_peak["total_estimated_cost_eur"] > cost_offpeak["total_estimated_cost_eur"], (
            f"Peak cost {cost_peak['total_estimated_cost_eur']} should exceed "
            f"offpeak cost {cost_offpeak['total_estimated_cost_eur']} for same error magnitude"
        )


# ============================================================
# DEMAND BAND ACCURACY TESTS  (2 tests)
# ============================================================

class TestDemandBandAccuracy:

    def test_perfect_band_accuracy(self):
        y = np.array([20000.0, 30000.0, 42000.0, 60000.0])
        result = demand_band_accuracy(y, y)
        assert result["band_accuracy_pct"] == pytest.approx(100.0)

    def test_wrong_band_lower_accuracy(self):
        y_true = np.array([45000.0] * 10)   # all HIGH
        y_pred = np.array([20000.0] * 10)   # all predicted LOW
        result = demand_band_accuracy(y_true, y_pred)
        assert result["band_accuracy_pct"] < 100.0


# ============================================================
# FORECAST ENGINE TESTS  (6 tests)
# ============================================================

class TestForecastEngine:

    def test_get_demand_band_low(self):
        assert engine_get_band(15000.0) == "LOW"

    def test_get_demand_band_normal(self):
        assert engine_get_band(30000.0) == "NORMAL"

    def test_get_demand_band_high(self):
        assert engine_get_band(42000.0) == "HIGH"

    def test_get_demand_band_peak(self):
        assert engine_get_band(60000.0) == "PEAK"

    def test_score_forecast_normal_operation(self):
        row    = {"price_actual": 50.0}
        result = score_forecast(row, forecast_mw=30000.0)
        assert result["dispatch_action"] == "NORMAL_OPERATION"
        assert result["demand_band"] == "NORMAL"

    def test_score_forecast_activate_reserve(self):
        row    = {"price_actual": 50.0}
        result = score_forecast(row, forecast_mw=45000.0)
        assert result["dispatch_action"] == "ACTIVATE_RESERVE"
        assert result["rule_triggered"] == "PEAK_DEMAND_THRESHOLD"


# ============================================================
# CONFIG TESTS  (4 tests)
# ============================================================

class TestConfig:

    def test_psi_thresholds_ordered(self):
        assert PSI_MODERATE < PSI_HIGH

    def test_challenger_gates_reasonable(self):
        assert MIN_RMSE_IMPROVEMENT  > 0
        assert 0.0 < MIN_R2_THRESHOLD < 1.0
        assert 0.0 < MAX_GENERALIZATION_GAP < 1.0

    def test_demand_band_boundaries_cover_full_range(self):
        """LOW band must start at 0, PEAK must end above realistic demand."""
        band_lows  = [v[0] for v in DEMAND_BANDS.values()]
        band_highs = [v[1] for v in DEMAND_BANDS.values()]
        assert min(band_lows)  == 0
        assert max(band_highs) > 50000

    def test_business_rule_thresholds_positive(self):
        assert PEAK_DEMAND_THRESHOLD   > 0
        assert LOW_DEMAND_THRESHOLD    > 0
        assert PRICE_SPIKE_THRESHOLD   > 0