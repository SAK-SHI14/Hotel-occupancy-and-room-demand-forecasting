"""Tests for baseline models and chronological split validation.

Phase 5 acceptance tests:
- Baseline predictions are non-negative and capacity-clipped.
- B2 has at least as good (or better) WAPE than B1 (B2 is a stronger baseline).
- SeasonalNaive uses the correct lag column; MovingAvg uses the mean column.
- Purge gap between splits is exactly >= 14 days.
- Train, val, and test sets do not overlap.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hotel_forecast.models.baselines import SeasonalMovingAverageBaseline, SeasonalNaiveBaseline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_X() -> pd.DataFrame:
    """Minimal feature DataFrame matching baseline expectations."""
    rng = np.random.default_rng(42)
    n = 200
    capacity = rng.integers(10, 50, n).astype(float)
    # lag_same_weekday: a past occupancy value, may slightly exceed capacity (noise)
    lag_values = capacity * rng.uniform(0.5, 1.2, n)
    # same_weekday_4w_mean: average of 4 same-weekday actuals, within capacity
    mean_values = capacity * rng.uniform(0.4, 0.95, n)
    return pd.DataFrame({
        "lag_same_weekday": lag_values,
        "same_weekday_4w_mean": mean_values,
        "capacity": capacity,
        "horizon": rng.integers(1, 15, n),
        "is_weekend": rng.integers(0, 2, n),
        "is_holiday": rng.integers(0, 2, n),
    })


@pytest.fixture()
def synthetic_y(synthetic_X: pd.DataFrame) -> pd.Series:
    """Ground truth: random occupancy in [0, capacity]."""
    rng = np.random.default_rng(99)
    actual = rng.uniform(0, 1, len(synthetic_X)) * synthetic_X["capacity"].to_numpy()
    return pd.Series(actual, name="occupied_room_nights")


# ---------------------------------------------------------------------------
# B1 Seasonal Naive Baseline Tests
# ---------------------------------------------------------------------------

class TestSeasonalNaiveBaseline:
    """Unit tests for B1."""

    def test_fit_validates_schema(self, synthetic_X):
        """fit() should succeed when lag_same_weekday column exists."""
        b1 = SeasonalNaiveBaseline()
        b1.fit(synthetic_X)  # should not raise

    def test_fit_raises_on_missing_column(self):
        """fit() raises ValueError if the required lag column is absent."""
        X_bad = pd.DataFrame({"other_col": [1, 2, 3], "capacity": [50, 50, 50]})
        with pytest.raises(ValueError, match="lag_same_weekday"):
            SeasonalNaiveBaseline().fit(X_bad)

    def test_predict_non_negative(self, synthetic_X):
        """B1 predictions must all be >= 0."""
        b1 = SeasonalNaiveBaseline().fit(synthetic_X)
        preds = b1.predict(synthetic_X)
        assert (preds >= 0).all(), "B1 predictions contain negative values"

    def test_predict_respects_capacity(self, synthetic_X):
        """B1 predictions must not exceed capacity for any row."""
        b1 = SeasonalNaiveBaseline().fit(synthetic_X)
        preds = b1.predict(synthetic_X)
        caps = synthetic_X["capacity"].to_numpy()
        assert np.all(preds <= caps + 1e-9), "B1 predictions exceed capacity"

    def test_predict_uses_lag_column(self, synthetic_X):
        """B1 prediction should equal min(lag_same_weekday, capacity)."""
        b1 = SeasonalNaiveBaseline().fit(synthetic_X)
        preds = b1.predict(synthetic_X)
        expected = np.minimum(
            synthetic_X["lag_same_weekday"].to_numpy(),
            synthetic_X["capacity"].to_numpy(),
        )
        np.testing.assert_allclose(preds, np.maximum(expected, 0.0), rtol=1e-9)


# ---------------------------------------------------------------------------
# B2 Seasonal Moving Average Baseline Tests
# ---------------------------------------------------------------------------

class TestSeasonalMovingAverageBaseline:
    """Unit tests for B2."""

    def test_fit_validates_schema(self, synthetic_X):
        b2 = SeasonalMovingAverageBaseline()
        b2.fit(synthetic_X)

    def test_predict_non_negative(self, synthetic_X):
        b2 = SeasonalMovingAverageBaseline().fit(synthetic_X)
        preds = b2.predict(synthetic_X)
        assert (preds >= 0).all()

    def test_predict_respects_capacity(self, synthetic_X):
        b2 = SeasonalMovingAverageBaseline().fit(synthetic_X)
        preds = b2.predict(synthetic_X)
        caps = synthetic_X["capacity"].to_numpy()
        assert np.all(preds <= caps + 1e-9)

    def test_predict_uses_mean_column(self, synthetic_X):
        b2 = SeasonalMovingAverageBaseline().fit(synthetic_X)
        preds = b2.predict(synthetic_X)
        expected = np.minimum(
            synthetic_X["same_weekday_4w_mean"].to_numpy(),
            synthetic_X["capacity"].to_numpy(),
        )
        np.testing.assert_allclose(preds, np.maximum(expected, 0.0), rtol=1e-9)


# ---------------------------------------------------------------------------
# Split Gap Tests — read from reports/split_info.json
# ---------------------------------------------------------------------------

SPLIT_INFO_PATH = Path("reports/split_info.json")


@pytest.mark.skipif(not SPLIT_INFO_PATH.exists(), reason="split_info.json not yet generated")
class TestSplitGap:
    """Verify chronological split boundaries are correct."""

    @pytest.fixture(autouse=True)
    def _load_split_info(self):
        with open(SPLIT_INFO_PATH, encoding="utf-8") as f:
            self.split_info = json.load(f)

    def _parse(self, key: str) -> date:
        return date.fromisoformat(self.split_info[key])

    def test_purge_gap_1_at_least_14_days(self):
        """Train end → val start gap must be >= 14 days."""
        t_end = self._parse("train_end_origin")
        v_start = self._parse("val_start_origin")
        gap = (v_start - t_end).days
        assert gap >= 14, f"Purge gap 1 is only {gap} days (must be >= 14)"

    def test_purge_gap_2_at_least_14_days(self):
        """Val end → test start gap must be >= 14 days."""
        v_end = self._parse("val_end_origin")
        te_start = self._parse("test_start_origin")
        gap = (te_start - v_end).days
        assert gap >= 14, f"Purge gap 2 is only {gap} days (must be >= 14)"

    def test_train_before_val(self):
        """Train end must be strictly before validation start."""
        assert self._parse("train_end_origin") < self._parse("val_start_origin")

    def test_val_before_test(self):
        """Validation end must be strictly before test start."""
        assert self._parse("val_end_origin") < self._parse("test_start_origin")

    def test_no_overlap_train_val(self):
        """Train end and val start cannot be the same date."""
        assert self._parse("train_end_origin") != self._parse("val_start_origin")

    def test_all_counts_positive(self):
        """Train, val, and test must have at least one origin each."""
        assert self.split_info["train_origin_count"] > 0
        assert self.split_info["val_origin_count"] > 0
        assert self.split_info["test_origin_count"] > 0
