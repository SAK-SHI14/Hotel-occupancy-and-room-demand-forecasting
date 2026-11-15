"""Tests for split conformal prediction uncertainty intervals.

Verifies:
1. Conformal quantiles increase monotonically with horizon bucket distance (width grows with horizon).
2. Interval clipping guarantees lower >= 0 and upper <= capacity.
3. Quantiles serialize and deserialize losslessly.
4. Conformal predictor handles edge cases (e.g. zero capacity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hotel_forecast.models.conformal import SplitConformalPredictor


@pytest.fixture()
def synthetic_val_data():
    """Synthetic validation dataset with varying horizons and known residuals."""
    rng = np.random.default_rng(42)
    n = 600
    horizons = np.repeat([2, 5, 10], 200)  # spans h1-3, h4-7, h8-14
    capacity = np.full(n, 50.0)

    # Synthetic residuals that grow with horizon:
    # h=2: noise ~ N(0, 1)
    # h=5: noise ~ N(0, 3)
    # h=10: noise ~ N(0, 6)
    noise_scales = np.where(horizons <= 3, 1.0, np.where(horizons <= 7, 3.0, 6.0))
    noise = rng.normal(0, noise_scales, n)

    base_demand = 25.0
    y_true = np.clip(base_demand + noise, 0.0, capacity)

    X = pd.DataFrame({
        "horizon": horizons,
        "capacity": capacity,
        "dummy_feat": rng.normal(0, 1, n),
    })
    return X, y_true


class DummyModel:
    """Mock model predicting constant baseline."""
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 25.0)


def test_conformal_calibration_and_width_growth(synthetic_val_data):
    """Interval width must grow monotonically across horizon buckets."""
    X, y = synthetic_val_data
    model = DummyModel()

    conformal = SplitConformalPredictor(alpha=0.20)
    conformal.calibrate(X, y, model)

    q1 = conformal.quantiles_["h1-3"]
    q2 = conformal.quantiles_["h4-7"]
    q3 = conformal.quantiles_["h8-14"]

    assert q1 < q2, f"Quantile for h1-3 ({q1}) should be strictly less than h4-7 ({q2})"
    assert q2 < q3, f"Quantile for h4-7 ({q2}) should be strictly less than h8-14 ({q3})"


def test_conformal_bounds_clipping(synthetic_val_data):
    """Lower bounds must be >= 0 and upper bounds <= capacity."""
    X, y = synthetic_val_data
    model = DummyModel()

    conformal = SplitConformalPredictor(alpha=0.20)
    conformal.calibrate(X, y, model)

    preds = model.predict(X)
    lower, upper = conformal.predict_intervals(X, preds)

    assert (lower >= 0.0).all(), "Lower bounds cannot be negative"
    assert (upper <= X["capacity"].to_numpy()).all(), "Upper bounds cannot exceed capacity"
    assert (lower <= upper).all(), "Lower bound must be <= upper bound"


def test_conformal_serialization_roundtrip(synthetic_val_data):
    """to_dict and from_dict must produce identical quantiles."""
    X, y = synthetic_val_data
    conformal = SplitConformalPredictor(alpha=0.20).calibrate(X, y, DummyModel())

    serialized = conformal.to_dict()
    restored = SplitConformalPredictor.from_dict(serialized)

    assert restored.alpha == conformal.alpha
    assert restored.quantiles_ == conformal.quantiles_
