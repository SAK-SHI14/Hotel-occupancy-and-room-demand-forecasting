"""Test suite for FastAPI endpoints and prediction interfaces.

Tests endpoints directly to ensure deterministic, sandbox-safe execution:
1. GET /health returns status 'ok'.
2. GET /model-info returns valid metadata and model details.
3. POST /forecast generates valid, bounded predictions from live model.
4. Changing origin_date changes forecast values (proves live computation, not hardcoded).
5. Forecast predictions strictly respect room category capacities.
6. Pydantic schema validation rejects invalid origins, out-of-range horizons, and unknown categories.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hotel_forecast.api.main import (
    ForecastRequest,
    ForecastResponse,
    forecast,
    health,
    model_info,
)

MODEL_PATH = Path("models/model.joblib")


def test_health_endpoint():
    """Verify /health returns 'ok' status and timestamp."""
    data = health()
    assert data["status"] == "ok"
    assert "timestamp" in data


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model artifact not found")
def test_model_info_endpoint():
    """Verify /model-info provides metadata matching saved artifact."""
    data = model_info()
    assert "model_name" in data
    assert "conformal_quantiles" in data
    assert "val_metrics" in data
    assert len(data["conformal_quantiles"]) == 3
    assert data["conformal_coverage_target"] == "80%"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model artifact not found")
def test_forecast_live_computation():
    """Verify POST /forecast produces valid prediction rows bounded by capacity."""
    req = ForecastRequest(
        origin_date="2025-09-01",
        horizon_days=3,
        room_categories=["Standard", "Deluxe"],
    )
    res = forecast(req)
    assert isinstance(res, ForecastResponse)
    assert res.origin_date == "2025-09-01"
    assert len(res.forecasts) == 6  # 3 days * 2 categories

    for fc in res.forecasts:
        assert fc.prediction >= 0.0
        assert fc.prediction <= fc.capacity
        assert fc.lower <= fc.upper
        assert fc.occupancy_rate >= 0.0
        assert fc.occupancy_rate <= 1.0
        assert fc.interval_level == "80%"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model artifact not found")
def test_forecast_changes_with_origin_not_hardcoded():
    """Changing origin date must change the resulting forecast (demonstrates live inference)."""
    req1 = ForecastRequest(origin_date="2025-03-01", horizon_days=1, room_categories=["Standard"])
    req2 = ForecastRequest(origin_date="2025-08-01", horizon_days=1, room_categories=["Standard"])

    r1 = forecast(req1)
    r2 = forecast(req2)

    pred1 = r1.forecasts[0].prediction
    pred2 = r2.forecasts[0].prediction
    assert pred1 != pred2, f"Predictions must vary with origin date: {pred1} == {pred2}"


def test_forecast_validation_rejections():
    """Pydantic schema must reject malformed inputs."""
    # Invalid date string
    with pytest.raises(ValidationError):
        ForecastRequest(origin_date="not-a-date", horizon_days=7)

    # Horizon out of bounds (>14)
    with pytest.raises(ValidationError):
        ForecastRequest(origin_date="2025-01-01", horizon_days=20)

    # Horizon out of bounds (<1)
    with pytest.raises(ValidationError):
        ForecastRequest(origin_date="2025-01-01", horizon_days=0)

    # Nonexistent room category
    with pytest.raises(ValidationError):
        ForecastRequest(origin_date="2025-01-01", room_categories=["Penthouse"])
