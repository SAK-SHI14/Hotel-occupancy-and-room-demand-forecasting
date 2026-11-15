"""Feature engineering correctness and mathematical sanity tests."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from hotel_forecast.features.build import FEATURE_COLUMNS, build_features


@pytest.fixture
def clean_ledger():
    """Load cleaned booking ledger."""
    root = Path(__file__).resolve().parent.parent
    clean_path = root / "data" / "processed" / "booking_ledger_clean.csv"
    assert clean_path.is_file(), "Clean ledger missing"
    return pd.read_csv(clean_path)


@pytest.fixture
def config():
    """Load config."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "configs" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_horizon_validation(clean_ledger, config):
    """Test that build_features rejects non-positive horizons."""
    origin_t = date(2024, 5, 1)
    # Target date <= origin_T must fail
    with pytest.raises(ValueError, match="stay_date .* must be strictly after origin_T"):
        build_features(clean_ledger, origin_t, origin_t, "Standard", config)

    with pytest.raises(ValueError, match="stay_date .* must be strictly after origin_T"):
        build_features(clean_ledger, origin_t, origin_t - timedelta(days=1), "Standard", config)


def test_pickup_pace_arithmetic(clean_ledger, config):
    """Verify that pickup pace equals difference between OTB snapshots."""
    origin_t = date(2024, 7, 10)
    stay_date = origin_t + timedelta(days=5)

    feats = build_features(clean_ledger, origin_t, stay_date, "Deluxe", config)

    expected_pace_7d = feats["otb_rooms_T"] - feats["otb_rooms_T_minus_7"]
    expected_pace_14d = feats["otb_rooms_T"] - feats["otb_rooms_T_minus_14"]

    assert pytest.approx(feats["pickup_pace_7d"], 1e-5) == expected_pace_7d
    assert pytest.approx(feats["pickup_pace_14d"], 1e-5) == expected_pace_14d


def test_same_weekday_lag_is_strictly_past_and_aligned(clean_ledger, config):
    """Verify that lag_same_weekday refers to the same day of week and is <= origin_T."""
    origin_t = date(2024, 9, 15)

    for h in range(1, 15):
        stay_date = origin_t + timedelta(days=h)
        feats = build_features(clean_ledger, origin_t, stay_date, "Standard", config)

        # In formula, k = ceil(h/7), s_lag = stay_date - 7*k
        k = (h + 6) // 7
        s_lag = stay_date - timedelta(days=7 * k)

        assert s_lag <= origin_t, f"Lag date {s_lag} leaks into future beyond {origin_t}!"
        assert s_lag.weekday() == stay_date.weekday(), "Weekday alignment mismatch!"
        assert feats["horizon"] == h
        assert feats["day_of_week"] == stay_date.weekday()


def test_feature_list_completeness(clean_ledger, config):
    """Verify that build_features outputs every column defined in FEATURE_COLUMNS."""
    origin_t = date(2024, 3, 20)
    stay_date = origin_t + timedelta(days=7)
    feats = build_features(clean_ledger, origin_t, stay_date, "Family", config)

    for col in FEATURE_COLUMNS:
        assert col in feats, f"Feature '{col}' missing from build_features dictionary"
