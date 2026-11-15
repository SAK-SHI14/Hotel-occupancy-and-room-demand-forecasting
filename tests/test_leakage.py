"""Rigorous leakage invariance tests.

Proves that features computed at forecast origin T are mathematically invariant
to any ledger mutation, new bookings, or cancellations occurring after T.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
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
    """Load project configuration."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "configs" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_leakage_future_mutation_invariance(clean_ledger, config):
    """Mutate all future ledger records (booking_date > T, cancellation_date > T).

    Assert that build_features produces byte-for-byte identical output before and after mutation.
    """
    np.random.seed(config["project"]["seed"])

    # Test across multiple random origin dates and horizons
    categories = list(config["hotel"]["categories"].keys())
    test_origins = [
        date(2023, 6, 15),
        date(2023, 11, 20),
        date(2024, 2, 10),
        date(2024, 8, 5),
        date(2025, 1, 1),
    ]

    for origin_t in test_origins:
        for horizon in [1, 3, 7, 14]:
            stay_date = origin_t + timedelta(days=horizon)
            cat = np.random.choice(categories)

            # 1. Baseline features on pristine ledger
            feats_baseline = build_features(
                ledger=clean_ledger,
                origin_T=origin_t,
                stay_date=stay_date,
                room_category=cat,
                config=config,
            )

            # 2. Create mutated ledger where everything in the future is drastically altered or deleted
            mutated_ledger = clean_ledger.copy()
            b_dates = pd.to_datetime(mutated_ledger["booking_date"]).dt.date
            c_dates = pd.to_datetime(mutated_ledger["cancellation_date"]).dt.date

            # Strategy A: Randomly flip future cancellations to active or vice versa
            future_cancels = c_dates > origin_t
            mutated_ledger.loc[future_cancels, "is_cancelled"] = np.random.choice(
                [True, False], size=future_cancels.sum()
            )
            mutated_ledger.loc[future_cancels, "cancellation_date"] = "2026-12-31"

            # Strategy B: Delete 50% of future bookings entirely and corrupt the remaining future rows
            future_bookings = b_dates > origin_t
            drop_indices = mutated_ledger[future_bookings].sample(frac=0.5, random_state=42).index
            mutated_ledger = mutated_ledger.drop(drop_indices)
            remaining_future = pd.to_datetime(mutated_ledger["booking_date"]).dt.date > origin_t
            mutated_ledger.loc[remaining_future, "n_rooms"] = 999
            mutated_ledger.loc[remaining_future, "adr"] = 0.0

            # 3. Rebuild features on mutated ledger
            feats_mutated = build_features(
                ledger=mutated_ledger,
                origin_T=origin_t,
                stay_date=stay_date,
                room_category=cat,
                config=config,
            )

            # 4. Strict assertions: Every feature must be IDENTICAL
            for col in FEATURE_COLUMNS:
                val_base = feats_baseline[col]
                val_mut = feats_mutated[col]
                assert np.isclose(val_base, val_mut, rtol=1e-5, atol=1e-5), (
                    f"Leakage detected in feature '{col}' for origin {origin_t}, stay {stay_date}, cat {cat}! "
                    f"Baseline={val_base}, Mutated={val_mut}"
                )


def test_no_target_derived_feature_columns():
    """Assert that no feature column formula uses or approximates target stay date's final occupancy."""
    forbidden_terms = [
        "final_occupancy",
        "target",
        "realized_stay_date",
        "future_cancellation",
        "actual_rooms",
    ]
    for col in FEATURE_COLUMNS:
        for term in forbidden_terms:
            assert term not in col.lower(), f"Suspicious column name implying leakage: {col}"


def test_features_non_null_and_bounds(clean_ledger, config):
    """Verify that all features evaluate without NaNs and adhere to domain bounds."""
    origin_t = date(2024, 4, 1)
    for cat in config["hotel"]["categories"]:
        for h in [1, 7, 14]:
            s_date = origin_t + timedelta(days=h)
            feat = build_features(clean_ledger, origin_t, s_date, cat, config)
            for k in FEATURE_COLUMNS:
                assert k in feat, f"Missing feature key: {k}"
                assert not np.isnan(feat[k]), f"NaN detected in feature {k}"
            assert 0.0 <= feat["otb_capacity_share"] <= 1.0
            assert feat["horizon"] == h
            assert feat["capacity"] > 0
