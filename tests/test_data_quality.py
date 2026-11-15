"""Data quality and cleaning verification tests."""

from pathlib import Path

import pandas as pd
import yaml

from hotel_forecast.data.clean import clean_booking_ledger
from hotel_forecast.data.validate import validate_ledger


def test_data_quality_cleaning_pipeline(tmp_path):
    """Test that data validation identifies issues and clean_booking_ledger fixes them."""
    root = Path(__file__).resolve().parent.parent
    raw_path = root / "data" / "raw" / "booking_ledger.csv"
    assert raw_path.is_file(), "Raw booking ledger missing"

    raw_df = pd.read_csv(raw_path)
    assert len(raw_df) >= 30000, f"Expected at least 30k bookings, found {len(raw_df)}"

    # Audit raw dataframe
    raw_audit = validate_ledger(raw_df)
    # Ensure raw data actually had injected anomalies to prove cleaning isn't decorative
    assert raw_audit["duplicate_ids"] > 0, "Expected raw ledger to contain duplicate IDs"
    assert raw_audit["arrival_before_booking"] > 0, "Expected raw ledger to contain arrival < booking dates"
    assert raw_audit["departure_before_or_on_arrival"] > 0, "Expected departure <= arrival dates"
    assert raw_audit["negative_adr"] > 0, "Expected negative ADR anomalies"

    # Test cleaning function
    clean_test_path = tmp_path / "test_clean_ledger.csv"
    cleaned_df = clean_booking_ledger(raw_path=raw_path, output_path=clean_test_path)

    # Audit cleaned dataframe
    clean_audit = validate_ledger(cleaned_df)
    assert clean_audit["duplicate_ids"] == 0, "Cleaned ledger must have zero duplicate IDs"
    assert clean_audit["arrival_before_booking"] == 0, "Cleaned ledger must have zero arrival < booking dates"
    assert clean_audit["departure_before_or_on_arrival"] == 0, "Cleaned ledger must have zero departure <= arrival dates"
    assert clean_audit["negative_adr"] == 0, "Cleaned ledger must have zero negative ADRs"
    assert clean_audit["adr_outliers"] == 0, "Cleaned ledger must have zero extreme ADR outliers"
    assert clean_audit["missing_counts"]["channel"] == 0, "Cleaned ledger must have zero missing channels"
    assert clean_audit["invalid_categories"] == 0, "All room categories must match valid schema"
    assert clean_audit["invalid_rooms"] == 0, "All room counts must be >= 1"


def test_room_categories_and_capacities():
    """Verify that cleaned dataset contains all configured room categories."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "configs" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    clean_path = root / "data" / "processed" / "booking_ledger_clean.csv"
    assert clean_path.is_file(), "Clean booking ledger missing"

    df = pd.read_csv(clean_path)
    configured_cats = set(cfg["hotel"]["categories"].keys())
    present_cats = set(df["room_category"].unique())
    assert configured_cats == present_cats, f"Category mismatch: {configured_cats} vs {present_cats}"
