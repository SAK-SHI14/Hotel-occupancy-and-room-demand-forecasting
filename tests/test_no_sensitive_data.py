"""Privacy-by-design assertions: verify zero guest PII exists in datasets."""

from pathlib import Path

import pandas as pd

FORBIDDEN_PII_PATTERNS = [
    "name",
    "email",
    "phone",
    "mobile",
    "ssn",
    "passport",
    "national_id",
    "address",
    "credit_card",
    "card_number",
    "cvv",
    "guest_id",
    "guest_name",
    "ip_address",
]


def test_no_sensitive_columns_in_raw_and_clean():
    """Verify that neither raw nor cleaned datasets contain any PII columns."""
    root = Path(__file__).resolve().parent.parent
    raw_path = root / "data" / "raw" / "booking_ledger.csv"
    clean_path = root / "data" / "processed" / "booking_ledger_clean.csv"

    for path in (raw_path, clean_path):
        if not path.is_file():
            continue
        df = pd.read_csv(path, nrows=100)
        for col in df.columns:
            col_lower = col.lower()
            for pattern in FORBIDDEN_PII_PATTERNS:
                assert pattern not in col_lower, (
                    f"Forbidden PII term '{pattern}' detected in column '{col}' of {path.name}"
                )


def test_opaque_booking_id_format():
    """Verify that booking_id is strictly an opaque surrogate key."""
    root = Path(__file__).resolve().parent.parent
    clean_path = root / "data" / "processed" / "booking_ledger_clean.csv"
    assert clean_path.is_file(), "Clean ledger missing"

    df = pd.read_csv(clean_path, nrows=500)
    for bid in df["booking_id"]:
        assert str(bid).startswith("BK-"), f"Expected opaque prefix 'BK-', got {bid}"
        assert str(bid)[3:].isdigit(), f"Expected numeric suffix after 'BK-', got {bid}"


def test_no_sensitive_columns_in_feature_catalog():
    """Verify that feature catalog columns contain no sensitive identifiers."""
    from hotel_forecast.features.build import FEATURE_COLUMNS

    for col in FEATURE_COLUMNS:
        col_lower = col.lower()
        for pattern in FORBIDDEN_PII_PATTERNS:
            assert pattern not in col_lower, f"Forbidden PII term '{pattern}' detected in feature '{col}'"

