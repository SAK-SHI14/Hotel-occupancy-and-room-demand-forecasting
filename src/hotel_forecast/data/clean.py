"""Data cleaning module for hotel booking ledger.

Transforms raw ledger into a validated, cleaned dataset free of impossible dates,
duplicates, missing channels, and pricing outliers.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hotel_forecast.data.validate import generate_data_quality_report, validate_ledger


def load_config(config_path: Path | str = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration dictionary."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_booking_ledger(
    raw_path: Path | str = "data/raw/booking_ledger.csv",
    output_path: Path | str = "data/processed/booking_ledger_clean.csv",
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Clean the raw booking ledger according to documented remediation decisions.

    Parameters
    ----------
    raw_path : Path or str
        Path to raw booking ledger CSV.
    output_path : Path or str
        Destination path for cleaned CSV.
    config : Dict[str, Any], optional
        System configuration.

    Returns
    -------
    pd.DataFrame
        Cleaned booking ledger DataFrame.
    """
    if config is None:
        config = load_config()

    raw_df = pd.read_csv(raw_path)
    raw_audit = validate_ledger(raw_df, config=config)
    initial_count = len(raw_df)

    df = raw_df.copy()

    # 1. Deduplicate by booking_id (retain first instance)
    df = df.drop_duplicates(subset=["booking_id"], keep="first").copy()

    # 2. Impute missing channel with mode
    mode_channel = df["channel"].mode()[0] if not df["channel"].dropna().empty else "OTA"
    df["channel"] = df["channel"].fillna(mode_channel)

    # 3. Parse and filter impossible dates
    b_date = pd.to_datetime(df["booking_date"], errors="coerce")
    a_date = pd.to_datetime(df["arrival_date"], errors="coerce")
    d_date = pd.to_datetime(df["departure_date"], errors="coerce")

    valid_dates_mask = (a_date >= b_date) & (d_date > a_date)
    df = df[valid_dates_mask].copy()

    # If cancelled, check cancellation_date logic
    df.loc[~df["is_cancelled"], "cancellation_date"] = None
    b_valid = pd.to_datetime(df["booking_date"], errors="coerce")
    d_valid = pd.to_datetime(df["departure_date"], errors="coerce")
    c_valid = pd.to_datetime(df["cancellation_date"], errors="coerce")

    invalid_c_mask = df["is_cancelled"] & ((c_valid < b_valid) | (c_valid > d_valid))
    if invalid_c_mask.any():
        df.loc[invalid_c_mask, "cancellation_date"] = df.loc[invalid_c_mask, "booking_date"]

    # 4. Clean ADR pricing outliers
    categories = config["hotel"]["categories"]
    for cat, props in categories.items():
        base_adr = float(props["base_adr"])
        cat_mask = df["room_category"] == cat
        cat_adrs = df.loc[cat_mask, "adr"]

        q25 = cat_adrs.quantile(0.25)
        q75 = cat_adrs.quantile(0.75)
        iqr = q75 - q25
        upper_fence = q75 + 3.0 * iqr
        lower_fence = max(100.0, q25 - 3.0 * iqr)

        median_adr = cat_adrs[(cat_adrs >= lower_fence) & (cat_adrs <= upper_fence)].median()
        if np.isnan(median_adr):
            median_adr = base_adr

        outlier_mask = cat_mask & ((df["adr"] < lower_fence) | (df["adr"] > upper_fence))
        df.loc[outlier_mask, "adr"] = round(median_adr, 2)

    # 5. Standardize column data types
    df["booking_id"] = df["booking_id"].astype(str)
    df["booking_date"] = df["booking_date"].astype(str)
    df["arrival_date"] = df["arrival_date"].astype(str)
    df["departure_date"] = df["departure_date"].astype(str)
    df["room_category"] = df["room_category"].astype(str)
    df["n_rooms"] = df["n_rooms"].astype(int)
    df["channel"] = df["channel"].astype(str)
    df["is_cancelled"] = df["is_cancelled"].astype(bool)
    df["adr"] = df["adr"].astype(float)

    # 6. Save cleaned dataset
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    final_count = len(df)
    dropped = initial_count - final_count
    print(f"Cleaned ledger: {final_count:,} valid records retained ({dropped} corrupted records dropped). Saved to {out_file}")

    # Generate data quality report comparing raw vs cleaned
    clean_audit = validate_ledger(df, config=config)
    generate_data_quality_report(raw_audit, clean_audit, output_path="reports/data_quality_report.md")

    return df


if __name__ == "__main__":
    raw_file = Path("data/raw/booking_ledger.csv")
    clean_booking_ledger(raw_path=raw_file)
