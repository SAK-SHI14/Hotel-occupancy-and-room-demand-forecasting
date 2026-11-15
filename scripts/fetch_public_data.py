"""Public Dataset Adapter & Schema Mapper (Antonio et al., 2019).

Provides instructions and schema mapping for the open-access Hotel Booking Demand dataset.
Downloads NOTHING automatically to comply with assessment guidelines.
"""

import sys
from pathlib import Path

import pandas as pd

DATASET_CITATION = """
Antonio, N., de Almeida, A., & Nunes, L. (2019).
Hotel booking demand datasets. Data in Brief, 22, 41-49.
License: Creative Commons Attribution 4.0 International (CC BY 4.0).
Source: https://doi.org/10.1016/j.dib.2018.11.126
"""


def print_instructions():
    """Print instructions for manually downloading and citing the public dataset."""
    print("=" * 80)
    print("HOTEL BOOKING DEMAND PUBLIC DATASET ADAPTER (Antonio et al., 2019)")
    print("=" * 80)
    print(DATASET_CITATION)
    print("\nInstructions:")
    print("1. Download 'hotel_bookings.csv' from Kaggle or the ScienceDirect Data in Brief repository.")
    print("2. Place the file at: data/raw/public_hotel_bookings.csv")
    print("3. Run: python scripts/fetch_public_data.py --map")
    print("=" * 80)


def map_public_dataset(
    input_path: Path | str = "data/raw/public_hotel_bookings.csv",
    output_path: Path | str = "data/processed/public_booking_ledger_mapped.csv",
) -> pd.DataFrame | None:
    """Map public dataset columns into our canonical booking ledger schema."""
    in_file = Path(input_path)
    if not in_file.exists():
        print(f"File not found: {in_file}")
        print_instructions()
        return None

    df = pd.read_csv(in_file)
    print(f"Loaded {len(df):,} records from {in_file}")

    # Map month string to number
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }

    mapped_rows = []
    for idx, row in df.iterrows():
        yr = int(row["arrival_date_year"])
        mo = month_map.get(row["arrival_date_month"], 1)
        day = int(row["arrival_date_day_of_month"])
        arrival = pd.Timestamp(year=yr, month=mo, day=day)

        lead_time = int(row["lead_time"])
        booking = arrival - pd.Timedelta(days=lead_time)

        stay_nights = int(row.get("stays_in_weekend_nights", 0)) + int(row.get("stays_in_week_nights", 1))
        departure = arrival + pd.Timedelta(days=max(1, stay_nights))

        # Map distribution channel
        market_seg = str(row.get("market_segment", "Direct")).lower()
        if "online" in market_seg or "ota" in market_seg:
            channel = "OTA"
        elif "corporate" in market_seg:
            channel = "Corporate"
        else:
            channel = "Direct"

        # Map reserved room type to our 4 categories
        room_type = str(row.get("reserved_room_type", "A"))
        if room_type in ["A", "B"]:
            category = "Standard"
        elif room_type in ["D", "E"]:
            category = "Deluxe"
        elif room_type in ["F", "G"]:
            category = "Family"
        else:
            category = "Suite"

        is_cancelled = bool(row["is_canceled"] == 1)
        cancellation_date = None
        if is_cancelled:
            reservation_status_date = str(row.get("reservation_status_date", ""))
            cancellation_date = reservation_status_date if reservation_status_date else str(booking.date())

        mapped_rows.append({
            "booking_id": f"PUB-{idx + 1:06d}",
            "booking_date": str(booking.date()),
            "arrival_date": str(arrival.date()),
            "departure_date": str(departure.date()),
            "room_category": category,
            "n_rooms": 1,
            "channel": channel,
            "is_cancelled": is_cancelled,
            "cancellation_date": cancellation_date,
            "adr": float(row.get("adr", 100.0)),
        })

    mapped_df = pd.DataFrame(mapped_rows)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mapped_df.to_csv(out_file, index=False)
    print(f"Mapped public dataset saved to {out_file} ({len(mapped_df):,} records)")
    return mapped_df


if __name__ == "__main__":
    if "--map" in sys.argv:
        map_public_dataset()
    else:
        print_instructions()
