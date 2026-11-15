"""Synthetic booking-level ledger generator for hotel demand forecasting.

Generates a realistic, multi-year booking ledger with weekly/annual seasonality,
holiday spikes, lead-time distributions, cancellation dynamics, and controlled anomalies.
"""

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hotel_forecast.features.calendar import CalendarEngine


def load_config(config_path: Path | str = "configs/config.yaml") -> dict[str, Any]:
    """Load system configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_booking_ledger(
    config: dict[str, Any] | None = None,
    output_path: Path | str = "data/raw/booking_ledger.csv",
) -> pd.DataFrame:
    """Generate synthetic booking ledger following realistic hospitality dynamics.

    Parameters
    ----------
    config : Dict[str, Any], optional
        System configuration dictionary. If None, loaded from configs/config.yaml.
    output_path : Path or str
        Destination path for raw CSV artifact.

    Returns
    -------
    pd.DataFrame
        Generated raw booking ledger DataFrame.
    """
    if config is None:
        config = load_config()

    seed = config["project"]["seed"]
    np.random.seed(seed)

    start_date = datetime.strptime(config["data"]["start_date"], "%Y-%m-%d").date()
    end_date = datetime.strptime(config["data"]["end_date"], "%Y-%m-%d").date()
    target_bookings = int(config["data"].get("target_bookings", 35000))
    categories = config["hotel"]["categories"]
    cat_names = list(categories.keys())
    noise_rate = float(config["data"].get("noise_rate", 0.015))

    calendar = CalendarEngine(country=config["data"].get("country_holidays", "IN"))

    total_days = (end_date - start_date).days
    avg_bookings_per_day = target_bookings / total_days

    bookings: list[dict[str, Any]] = []
    booking_counter = 100000

    # Base category distribution weights
    cat_base_weights = {
        "Standard": 0.45,
        "Deluxe": 0.30,
        "Family": 0.15,
        "Suite": 0.10,
    }

    current_date = start_date
    while current_date < end_date:
        # 1. Seasonality & Trend multipliers for arrival demand
        month = current_date.month
        is_weekend = calendar.is_weekend(current_date)
        is_holiday = calendar.is_holiday(current_date)
        is_event = calendar.is_event_period(current_date)

        # Yearly seasonality: High season Oct-Feb (winter tourism & weddings), Low June-Aug (monsoon)
        if month in (11, 12, 1, 2):
            season_mult = 1.30
        elif month in (10, 3):
            season_mult = 1.15
        elif month in (6, 7, 8):
            season_mult = 0.75
        else:
            season_mult = 0.95

        # Secular upward trend: ~4% per year
        days_from_start = (current_date - start_date).days
        trend_mult = 1.0 + 0.04 * (days_from_start / 365.25)

        # Holiday and Event lift
        holiday_mult = 1.25 if is_holiday else 1.0
        event_mult = 1.45 if is_event else 1.0

        # Daily arrival volume expectation
        expected_arrivals = avg_bookings_per_day * season_mult * trend_mult * holiday_mult * event_mult
        n_arrivals_today = np.random.poisson(lam=max(5.0, expected_arrivals))

        for _ in range(n_arrivals_today):
            booking_counter += 1
            booking_id = f"BK-{booking_counter}"

            # Category selection depends on day-of-week and event
            if is_weekend or is_holiday:
                weights = [0.30, 0.35, 0.22, 0.13]  # Leisure boosts Deluxe, Family, Suite
            elif is_event:
                weights = [0.50, 0.35, 0.05, 0.10]  # Corporate conference boosts Standard/Deluxe
            else:
                weights = [cat_base_weights[c] for c in cat_names]

            category = np.random.choice(cat_names, p=weights)

            # Distribution Channel selection
            if is_event:
                channel_p = [0.15, 0.25, 0.60]  # Direct, OTA, Corporate
            elif is_weekend or is_holiday:
                channel_p = [0.35, 0.60, 0.05]
            else:
                channel_p = [0.30, 0.45, 0.25]
            channel = np.random.choice(["Direct", "OTA", "Corporate"], p=channel_p)

            # Length of stay: geometric/discrete distribution
            if channel == "Corporate":
                stay_nights = int(np.random.choice([1, 2, 3, 4], p=[0.55, 0.30, 0.12, 0.03]))
            elif category == "Family":
                stay_nights = int(np.random.choice([2, 3, 4, 5, 7], p=[0.40, 0.35, 0.15, 0.07, 0.03]))
            else:
                stay_nights = int(np.random.choice([1, 2, 3, 4, 5], p=[0.48, 0.32, 0.12, 0.05, 0.03]))

            arrival_date = current_date
            departure_date = arrival_date + timedelta(days=stay_nights)

            # Lead time: Gamma distribution (right skewed)
            # Corporate: shorter lead time (mean ~7 days)
            # Leisure / Holiday: longer lead time (mean ~25-35 days)
            if channel == "Corporate":
                lead_time = int(np.random.gamma(shape=2.0, scale=4.0))
            elif is_holiday:
                lead_time = int(np.random.gamma(shape=3.5, scale=8.0))
            else:
                lead_time = int(np.random.gamma(shape=2.5, scale=9.0))
            lead_time = max(0, min(lead_time, 180))

            booking_date = arrival_date - timedelta(days=lead_time)

            # Rooms per booking: mostly 1, occasionally 2 for Family
            n_rooms = 2 if category == "Family" and np.random.rand() < 0.20 else 1

            # Base ADR calculation with dynamic pricing effects
            base_adr = float(categories[category]["base_adr"])
            # Demand premium: higher during high season, weekends, events
            adr_multiplier = season_mult * (1.15 if is_weekend else 1.0) * (1.20 if is_event else 1.0)
            # Add stochastic pricing variation (+/- 12%)
            adr_noise = np.random.normal(loc=1.0, scale=0.06)
            adr = round(base_adr * adr_multiplier * adr_noise, 2)

            # Cancellation dynamics:
            # Base rate: OTA has highest, corporate medium, direct lowest
            base_cancel_prob = {"OTA": 0.28, "Corporate": 0.14, "Direct": 0.10}[channel]
            # Cancellation risk scales with lead time
            lead_time_factor = min(1.6, 1.0 + 0.012 * lead_time)
            cancel_prob = min(0.65, base_cancel_prob * lead_time_factor)
            is_cancelled = bool(np.random.rand() < cancel_prob)

            cancellation_date = None
            if is_cancelled:
                if lead_time > 0:
                    # Cancelled between booking_date and arrival_date (Beta distribution skewed toward arrival)
                    fraction = np.random.beta(a=2.0, b=1.5)
                    days_after_booking = int(math.floor(fraction * lead_time))
                    cancellation_date = booking_date + timedelta(days=days_after_booking)
                else:
                    # Same day cancellation
                    cancellation_date = booking_date

            bookings.append({
                "booking_id": booking_id,
                "booking_date": booking_date.strftime("%Y-%m-%d"),
                "arrival_date": arrival_date.strftime("%Y-%m-%d"),
                "departure_date": departure_date.strftime("%Y-%m-%d"),
                "room_category": category,
                "n_rooms": n_rooms,
                "channel": channel,
                "is_cancelled": is_cancelled,
                "cancellation_date": cancellation_date.strftime("%Y-%m-%d") if cancellation_date else None,
                "adr": adr,
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(bookings)

    # 8. Inject controlled messiness / edge cases (noise_rate ~1.5%)
    n_noisy = int(len(df) * noise_rate)
    noisy_indices = np.random.choice(df.index, size=n_noisy, replace=False)

    for i, idx in enumerate(noisy_indices):
        issue_type = i % 5
        if issue_type == 0:
            # Missing channel
            df.at[idx, "channel"] = None
        elif issue_type == 1:
            # Duplicate booking ID with another row
            target_id = df.at[noisy_indices[(i + 1) % len(noisy_indices)], "booking_id"]
            df.at[idx, "booking_id"] = target_id
        elif issue_type == 2:
            # Impossible dates: arrival before booking
            b_dt = datetime.strptime(df.at[idx, "booking_date"], "%Y-%m-%d").date()
            df.at[idx, "arrival_date"] = (b_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        elif issue_type == 3:
            # Impossible dates: departure before arrival
            a_dt = datetime.strptime(df.at[idx, "arrival_date"], "%Y-%m-%d").date()
            df.at[idx, "departure_date"] = (a_dt - timedelta(days=2)).strftime("%Y-%m-%d")
        elif issue_type == 4:
            # Outlier / erroneous ADR (e.g. negative or 99999.0)
            df.at[idx, "adr"] = -500.0 if i % 2 == 0 else 99999.0

    # Ensure output directory exists
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Generated synthetic booking ledger with {len(df):,} records at {out_file}")
    return df


if __name__ == "__main__":
    generate_booking_ledger()
