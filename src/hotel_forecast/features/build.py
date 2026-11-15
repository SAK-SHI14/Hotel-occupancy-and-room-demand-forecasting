"""Leakage-safe feature engineering pipeline for hotel demand forecasting.

Constructs point-in-time features strictly as-of forecast origin T:
OTB pace, horizon-aware realized lags, calendar/holiday features, and booking dynamics.
"""

import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hotel_forecast.features.calendar import CalendarEngine

CATEGORY_CODES = {
    "Standard": 0,
    "Deluxe": 1,
    "Family": 2,
    "Suite": 3,
}

FEATURE_COLUMNS = [
    "otb_rooms_T",
    "otb_rooms_T_minus_7",
    "otb_rooms_T_minus_14",
    "pickup_pace_7d",
    "pickup_pace_14d",
    "otb_capacity_share",
    "lag_same_weekday",
    "rolling_mean_7d",
    "rolling_mean_28d",
    "same_weekday_4w_mean",
    "horizon",
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "days_to_next_holiday",
    "days_since_last_holiday",
    "is_event_period",
    "room_category_code",
    "capacity",
    "avg_lead_time_otb",
    "ota_share_otb",
]


def load_config(config_path: Path | str = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_date(d: date | datetime | str | pd.Timestamp) -> date:
    """Normalize various date types into datetime.date."""
    if isinstance(d, (pd.Timestamp, datetime)):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    return d


def compute_daily_realized_occupancy(ledger: pd.DataFrame) -> pd.DataFrame:
    """Compute realized daily occupied room nights per category for non-cancelled bookings.

    Parameters
    ----------
    ledger : pd.DataFrame
        Booking ledger.

    Returns
    -------
    pd.DataFrame
        Index: stay_date, Columns: category names. Values: occupied rooms.
    """
    df = ledger[~ledger["is_cancelled"]].copy()
    df["arr"] = pd.to_datetime(df["arrival_date"]).dt.date
    df["dep"] = pd.to_datetime(df["departure_date"]).dt.date

    records = []
    for _, row in df.iterrows():
        cur = row["arr"]
        dep = row["dep"]
        cat = row["room_category"]
        rooms = row["n_rooms"]
        while cur < dep:
            records.append({"stay_date": cur, "room_category": cat, "rooms": rooms})
            cur += timedelta(days=1)

    if not records:
        return pd.DataFrame()

    rec_df = pd.DataFrame(records)
    agg = rec_df.groupby(["stay_date", "room_category"])["rooms"].sum().unstack(fill_value=0)
    return agg


def build_features(
    ledger: pd.DataFrame,
    origin_T: date | str,
    stay_date: date | str,
    room_category: str,
    config: dict[str, Any] | None = None,
    calendar: CalendarEngine | None = None,
    realized_occ_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Extract point-in-time features strictly using information observable at origin T.

    Parameters
    ----------
    ledger : pd.DataFrame
        Booking ledger.
    origin_T : date or str
        Forecast origin date (information cutoff).
    stay_date : date or str
        Target stay date to forecast (T < stay_date <= T + 14).
    room_category : str
        Room category name ('Standard', 'Deluxe', 'Family', 'Suite').
    config : dict, optional
        Project configuration.
    calendar : CalendarEngine, optional
        Calendar engine instance.
    realized_occ_df : pd.DataFrame, optional
        Precomputed realized occupancy table for speed.

    Returns
    -------
    dict[str, Any]
        Feature dictionary containing all named features in FEATURE_COLUMNS.
    """
    if config is None:
        config = load_config()
    if calendar is None:
        calendar = CalendarEngine(country=config["data"].get("country_holidays", "IN"))

    # Ensure precomputed realized occupancy exists for instantaneous lag lookups
    if realized_occ_df is None:
        if hasattr(ledger, "attrs") and "realized_occ_df" in ledger.attrs:
            realized_occ_df = ledger.attrs["realized_occ_df"]
        else:
            realized_occ_df = compute_daily_realized_occupancy(ledger)
            if hasattr(ledger, "attrs"):
                ledger.attrs["realized_occ_df"] = realized_occ_df

    t_origin = _to_date(origin_T)
    s_date = _to_date(stay_date)
    horizon = (s_date - t_origin).days

    if horizon <= 0:
        raise ValueError(f"stay_date ({s_date}) must be strictly after origin_T ({t_origin}). Horizon={horizon}")

    capacities = {k: v["capacity"] for k, v in config["hotel"]["categories"].items()}
    capacity = float(capacities.get(room_category, 50))
    cat_code = CATEGORY_CODES.get(room_category, 0)

    # -------------------------------------------------------------
    # 1. On-The-Books (OTB) and Pickup Features (Strict Information Rule)
    # -------------------------------------------------------------
    # Ensure ledger date objects
    if "_b_date" not in ledger.columns:
        ledger = ledger.copy()
        ledger["_b_date"] = pd.to_datetime(ledger["booking_date"]).dt.date
        ledger["_a_date"] = pd.to_datetime(ledger["arrival_date"]).dt.date
        ledger["_d_date"] = pd.to_datetime(ledger["departure_date"]).dt.date
        ledger["_c_date"] = pd.to_datetime(ledger["cancellation_date"]).dt.date

    # Function to get OTB active bookings as of snapshot time
    def _get_otb_snapshot(as_of: date):
        # Known at as_of: booking_date <= as_of
        # Active at as_of: not cancelled OR cancellation_date > as_of
        # Occupies stay_date: arrival_date <= stay_date < departure_date
        # Category matches
        mask = (
            (ledger["room_category"] == room_category)
            & (ledger["_b_date"] <= as_of)
            & (
                (~ledger["is_cancelled"])
                | (ledger["_c_date"] > as_of)
            )
            & (ledger["_a_date"] <= s_date)
            & (s_date < ledger["_d_date"])
        )
        return ledger[mask]

    otb_t_df = _get_otb_snapshot(t_origin)
    otb_rooms_t = float(otb_t_df["n_rooms"].sum()) if not otb_t_df.empty else 0.0

    otb_t7_df = _get_otb_snapshot(t_origin - timedelta(days=7))
    otb_rooms_t7 = float(otb_t7_df["n_rooms"].sum()) if not otb_t7_df.empty else 0.0

    otb_t14_df = _get_otb_snapshot(t_origin - timedelta(days=14))
    otb_rooms_t14 = float(otb_t14_df["n_rooms"].sum()) if not otb_t14_df.empty else 0.0

    pickup_pace_7d = otb_rooms_t - otb_rooms_t7
    pickup_pace_14d = otb_rooms_t - otb_rooms_t14
    otb_capacity_share = min(1.0, otb_rooms_t / capacity)

    # Booking dynamics in current OTB
    if not otb_t_df.empty:
        lead_times = [(s_date - b).days for b in otb_t_df["_b_date"]]
        avg_lead_time_otb = float(np.mean(lead_times))
        ota_count = (otb_t_df["channel"] == "OTA").sum()
        ota_share_otb = float(ota_count / len(otb_t_df))
    else:
        avg_lead_time_otb = 0.0
        ota_share_otb = 0.0

    # -------------------------------------------------------------
    # 2. Lagged Realized Actuals (Horizon-Aware, Strictly <= T)
    # -------------------------------------------------------------
    # Occupancy for date t <= T
    def _get_past_occ(t: date) -> float:
        if realized_occ_df is not None:
            if t in realized_occ_df.index and room_category in realized_occ_df.columns:
                return float(realized_occ_df.loc[t, room_category])
            return 0.0
        # Fallback calculation directly on ledger
        mask = (
            (ledger["room_category"] == room_category)
            & (~ledger["is_cancelled"])
            & (ledger["_a_date"] <= t)
            & (t < ledger["_d_date"])
        )
        return float(ledger[mask]["n_rooms"].sum())

    # Same-weekday lag: s - 7 * ceil(h/7) <= T
    k = int(math.ceil(horizon / 7.0))
    s_lag = s_date - timedelta(days=7 * k)
    lag_same_weekday = _get_past_occ(s_lag)

    # Rolling 7-day and 28-day mean ending at T
    past_7d = [_get_past_occ(t_origin - timedelta(days=d)) for d in range(7)]
    rolling_mean_7d = float(np.mean(past_7d))

    past_28d = [_get_past_occ(t_origin - timedelta(days=d)) for d in range(28)]
    rolling_mean_28d = float(np.mean(past_28d))

    # Same-weekday 4-week average ending at T
    same_weekday_4w = [_get_past_occ(s_lag - timedelta(days=7 * j)) for j in range(4)]
    same_weekday_4w_mean = float(np.mean(same_weekday_4w))

    # -------------------------------------------------------------
    # 3. Calendar & Holiday Features for Target Stay Date
    # -------------------------------------------------------------
    dow = s_date.weekday()
    month = s_date.month
    woy = s_date.isocalendar()[1]
    is_weekend = 1 if calendar.is_weekend(s_date) else 0
    is_holiday = 1 if calendar.is_holiday(s_date) else 0
    days_to_next_holiday = calendar.days_to_next_holiday(s_date)
    days_since_last_holiday = calendar.days_since_last_holiday(s_date)
    is_event_period = 1 if calendar.is_event_period(s_date) else 0

    features = {
        "otb_rooms_T": otb_rooms_t,
        "otb_rooms_T_minus_7": otb_rooms_t7,
        "otb_rooms_T_minus_14": otb_rooms_t14,
        "pickup_pace_7d": pickup_pace_7d,
        "pickup_pace_14d": pickup_pace_14d,
        "otb_capacity_share": otb_capacity_share,
        "lag_same_weekday": lag_same_weekday,
        "rolling_mean_7d": rolling_mean_7d,
        "rolling_mean_28d": rolling_mean_28d,
        "same_weekday_4w_mean": same_weekday_4w_mean,
        "horizon": horizon,
        "day_of_week": dow,
        "month": month,
        "week_of_year": woy,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "days_to_next_holiday": days_to_next_holiday,
        "days_since_last_holiday": days_since_last_holiday,
        "is_event_period": is_event_period,
        "room_category_code": cat_code,
        "capacity": capacity,
        "avg_lead_time_otb": avg_lead_time_otb,
        "ota_share_otb": ota_share_otb,
    }
    return features


def build_feature_dataset(
    ledger: pd.DataFrame,
    config: dict[str, Any] | None = None,
    start_origin: date | str | None = None,
    end_origin: date | str | None = None,
    origin_step_days: int = 1,
    max_horizon: int = 14,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build full point-in-time feature matrix X and target vector y across origins and horizons.

    Parameters
    ----------
    ledger : pd.DataFrame
        Cleaned booking ledger.
    config : dict, optional
        Project configuration.
    start_origin : date or str, optional
        Earliest origin T date. Defaults to 45 days after ledger start (to ensure 28d lag burn-in).
    end_origin : date or str, optional
        Latest origin T date. Defaults to 15 days before ledger end (to observe 14d horizon).
    origin_step_days : int
        Step between consecutive forecast origins (default=1 for daily forecasts).
    max_horizon : int
        Maximum forward horizon in days (default=14).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature DataFrame X and target Series y (occupied_room_nights).
    """
    if config is None:
        config = load_config()

    calendar = CalendarEngine(country=config["data"].get("country_holidays", "IN"))
    categories = list(config["hotel"]["categories"].keys())

    # Pre-index ledger dates for performance
    df = ledger.copy()
    df["_b_date"] = pd.to_datetime(df["booking_date"]).dt.date
    df["_a_date"] = pd.to_datetime(df["arrival_date"]).dt.date
    df["_d_date"] = pd.to_datetime(df["departure_date"]).dt.date
    df["_c_date"] = pd.to_datetime(df["cancellation_date"]).dt.date

    # Precompute realized occupancy matrix for instant lag lookups
    realized_occ_df = compute_daily_realized_occupancy(df)

    min_date = df["_a_date"].min()
    max_date = df["_d_date"].max()

    first_origin = _to_date(start_origin) if start_origin else (min_date + timedelta(days=45))
    last_origin = _to_date(end_origin) if end_origin else (max_date - timedelta(days=max_horizon + 1))

    current_origin = first_origin
    rows = []
    targets = []
    metadata = []

    print(f"Building feature dataset from origin {first_origin} to {last_origin} (step={origin_step_days}d)...")

    while current_origin <= last_origin:
        # Optimization: slice relevant active window for origin T to accelerate OTB scans
        window_end = current_origin + timedelta(days=max_horizon + 1)
        sub_ledger = df[
            (df["_b_date"] <= current_origin)
            & (df["_a_date"] <= window_end)
            & (df["_d_date"] > current_origin)
        ]

        for cat in categories:
            cat_ledger = sub_ledger[sub_ledger["room_category"] == cat]
            for h in range(1, max_horizon + 1):
                s_date = current_origin + timedelta(days=h)
                feat = build_features(
                    ledger=cat_ledger,
                    origin_T=current_origin,
                    stay_date=s_date,
                    room_category=cat,
                    config=config,
                    calendar=calendar,
                    realized_occ_df=realized_occ_df,
                )
                target_val = (
                    float(realized_occ_df.loc[s_date, cat])
                    if (s_date in realized_occ_df.index and cat in realized_occ_df.columns)
                    else 0.0
                )

                rows.append(feat)
                targets.append(target_val)
                metadata.append({
                    "origin_date": current_origin,
                    "stay_date": s_date,
                    "room_category": cat,
                    "horizon": h,
                })

        current_origin += timedelta(days=origin_step_days)

    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    y = pd.Series(targets, name="occupied_room_nights")
    meta_df = pd.DataFrame(metadata)

    # Attach metadata columns to X index / attributes
    for col in meta_df.columns:
        X[f"meta_{col}"] = meta_df[col]

    print(f"Feature dataset constructed: {len(X):,} samples with {len(FEATURE_COLUMNS)} features.")
    return X, y


if __name__ == "__main__":
    ledger_path = Path("data/processed/booking_ledger_clean.csv")
    if not ledger_path.exists():
        raise FileNotFoundError(f"Clean ledger not found at {ledger_path}")
    ledger_df = pd.read_csv(ledger_path)
    X, y = build_feature_dataset(ledger_df, origin_step_days=2)  # fast sample
    print(X.head(2))
