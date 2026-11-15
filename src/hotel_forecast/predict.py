"""Single shared prediction function used by both API and Streamlit dashboard.

Design principle: one function, two callers.
The API and the dashboard BOTH call predict() — there is no alternative path.
This guarantees that live API predictions and dashboard visualizations are
computed by the same code that uses the saved model artifact.

No hard-coded predictions anywhere in this file or any file that imports it.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hotel_forecast.features.build import FEATURE_COLUMNS, build_features
from hotel_forecast.models.conformal import SplitConformalPredictor

# Default artifact path (overridable for tests)
DEFAULT_MODEL_PATH = Path("models/model.joblib")

# Flag: more than 2x mean interval width → human review
WIDE_INTERVAL_MULTIPLIER = 2.0


def load_artifact(model_path: Path | str = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load and return the saved model artifact dict.

    Raises FileNotFoundError with a helpful message if the artifact is missing.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. "
            "Run 'python -m hotel_forecast.models.save_best' first to train and save the model."
        )
    return joblib.load(path)


def predict(
    origin_date: date | str,
    stay_date: date | str,
    room_category: str,
    ledger_df: pd.DataFrame,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Compute a single point-in-time forecast with a conformal interval.

    Parameters
    ----------
    origin_date : date or str (YYYY-MM-DD)
        Forecast origin T — information cutoff date.
    stay_date : date or str (YYYY-MM-DD)
        Target stay date to forecast.
    room_category : str
        One of 'Standard', 'Deluxe', 'Suite', 'Family'.
    ledger_df : pd.DataFrame
        Full cleaned booking ledger (passed in to avoid re-loading per call).
    artifact : dict
        Loaded model artifact (from load_artifact()).

    Returns
    -------
    dict with keys:
        prediction, lower, upper, occupancy_rate, capacity,
        horizon, interval_level, needs_human_review
    """
    if isinstance(origin_date, str):
        origin_date = date.fromisoformat(origin_date)
    if isinstance(stay_date, str):
        stay_date = date.fromisoformat(stay_date)

    horizon = (stay_date - origin_date).days
    if horizon < 1:
        raise ValueError(f"stay_date ({stay_date}) must be after origin_date ({origin_date}).")
    if horizon > 14:
        raise ValueError(f"Horizon {horizon} exceeds maximum supported horizon of 14 days.")

    # Build point-in-time features (the ONLY place features are created)
    features = build_features(ledger_df, origin_T=origin_date, stay_date=stay_date, room_category=room_category)

    model = artifact["model"]
    feature_cols = artifact.get("feature_columns", FEATURE_COLUMNS)
    capacity = float(features.get("capacity", 1.0))

    # Predict using saved model
    X_row = pd.DataFrame([{col: features.get(col, 0.0) for col in feature_cols}])
    raw_pred = float(model.predict(X_row)[0])
    point_pred = max(0.0, min(raw_pred, capacity))

    # Conformal interval
    conformal = SplitConformalPredictor.from_dict(artifact["conformal"])
    lower_arr, upper_arr = conformal.predict_intervals(X_row, np.array([point_pred]))
    lower = float(lower_arr[0])
    upper = float(upper_arr[0])
    interval_width = upper - lower

    # Wide interval → human review flag
    # Threshold: conformal quantile for this horizon bucket
    h_label = "h1-3" if horizon <= 3 else "h4-7" if horizon <= 7 else "h8-14"
    base_half_width = conformal.quantiles_.get(h_label, 1.0)
    needs_review = interval_width > WIDE_INTERVAL_MULTIPLIER * base_half_width * 2

    occupancy_rate = round(point_pred / capacity, 4) if capacity > 0 else 0.0

    return {
        "origin_date": str(origin_date),
        "stay_date": str(stay_date),
        "room_category": room_category,
        "horizon": horizon,
        "prediction": round(point_pred, 2),
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "interval_level": "80%",
        "occupancy_rate": occupancy_rate,
        "capacity": int(capacity),
        "needs_human_review": needs_review,
    }


def predict_horizon(
    origin_date: date | str,
    horizon_days: int,
    room_categories: list[str],
    ledger_df: pd.DataFrame,
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate forecasts for all (stay_date, room_category) pairs over a horizon.

    Parameters
    ----------
    origin_date : date or str
        Forecast origin T.
    horizon_days : int
        Number of days to forecast ahead (1..14).
    room_categories : list[str]
        Categories to forecast. Pass all four for a full run.
    ledger_df : pd.DataFrame
        Cleaned booking ledger.
    artifact : dict
        Loaded model artifact.

    Returns
    -------
    list[dict] — one entry per (stay_date, room_category) combination.
    """
    if isinstance(origin_date, str):
        origin_date = date.fromisoformat(origin_date)

    horizon_days = min(max(horizon_days, 1), 14)
    results = []
    for h in range(1, horizon_days + 1):
        stay = origin_date + timedelta(days=h)
        for cat in room_categories:
            try:
                row = predict(origin_date, stay, cat, ledger_df, artifact)
                results.append(row)
            except Exception as exc:
                # Log but continue: one bad row shouldn't kill the whole batch
                results.append({
                    "origin_date": str(origin_date),
                    "stay_date": str(stay),
                    "room_category": cat,
                    "horizon": h,
                    "error": str(exc),
                })
    return results
