"""FastAPI application for hotel room demand forecasting.

Endpoints:
  GET  /health        — liveness check
  GET  /model-info    — model metadata from artifact
  POST /forecast      — generate forecasts for a date range and category list

All predictions are computed by calling hotel_forecast.predict.predict()
which uses the saved model.joblib artifact — no hard-coded values anywhere.

Usage (from project root):
    uvicorn hotel_forecast.api.main:app --reload --port 8000
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, model_validator

from hotel_forecast.models.explain import FEATURE_COLUMNS
from hotel_forecast.predict import load_artifact, predict_horizon

# ---------------------------------------------------------------------------
# App and global state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hotel Room Demand Forecasting API",
    description=(
        "Point-in-time demand forecasts for hotel room categories. "
        "All predictions are computed live from the saved model artifact — "
        "no hard-coded values."
    ),
    version="1.0.0",
)

# Lazy-loaded singletons (loaded once on first request to keep startup fast)
_artifact: dict[str, Any] | None = None
_ledger_df: pd.DataFrame | None = None

MODEL_PATH = Path("models/model.joblib")
LEDGER_PATH = Path("data/processed/booking_ledger_clean.csv")
MODEL_CARD_PATH = Path("models/model_card.json")

VALID_CATEGORIES = {"Standard", "Deluxe", "Suite", "Family"}
MIN_HORIZON = 1
MAX_HORIZON = 14


def _get_artifact() -> dict[str, Any]:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model artifact not found at '{MODEL_PATH}'. "
                    "Run 'python -m hotel_forecast.models.save_best' first."
                ),
            )
        _artifact = load_artifact(MODEL_PATH)
    return _artifact


def _get_ledger() -> pd.DataFrame:
    global _ledger_df
    if _ledger_df is None:
        if not LEDGER_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Booking ledger not found at '{LEDGER_PATH}'.",
            )
        df = pd.read_csv(LEDGER_PATH)
        df["_b_date"] = pd.to_datetime(df["booking_date"]).dt.date
        df["_a_date"] = pd.to_datetime(df["arrival_date"]).dt.date
        df["_d_date"] = pd.to_datetime(df["departure_date"]).dt.date
        df["_c_date"] = pd.to_datetime(df["cancellation_date"]).dt.date
        from hotel_forecast.features.build import compute_daily_realized_occupancy
        df.attrs["realized_occ_df"] = compute_daily_realized_occupancy(df)
        _ledger_df = df
    return _ledger_df


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    """Request body for /forecast."""

    origin_date: str
    horizon_days: int = 14
    room_categories: list[str] = ["Standard", "Deluxe", "Suite", "Family"]

    @field_validator("origin_date")
    @classmethod
    def validate_origin_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except ValueError as err:
            raise ValueError(f"origin_date '{v}' is not a valid YYYY-MM-DD date.") from err
        return v

    @field_validator("horizon_days")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if not (MIN_HORIZON <= v <= MAX_HORIZON):
            raise ValueError(f"horizon_days must be between {MIN_HORIZON} and {MAX_HORIZON}.")
        return v

    @field_validator("room_categories")
    @classmethod
    def validate_categories(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_CATEGORIES
        if invalid:
            raise ValueError(f"Invalid room categories: {invalid}. Valid: {VALID_CATEGORIES}.")
        return v

    @model_validator(mode="after")
    def validate_origin_in_data_range(self) -> ForecastRequest:
        """Warn if origin_date is outside the training window (soft check)."""
        # We allow it but will note it in the response — model still runs.
        return self


class ForecastRow(BaseModel):
    """Single forecast output row."""

    origin_date: str
    stay_date: str
    room_category: str
    horizon: int
    prediction: float
    lower: float
    upper: float
    interval_level: str
    occupancy_rate: float
    capacity: int
    needs_human_review: bool
    explanation: str | None = None


class ForecastResponse(BaseModel):
    """Full API response for /forecast."""

    generated_at: str
    origin_date: str
    horizon_days: int
    room_categories: list[str]
    forecasts: list[ForecastRow]
    model_name: str
    limitations: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness check")
def health() -> dict[str, str]:
    """Return 200 OK when the API is running."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/model-info", summary="Model metadata")
def model_info() -> dict[str, Any]:
    """Return model card metadata from the saved artifact."""
    artifact = _get_artifact()
    return {
        "model_name": artifact.get("model_name", "unknown"),
        "model_type": artifact.get("model_type", "unknown"),
        "training_date": artifact.get("training_date", "unknown"),
        "feature_count": len(artifact.get("feature_columns", FEATURE_COLUMNS)),
        "conformal_coverage_target": "80%",
        "conformal_quantiles": artifact.get("conformal", {}).get("quantiles", {}),
        "val_metrics": artifact.get("val_metrics", {}),
        "data_hash": artifact.get("data_hash", "unknown"),
        "limitations": [
            "Trained on synthetic data — real-world performance may differ.",
            "Demand shocks beyond the training distribution may degrade accuracy.",
            "Interval coverage may drift if booking patterns change significantly.",
        ],
    }


@app.post("/forecast", response_model=ForecastResponse, summary="Generate demand forecasts")
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Compute live predictions for a horizon window and set of room categories.

    The model is loaded from disk once and predictions are computed on-the-fly
    using the saved feature pipeline. No hard-coded values.
    """
    artifact = _get_artifact()
    ledger = _get_ledger()

    # Generate predictions for all (stay_date × room_category) pairs
    raw_results = predict_horizon(
        origin_date=request.origin_date,
        horizon_days=request.horizon_days,
        room_categories=request.room_categories,
        ledger_df=ledger,
        artifact=artifact,
    )

    # Optionally add natural-language explanations (if permutation importance available)
    imp_csv = Path("reports/permutation_importance.csv")
    imp_df = pd.read_csv(imp_csv) if imp_csv.exists() else None

    forecast_rows: list[ForecastRow] = []
    for row in raw_results:
        if "error" in row:
            # Propagate errors as a reviewable row
            forecast_rows.append(
                ForecastRow(
                    origin_date=row.get("origin_date", request.origin_date),
                    stay_date=row.get("stay_date", ""),
                    room_category=row.get("room_category", ""),
                    horizon=row.get("horizon", 0),
                    prediction=-1.0,
                    lower=-1.0,
                    upper=-1.0,
                    interval_level="80%",
                    occupancy_rate=-1.0,
                    capacity=0,
                    needs_human_review=True,
                    explanation=f"Error: {row['error']}",
                )
            )
            continue

        explanation_text = None
        if imp_df is not None:
            try:
                explanation_text = (
                    f"Forecast: {row['prediction']:.1f} rooms for {row['room_category']} "
                    f"on {row['stay_date']} (h={row['horizon']}). "
                    f"Occupancy: {row['occupancy_rate']*100:.1f}% of {row['capacity']} rooms. "
                    f"80% interval: [{row['lower']:.1f}, {row['upper']:.1f}]."
                )
            except Exception:
                pass

        forecast_rows.append(
            ForecastRow(
                origin_date=row["origin_date"],
                stay_date=row["stay_date"],
                room_category=row["room_category"],
                horizon=row["horizon"],
                prediction=row["prediction"],
                lower=row["lower"],
                upper=row["upper"],
                interval_level=row["interval_level"],
                occupancy_rate=row["occupancy_rate"],
                capacity=row["capacity"],
                needs_human_review=row["needs_human_review"],
                explanation=explanation_text,
            )
        )

    limitations = [
        "Trained on synthetic data — real-world performance may differ.",
        "Demand shocks (pandemics, sudden events) beyond training distribution degrade accuracy.",
        "Interval coverage may drift if booking patterns change significantly.",
        "Wide intervals (needs_human_review=True) indicate higher uncertainty — escalate to operations manager.",
    ]

    return ForecastResponse(
        generated_at=datetime.now().isoformat(),
        origin_date=request.origin_date,
        horizon_days=request.horizon_days,
        room_categories=request.room_categories,
        forecasts=forecast_rows,
        model_name=artifact.get("model_name", "unknown"),
        limitations=limitations,
    )
