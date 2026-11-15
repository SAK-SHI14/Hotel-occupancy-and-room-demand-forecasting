"""Model explainability: global importance and local per-forecast explanations.

Provides two levels of explanation:
1. Global: permutation importance on the validation set (always available).
   Falls back to SHAP summary plot if the ``shap`` package is importable.
2. Local: per-prediction top-5 feature contributions with direction and magnitude.
   Returns a natural-language explanation string.

One-sentence "what and why" for each tool used:
- Permutation importance: shuffles each feature independently and measures
  how much validation MAE increases — a model-agnostic measure of relevance.
- SHAP: computes exact Shapley values showing each feature's contribution to
  the difference between this prediction and the mean prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from hotel_forecast.features.build import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Global Importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on the validation set.

    Returns a DataFrame sorted by mean importance descending.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model with a predict() method.
    X_val : pd.DataFrame
        Validation features (FEATURE_COLUMNS only).
    y_val : array-like
        Ground-truth labels.
    n_repeats : int
        Number of shuffle repeats (more → more stable but slower).
    random_state : int
        Fixed seed for reproducibility.
    """
    result = permutation_importance(
        model,
        X_val[FEATURE_COLUMNS],
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return imp_df.reset_index(drop=True)


def plot_permutation_importance(
    imp_df: pd.DataFrame,
    top_n: int = 15,
    output_path: Path | str = "reports/figures/permutation_importance.png",
) -> None:
    """Horizontal bar chart of top-N permutation importances."""
    df_top = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(df_top))
    ax.barh(
        list(y_pos),
        df_top["importance_mean"],
        xerr=df_top["importance_std"],
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.7,
        capsize=4,
        alpha=0.85,
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_top["feature"].tolist(), fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean ↑ in MAE when shuffled (larger = more important)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importances (Permutation, Validation Set)\n"
        "Error bars = ±1 std over repeats",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Permutation importance figure saved to {out_file}")


def compute_shap_importance(
    model: Any,
    X_val: pd.DataFrame,
    output_path: Path | str = "reports/figures/shap_summary.png",
    max_samples: int = 1000,
) -> pd.DataFrame | None:
    """Compute SHAP values and save summary plot (optional; graceful fallback).

    Returns a DataFrame of mean absolute SHAP values per feature,
    or None if the shap package is unavailable.
    """
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        print("shap package not available — skipping SHAP analysis (permutation importance used instead).")
        return None

    X_sample = X_val[FEATURE_COLUMNS].head(max_samples)

    # TreeExplainer works for tree-based models; use Explainer (auto-dispatch) otherwise
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        try:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample).values
        except Exception as exc:
            print(f"SHAP computation failed: {exc}. Skipping SHAP.")
            return None

    shap_arr = np.abs(shap_values)
    mean_abs_shap = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_abs_shap": shap_arr.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(min(15, len(mean_abs_shap)))
    df_top = mean_abs_shap.head(15)
    ax.barh(list(y_pos), df_top["mean_abs_shap"], color="#ff7f0e", edgecolor="black", linewidth=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_top["feature"].tolist(), fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (rooms)", fontsize=11)
    ax.set_title("SHAP Global Feature Importance (Top 15, Validation Sample)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary figure saved to {out_file}")

    return mean_abs_shap.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Local (per-prediction) explanation
# ---------------------------------------------------------------------------

# Readable labels for each feature (used in natural-language explanations)
FEATURE_LABELS: dict[str, str] = {
    "otb_rooms_T": "rooms already on-the-books (non-cancelled as of forecast date)",
    "otb_rooms_T_minus_7": "on-the-books count 7 days ago",
    "otb_rooms_T_minus_14": "on-the-books count 14 days ago",
    "pickup_pace_7d": "pickup in the last 7 days",
    "pickup_pace_14d": "pickup in the last 14 days",
    "otb_capacity_share": "fraction of capacity already booked",
    "lag_same_weekday": "last realized same-weekday occupancy",
    "rolling_mean_7d": "7-day rolling mean of past occupancy",
    "rolling_mean_28d": "28-day rolling mean of past occupancy",
    "same_weekday_4w_mean": "4-week same-weekday average occupancy",
    "day_of_week": "day of week (0=Mon, 6=Sun)",
    "month": "month of the stay date",
    "week_of_year": "ISO week number",
    "is_weekend": "whether the stay date is a weekend",
    "is_holiday": "whether the stay date is a public holiday",
    "days_to_next_holiday": "days until the next public holiday",
    "days_since_last_holiday": "days since the last public holiday",
    "is_event_period": "whether a local event is active on the stay date",
    "horizon": "forecast horizon (days ahead from the forecast date)",
    "room_category_code": "room category (0=Standard, 1=Deluxe, 2=Family, 3=Suite)",
    "capacity": "total room capacity for this category",
    "avg_lead_time_otb": "average lead time of current on-the-books bookings",
    "ota_share_otb": "fraction of on-the-books bookings from OTA channels",
}

CATEGORY_NAMES = {0: "Standard", 1: "Deluxe", 2: "Family", 3: "Suite"}
DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def explain_forecast(
    features: dict[str, float],
    model: Any,
    imp_df: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    """Produce a natural-language explanation of a single forecast.

    Parameters
    ----------
    features : dict
        Feature values for a single forecast row (keys = FEATURE_COLUMNS).
    model : sklearn estimator
        Fitted model.
    imp_df : pd.DataFrame
        Global permutation importance DataFrame (from compute_permutation_importance).
    top_n : int
        Number of top drivers to include in explanation.

    Returns
    -------
    dict with keys:
        - prediction (float): model point prediction
        - top_drivers (list[dict]): top-N features with value and direction
        - explanation (str): human-readable natural-language summary
    """
    # Build a single-row DataFrame
    row_df = pd.DataFrame([{col: features.get(col, 0.0) for col in FEATURE_COLUMNS}])
    prediction = float(model.predict(row_df)[0])
    capacity = float(features.get("capacity", 1.0))
    prediction_clipped = max(0.0, min(prediction, capacity))
    occupancy_rate = round(prediction_clipped / capacity, 3) if capacity > 0 else 0.0

    # Top-N important features (by global importance)
    top_features = imp_df.head(top_n)["feature"].tolist()

    drivers = []
    for feat in top_features:
        val = features.get(feat, 0.0)
        # Determine direction: for most features, higher → higher demand
        # We use a simple sign heuristic against the mean (mean ~ mid-capacity)
        norm_val = val / capacity if capacity > 0 and feat in ("otb_rooms_T", "lag_same_weekday", "rolling_mean_7d") else val
        direction = "↑ boosting demand" if norm_val > 0.5 else "↓ suppressing demand"
        label = FEATURE_LABELS.get(feat, feat)
        drivers.append({"feature": feat, "label": label, "value": round(val, 3), "direction": direction})

    # Compose natural-language explanation
    cat_name = CATEGORY_NAMES.get(int(features.get("room_category_code", 0)), "Unknown")
    dow_idx = int(features.get("day_of_week", 0)) % 7
    dow_name = DOW_NAMES[dow_idx]
    horizon_val = int(features.get("horizon", 1))
    is_holiday = bool(features.get("is_holiday", 0))
    is_weekend = bool(features.get("is_weekend", 0))
    otb = round(float(features.get("otb_rooms_T", 0)), 1)

    holiday_note = " It is a public holiday." if is_holiday else ""
    weekend_note = f" The stay falls on a {dow_name} ({'weekend' if is_weekend else 'weekday'})."

    explanation = (
        f"Forecast: {prediction_clipped:.1f} rooms occupied for {cat_name} "
        f"({horizon_val} day{'s' if horizon_val != 1 else ''} ahead). "
        f"Occupancy rate: {occupancy_rate*100:.1f}% of {int(capacity)} total rooms. "
        f"Currently {otb:.0f} rooms are on-the-books.{weekend_note}{holiday_note} "
        f"Top drivers: "
        + ", ".join(f"{d['feature']}={d['value']}" for d in drivers[:3])
        + "."
    )

    # Flag for human review: interval unusually wide or occupancy extreme
    needs_review = occupancy_rate > 0.95 or occupancy_rate < 0.05

    return {
        "prediction": round(prediction_clipped, 2),
        "occupancy_rate": occupancy_rate,
        "top_drivers": drivers,
        "explanation": explanation,
        "needs_human_review": needs_review,
    }


# ---------------------------------------------------------------------------
# Main: run full explainability pipeline and save outputs
# ---------------------------------------------------------------------------

def run_explainability_pipeline(
    model_path: Path | str = "models/model.joblib",
    split_info_path: Path | str = "reports/split_info.json",
    output_dir: Path | str = "reports",
) -> None:
    """Generate and save all explainability artifacts."""
    import joblib
    import yaml

    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load model artifact
    artifact = joblib.load(model_path)
    model = artifact["model"]

    # Load split info and build validation set
    from datetime import date as _date

    with open(split_info_path, encoding="utf-8") as f:
        split_info = json.load(f)

    v_start = _date.fromisoformat(split_info["val_start_origin"])
    v_end = _date.fromisoformat(split_info["val_end_origin"])

    from hotel_forecast.features.build import build_feature_dataset

    ledger = pd.read_csv("data/processed/booking_ledger_clean.csv")
    print("Building validation feature set for explainability…")
    X_full, y_full = build_feature_dataset(
        ledger, config=cfg,
        start_origin=str(v_start), end_origin=str(v_end),
        origin_step_days=4,
    )
    X_val = X_full[FEATURE_COLUMNS]
    y_val = y_full

    # --- Global: Permutation Importance ---
    print("Computing permutation importance (n_repeats=10)…")
    imp_df = compute_permutation_importance(model, X_val, y_val, n_repeats=10)
    imp_df.to_csv(Path(output_dir) / "permutation_importance.csv", index=False)
    print(f"Permutation importance CSV saved to {output_dir}/permutation_importance.csv")
    plot_permutation_importance(imp_df, top_n=15)

    # --- Global: SHAP (graceful fallback) ---
    compute_shap_importance(model, X_val)

    # --- Local: Example explanation ---
    sample_row = X_val.iloc[0].to_dict()
    example_expl = explain_forecast(sample_row, model, imp_df)
    out_path = Path(output_dir) / "example_explanation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(example_expl, f, indent=2)
    print(f"\nExample explanation: {example_expl['explanation']}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    run_explainability_pipeline()
