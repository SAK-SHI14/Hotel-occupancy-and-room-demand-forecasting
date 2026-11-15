"""Save the best model (retrained on train+val) and generate error analysis.

Usage
-----
    python -m hotel_forecast.models.save_best

What this does (in order):
1. Loads experiments.csv to identify the best model on validation WAPE.
2. Retrains that model on train+val combined (no test data).
3. Calibrates SplitConformalPredictor on the validation residuals.
4. Saves models/model.joblib with a rich artifact dict.
5. Writes models/model_card.json.
6. Generates error analysis figures:
   - error_by_horizon.png   (MAE/WAPE per h=1..14)
   - error_by_category.png  (MAE/WAPE per room category)
   - worst_forecasts.csv    (worst 10 predictions)
7. Saves reports/metrics.json with validation metrics for the final model.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hotel_forecast.evaluate import compute_metrics
from hotel_forecast.features.build import FEATURE_COLUMNS, build_feature_dataset
from hotel_forecast.models.baselines import SeasonalMovingAverageBaseline, SeasonalNaiveBaseline
from hotel_forecast.models.conformal import (
    SplitConformalPredictor,
    plot_interval_width_by_horizon,
)

REPO_ROOT = Path(__file__).parent.parent.parent.parent  # src/hotel_forecast/models -> root
CONFIGS_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path | str = CONFIGS_PATH) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(model_name: str) -> Any:
    """Instantiate a model from its experiment CSV name string."""
    name = model_name.lower()
    if "ridge" in name:
        # e.g. "Ridge (alpha=10.0)" -> extract alpha
        try:
            alpha = float(name.split("alpha=")[1].rstrip(")"))
        except (IndexError, ValueError):
            alpha = 1.0
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=42))])
    if "randomforest" in name:
        # "RandomForest (n=150, d=15)"
        try:
            n_est = int(name.split("n=")[1].split(",")[0])
            depth = int(name.split("d=")[1].rstrip(")"))
        except (IndexError, ValueError):
            n_est, depth = 100, 10
        return RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
    if "histgbm" in name or "gradientboost" in name:
        # "HistGBM (lr=0.1, d=10, leaf=15)"
        try:
            lr = float(name.split("lr=")[1].split(",")[0])
            d = int(name.split("d=")[1].split(",")[0])
            leaf = int(name.split("leaf=")[1].rstrip(")"))
        except (IndexError, ValueError):
            lr, d, leaf = 0.1, 10, 15
        return HistGradientBoostingRegressor(
            learning_rate=lr, max_depth=d, min_samples_leaf=leaf, random_state=42
        )
    # Default fallback
    return HistGradientBoostingRegressor(learning_rate=0.1, max_depth=10, min_samples_leaf=15, random_state=42)


def _data_hash(df: pd.DataFrame) -> str:
    """SHA256 hash of the data as a reproducibility fingerprint."""
    raw_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Error analysis figures
# ---------------------------------------------------------------------------

def _plot_error_by_horizon(X: pd.DataFrame, y: pd.Series, preds: np.ndarray, out_dir: Path) -> None:
    """MAE and WAPE for each h=1..14 on the validation set."""
    horizons = X["horizon"].to_numpy(dtype=int)
    y_arr = np.asarray(y, dtype=float)
    abs_err = np.abs(preds - y_arr)

    rows = []
    for h in range(1, 15):
        mask = horizons == h
        if mask.sum() == 0:
            continue
        mae_h = float(np.mean(abs_err[mask]))
        sum_true = float(np.sum(y_arr[mask]))
        wape_h = float(np.sum(abs_err[mask]) / sum_true) * 100 if sum_true > 0 else 0.0
        rows.append({"horizon": h, "mae": mae_h, "wape_pct": wape_h, "n": int(mask.sum())})
    df_h = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(df_h["horizon"], df_h["mae"], color="#1f77b4", edgecolor="black", linewidth=0.6)
    ax1.set_xlabel("Forecast Horizon (days ahead)", fontsize=11)
    ax1.set_ylabel("MAE (rooms)", fontsize=11)
    ax1.set_title("Mean Absolute Error by Horizon", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(1, 15))

    ax2.bar(df_h["horizon"], df_h["wape_pct"], color="#d62728", edgecolor="black", linewidth=0.6)
    ax2.set_xlabel("Forecast Horizon (days ahead)", fontsize=11)
    ax2.set_ylabel("WAPE (%)", fontsize=11)
    ax2.set_title("WAPE (%) by Horizon", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(1, 15))

    plt.suptitle("Error Analysis: Degradation with Forecast Horizon", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "error_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Error-by-horizon figure saved to {out_dir / 'error_by_horizon.png'}")


def _plot_error_by_category(
    X: pd.DataFrame,
    y: pd.Series,
    preds: np.ndarray,
    out_dir: Path,
) -> None:
    """MAE and WAPE for each room category."""
    y_arr = np.asarray(y, dtype=float)
    abs_err = np.abs(preds - y_arr)

    # room_category_code: Standard=0, Deluxe=1, Family=2, Suite=3
    cat_map = {0: "Standard", 1: "Deluxe", 2: "Family", 3: "Suite"}
    cats_enc = X["room_category_code"].to_numpy(dtype=int)

    rows = []
    for enc, name in cat_map.items():
        mask = cats_enc == enc
        if mask.sum() == 0:
            continue
        mae_c = float(np.mean(abs_err[mask]))
        sum_true = float(np.sum(y_arr[mask]))
        wape_c = float(np.sum(abs_err[mask]) / sum_true) * 100 if sum_true > 0 else 0.0
        rows.append({"category": name, "mae": mae_c, "wape_pct": wape_c, "n": int(mask.sum())})
    df_c = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    ax1.bar(df_c["category"], df_c["mae"], color=colors[: len(df_c)], edgecolor="black", linewidth=0.8)
    for i, row in df_c.iterrows():
        ax1.text(i, row["mae"] + 0.05, f"{row['mae']:.2f}", ha="center", fontsize=10)
    ax1.set_xlabel("Room Category", fontsize=11)
    ax1.set_ylabel("MAE (rooms)", fontsize=11)
    ax1.set_title("MAE by Room Category", fontsize=12, fontweight="bold")

    ax2.bar(df_c["category"], df_c["wape_pct"], color=colors[: len(df_c)], edgecolor="black", linewidth=0.8)
    for i, row in df_c.iterrows():
        ax2.text(i, row["wape_pct"] + 0.2, f"{row['wape_pct']:.1f}%", ha="center", fontsize=10)
    ax2.set_xlabel("Room Category", fontsize=11)
    ax2.set_ylabel("WAPE (%)", fontsize=11)
    ax2.set_title("WAPE by Room Category", fontsize=12, fontweight="bold")

    plt.suptitle("Error Analysis: Variation Across Room Categories", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "error_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Error-by-category figure saved to {out_dir / 'error_by_category.png'}")


def _save_worst_forecasts(
    X: pd.DataFrame,
    y: pd.Series,
    preds: np.ndarray,
    out_dir: Path,
    n: int = 10,
) -> None:
    """Write a CSV of the worst n predictions (highest absolute error)."""
    y_arr = np.asarray(y, dtype=float)
    abs_err = np.abs(preds - y_arr)

    out_df = X[["horizon", "room_category_code", "capacity", "otb_rooms_T", "lag_same_weekday"]].copy()
    out_df["actual"] = y_arr
    out_df["predicted"] = np.round(preds, 2)
    out_df["abs_error"] = np.round(abs_err, 2)
    out_df["error_direction"] = np.where(preds > y_arr, "over-forecast", "under-forecast")

    worst = out_df.nlargest(n, "abs_error")
    out_path = out_dir / "worst_forecasts.csv"
    worst.to_csv(out_path, index=False)
    print(f"Worst {n} forecasts saved to {out_path}")


def _plot_weekend_holiday_breakdown(
    X: pd.DataFrame,
    y: pd.Series,
    preds: np.ndarray,
    out_dir: Path,
) -> None:
    """MAE breakdown: weekday vs weekend, holiday vs non-holiday."""
    y_arr = np.asarray(y, dtype=float)
    abs_err = np.abs(preds - y_arr)

    groups = {
        "Weekday": X["is_weekend"].to_numpy() == 0,
        "Weekend": X["is_weekend"].to_numpy() == 1,
        "Non-Holiday": X["is_holiday"].to_numpy() == 0,
        "Holiday": X["is_holiday"].to_numpy() == 1,
    }

    rows = []
    for label, mask in groups.items():
        if mask.sum() == 0:
            continue
        mae_g = float(np.mean(abs_err[mask]))
        sum_true = float(np.sum(y_arr[mask]))
        wape_g = float(np.sum(abs_err[mask]) / sum_true) * 100 if sum_true > 0 else 0.0
        rows.append({"group": label, "mae": mae_g, "wape_pct": wape_g, "n": int(mask.sum())})
    df_g = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = range(len(df_g))
    bars = ax.bar(x_pos, df_g["mae"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], edgecolor="black")
    for bar, row in zip(bars, df_g.itertuples(), strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{row.mae:.2f}\n({row.wape_pct:.1f}%)", ha="center", fontsize=9)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(df_g["group"].tolist(), fontsize=11)
    ax.set_ylabel("MAE (rooms)", fontsize=11)
    ax.set_title("Error Breakdown: Weekend/Holiday vs Weekday/Non-Holiday", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "error_weekend_holiday.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Weekend/holiday breakdown figure saved to {out_dir / 'error_weekend_holiday.png'}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def save_best_model(
    experiments_path: Path | str = "reports/experiments.csv",
    split_info_path: Path | str = "reports/split_info.json",
    model_out_path: Path | str = "models/model.joblib",
    model_card_path: Path | str = "models/model_card.json",
    metrics_path: Path | str = "reports/metrics.json",
    figures_dir: Path | str = "reports/figures",
) -> None:
    """Select, retrain and serialize the best model; run error analysis."""
    cfg = load_config()
    c_under = float(cfg["metrics"]["asymmetric_cost"]["c_under"])
    c_over = float(cfg["metrics"]["asymmetric_cost"]["c_over"])

    # ------------------------------------------------------------------ #
    # 1. Load split info                                                   #
    # ------------------------------------------------------------------ #
    with open(split_info_path, encoding="utf-8") as f:
        split_info = json.load(f)

    t_start = date.fromisoformat(split_info["train_start_origin"])
    t_end = date.fromisoformat(split_info["train_end_origin"])
    v_start = date.fromisoformat(split_info["val_start_origin"])
    v_end = date.fromisoformat(split_info["val_end_origin"])

    # ------------------------------------------------------------------ #
    # 2. Build combined train+val feature dataset                          #
    # ------------------------------------------------------------------ #
    clean_path = Path("data/processed/booking_ledger_clean.csv")
    ledger_df = pd.read_csv(clean_path)

    print("Building train+val feature dataset (origin_step_days=2)…")
    X_full, y_full = build_feature_dataset(ledger_df, config=cfg, origin_step_days=2)

    # Masks for the three partitions
    origins = X_full["meta_origin_date"]
    train_mask = origins <= t_end
    val_mask = (origins >= v_start) & (origins <= v_end)
    trainval_mask = train_mask | val_mask

    X_train = X_full.loc[train_mask, FEATURE_COLUMNS]
    y_train = y_full.loc[train_mask]
    X_val = X_full.loc[val_mask, FEATURE_COLUMNS]
    y_val = y_full.loc[val_mask]
    X_trainval = X_full.loc[trainval_mask, FEATURE_COLUMNS]
    y_trainval = y_full.loc[trainval_mask]

    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples:   {len(X_val):,}")
    print(f"  Train+Val:     {len(X_trainval):,}")

    # ------------------------------------------------------------------ #
    # 3. Select best model from experiments.csv on validation WAPE         #
    # ------------------------------------------------------------------ #
    exp_df = pd.read_csv(experiments_path)
    # Exclude baseline rows from selection
    candidate_df = exp_df[~exp_df["model_family"].str.startswith("Baseline")].copy()
    best_row = candidate_df.loc[candidate_df["wape_pct"].idxmin()]

    best_name = best_row["model_name"]
    best_wape = best_row["wape_pct"]
    print(f"\nBest model on validation WAPE: '{best_name}' | WAPE={best_wape:.2f}%")

    # ------------------------------------------------------------------ #
    # 4. Retrain best model on train+val                                   #
    # ------------------------------------------------------------------ #
    print("Retraining best model on train+val combined…")
    final_model = _build_model(best_name)
    final_model.fit(X_trainval, y_trainval)

    # Validation predictions for conformal calibration
    val_preds = np.clip(final_model.predict(X_val), 0.0, X_val["capacity"].to_numpy())

    # Also get validation metrics for the final model (before conformal)
    b1 = SeasonalNaiveBaseline().fit(X_train, y_train)
    b1_preds_val = b1.predict(X_val)
    b1_metrics_val = compute_metrics(y_val, b1_preds_val, c_under, c_over)

    b2 = SeasonalMovingAverageBaseline().fit(X_train, y_train)
    b2_preds_val = b2.predict(X_val)
    b2_metrics_val = compute_metrics(y_val, b2_preds_val, c_under, c_over, b1_mae=b1_metrics_val["mae"])

    val_metrics = compute_metrics(y_val, val_preds, c_under, c_over, b1_mae=b1_metrics_val["mae"])

    print(f"  Val MAE={val_metrics['mae']:.3f} | WAPE={val_metrics['wape_pct']:.2f}% | Skill={val_metrics.get('skill_score_vs_b1', 0):.3f}")

    # ------------------------------------------------------------------ #
    # 5. Calibrate conformal predictor on validation                       #
    # ------------------------------------------------------------------ #
    print("\nCalibrating split conformal predictor on validation residuals…")
    conformal = SplitConformalPredictor(alpha=0.20)
    conformal.calibrate(X_val, y_val, final_model)

    # ------------------------------------------------------------------ #
    # 6. Error analysis figures (on validation — test stays sealed)        #
    # ------------------------------------------------------------------ #
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_error_by_horizon(X_val, y_val, val_preds, fig_dir)
    _plot_error_by_category(X_val, y_val, val_preds, fig_dir)
    _save_worst_forecasts(X_val, y_val, val_preds, fig_dir)
    _plot_weekend_holiday_breakdown(X_val, y_val, val_preds, fig_dir)
    plot_interval_width_by_horizon(conformal)

    # ------------------------------------------------------------------ #
    # 7. Save model artifact                                               #
    # ------------------------------------------------------------------ #
    # Determine model type string
    if "ridge" in best_name.lower():
        model_type_str = "Ridge"
    elif "randomforest" in best_name.lower():
        model_type_str = "RandomForestRegressor"
    else:
        model_type_str = "HistGradientBoostingRegressor"

    artifact: dict[str, Any] = {
        "model": final_model,
        "conformal": conformal.to_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "model_type": model_type_str,
        "model_name": best_name,
        "training_date": datetime.now().isoformat(),
        "train_period": {"start": str(t_start), "end": str(t_end)},
        "val_period": {"start": str(v_start), "end": str(v_end)},
        "val_metrics": val_metrics,
        "b1_val_metrics": b1_metrics_val,
        "b2_val_metrics": b2_metrics_val,
        "data_hash": _data_hash(ledger_df),
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    model_out = Path(model_out_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)
    print(f"\nModel artifact saved to {model_out}")

    # ------------------------------------------------------------------ #
    # 8. Write model card                                                  #
    # ------------------------------------------------------------------ #
    model_card = {
        "model_name": best_name,
        "model_type": model_type_str,
        "task": "Regression — hotel room demand forecasting",
        "prediction_unit": "(stay_date, room_category)",
        "target": "occupied_room_nights",
        "forecast_horizons": "1 to 14 days",
        "feature_count": len(FEATURE_COLUMNS),
        "training_samples": len(X_trainval),
        "train_period": {"start": str(t_start), "end": str(t_end)},
        "val_period": {"start": str(v_start), "end": str(v_end)},
        "training_date": artifact["training_date"],
        "data_hash": artifact["data_hash"],
        "val_metrics": val_metrics,
        "b1_baseline_val": b1_metrics_val,
        "b2_baseline_val": b2_metrics_val,
        "conformal_coverage_target": "80%",
        "conformal_quantiles_per_bucket": conformal.quantiles_,
        "intended_use": "Decision-support tool for hotel operations managers. Outputs are probabilistic estimates, not guarantees.",
        "limitations": [
            "Trained on synthetic data — real-world performance may differ.",
            "Demand shocks (pandemics, sudden events) beyond the training distribution may degrade accuracy.",
            "Interval coverage may drift if booking patterns change significantly.",
            "No PII processed — data is entirely synthetic and aggregated.",
        ],
        "responsible_ai_notes": "Wide intervals (> 2x mean) trigger a 'needs_human_review' flag in the API. Tool is advisory; final decisions are human-made.",
    }

    card_out = Path(model_card_path)
    card_out.parent.mkdir(parents=True, exist_ok=True)
    with open(card_out, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)
    print(f"Model card written to {card_out}")

    # ------------------------------------------------------------------ #
    # 9. Update reports/metrics.json with validation results               #
    # ------------------------------------------------------------------ #
    out_metrics = Path(metrics_path)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if out_metrics.exists():
        try:
            with open(out_metrics, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    existing["best_model"] = best_name
    existing["validation_metrics"] = val_metrics
    existing["b1_baseline_val"] = b1_metrics_val
    existing["b2_baseline_val"] = b2_metrics_val
    existing["conformal_quantiles"] = conformal.quantiles_

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"Metrics summary updated in {out_metrics}")

    print("\n" + "=" * 70)
    print("PHASE 5+6 COMPLETE: Best Model Saved and Conformal Calibrated")
    print("=" * 70)
    print(f"  Best model  : {best_name}")
    print(f"  Val WAPE    : {val_metrics['wape_pct']:.2f}%")
    print(f"  Val MAE     : {val_metrics['mae']:.3f} rooms")
    print(f"  Skill vs B1 : {val_metrics.get('skill_score_vs_b1', 0)*100:.1f}%")
    print(f"  Conformal   : {conformal.quantiles_}")
    print("=" * 70)


if __name__ == "__main__":
    save_best_model()
