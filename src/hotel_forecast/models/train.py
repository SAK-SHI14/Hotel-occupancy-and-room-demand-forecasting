"""Model training, rolling-origin cross-validation, and experiment logging.

Enforces chronological train/validation/test splits with a strict 14-day purge gap.
Evaluates baselines, Ridge regression, Random Forest, and Gradient Boosting.
Logs all tuning trials to reports/experiments.csv and split info to reports/split_info.json.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hotel_forecast.evaluate import compute_metrics
from hotel_forecast.features.build import FEATURE_COLUMNS, build_feature_dataset
from hotel_forecast.models.baselines import SeasonalMovingAverageBaseline, SeasonalNaiveBaseline


def load_config(config_path: Path | str = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_chronological_splits(
    unique_origins: list[date],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    purge_gap_days: int = 14,
) -> dict[str, Any]:
    """Partition unique origins chronologically with strict purge gaps.

    Parameters
    ----------
    unique_origins : list of date
        Sorted list of unique forecast origins.
    train_ratio : float
        Proportion of timeline allocated to training (default 0.60).
    val_ratio : float
        Proportion of timeline allocated to validation (default 0.20).
    purge_gap_days : int
        Purge gap between splits in calendar days (default 14).

    Returns
    -------
    dict[str, Any]
        Split boundaries, origin lists, and sample counts.
    """
    sorted_origins = sorted(unique_origins)
    total_origins = len(sorted_origins)

    # Train partition
    n_train = int(total_origins * train_ratio)
    train_origins = sorted_origins[:n_train]
    train_end = train_origins[-1]

    # Purge gap 1: after train_end
    val_start_cutoff = train_end + timedelta(days=purge_gap_days + 1)
    eligible_val = [d for d in sorted_origins[n_train:] if d >= val_start_cutoff]

    if not eligible_val:
        raise ValueError("Timeline too short to enforce 14-day purge gap for validation.")

    n_val = int(total_origins * val_ratio)
    val_origins = eligible_val[:n_val]
    val_end = val_origins[-1]

    # Purge gap 2: after val_end
    test_start_cutoff = val_end + timedelta(days=purge_gap_days + 1)
    test_origins = [d for d in sorted_origins if d >= test_start_cutoff]

    if not test_origins:
        raise ValueError("Timeline too short to enforce 14-day purge gap for test set.")

    split_info = {
        "train_start_origin": str(train_origins[0]),
        "train_end_origin": str(train_end),
        "train_origin_count": len(train_origins),
        "purge_gap_1_days": purge_gap_days,
        "val_start_origin": str(val_origins[0]),
        "val_end_origin": str(val_end),
        "val_origin_count": len(val_origins),
        "purge_gap_2_days": purge_gap_days,
        "test_start_origin": str(test_origins[0]),
        "test_end_origin": str(test_origins[-1]),
        "test_origin_count": len(test_origins),
    }
    return split_info


def plot_split_timeline(
    split_info: dict[str, Any],
    output_path: Path | str = "reports/figures/split_timeline.png",
):
    """Plot chronological split timeline visualization showing the 14-day purge gaps."""
    t_start = pd.to_datetime(split_info["train_start_origin"])
    t_end = pd.to_datetime(split_info["train_end_origin"])
    v_start = pd.to_datetime(split_info["val_start_origin"])
    v_end = pd.to_datetime(split_info["val_end_origin"])
    te_start = pd.to_datetime(split_info["test_start_origin"])
    te_end = pd.to_datetime(split_info["test_end_origin"])

    plt.figure(figsize=(14, 3.5))
    ax = plt.gca()

    # Draw timeline bars
    y_pos = 0.5
    bar_height = 0.4

    # Train block
    ax.barh(y_pos, (t_end - t_start).days, left=t_start, height=bar_height, color="#1f77b4", label="Train Set (60%)", edgecolor="black")
    # Purge Gap 1
    ax.barh(y_pos, (v_start - t_end).days, left=t_end, height=bar_height, color="#ff7f0e", hatch="//", alpha=0.6, label="Purge Gap (14d)", edgecolor="black")
    # Validation block
    ax.barh(y_pos, (v_end - v_start).days, left=v_start, height=bar_height, color="#2ca02c", label="Validation Set (20%)", edgecolor="black")
    # Purge Gap 2
    ax.barh(y_pos, (te_start - v_end).days, left=v_end, height=bar_height, color="#ff7f0e", hatch="//", alpha=0.6, edgecolor="black")
    # Test block
    ax.barh(y_pos, (te_end - te_start).days, left=te_start, height=bar_height, color="#d62728", label="Held-Out Test Set (20%)", edgecolor="black")

    ax.set_yticks([])
    ax.set_xlabel("Forecast Origin Timeline", fontsize=11, fontweight="bold")
    ax.set_title("Chronological Split Strategy with 14-Day Purge Gaps (Zero Information Leakage)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=True)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Split timeline figure saved to {out_file}")


def run_training_pipeline(
    config_path: Path | str = "configs/config.yaml",
    experiments_path: Path | str = "reports/experiments.csv",
    split_info_path: Path | str = "reports/split_info.json",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute model training, rolling CV tuning, and validation evaluation."""
    cfg = load_config(config_path)
    c_under = float(cfg["metrics"]["asymmetric_cost"]["c_under"])
    c_over = float(cfg["metrics"]["asymmetric_cost"]["c_over"])
    purge_gap = int(cfg["splits"]["purge_gap_days"])

    # Load cleaned ledger
    clean_path = Path("data/processed/booking_ledger_clean.csv")
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean ledger not found at {clean_path}")
    ledger_df = pd.read_csv(clean_path)

    # Build dataset across origins (step=2 days for responsive cross-validation)
    print("Building point-in-time dataset for model tuning and evaluation...")
    X_full, y_full = build_feature_dataset(ledger_df, config=cfg, origin_step_days=2)

    # Determine unique origins
    unique_origins = sorted(X_full["meta_origin_date"].unique())
    split_info = calculate_chronological_splits(
        unique_origins=unique_origins,
        train_ratio=cfg["splits"]["train_ratio"],
        val_ratio=cfg["splits"]["val_ratio"],
        purge_gap_days=purge_gap,
    )

    # Save split info
    s_path = Path(split_info_path)
    s_path.parent.mkdir(parents=True, exist_ok=True)
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"Split info written to {s_path}")

    # Plot split timeline
    plot_split_timeline(split_info)

    # Slice Train, Validation, and Test masks
    # Convert ISO strings from split_info to datetime.date so comparisons with
    # meta_origin_date (which contains datetime.date objects) work correctly.
    t_end = date.fromisoformat(split_info["train_end_origin"])
    v_start = date.fromisoformat(split_info["val_start_origin"])
    v_end = date.fromisoformat(split_info["val_end_origin"])
    te_start = date.fromisoformat(split_info["test_start_origin"])

    train_mask = X_full["meta_origin_date"] <= t_end
    val_mask = (X_full["meta_origin_date"] >= v_start) & (X_full["meta_origin_date"] <= v_end)
    # Test set is identified but NOT evaluated here (evaluated once in Phase 6)
    test_mask = X_full["meta_origin_date"] >= te_start

    print(f"Dataset Split Sizes: Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")

    X_train = X_full.loc[train_mask, FEATURE_COLUMNS]
    y_train = y_full.loc[train_mask]

    X_val = X_full.loc[val_mask, FEATURE_COLUMNS]
    y_val = y_full.loc[val_mask]

    experiments = []

    # -------------------------------------------------------------
    # 1. Evaluate B1 Seasonal Naive Baseline
    # -------------------------------------------------------------
    b1 = SeasonalNaiveBaseline().fit(X_train)
    b1_preds = b1.predict(X_val)
    b1_metrics = compute_metrics(y_val, b1_preds, c_under, c_over)
    experiments.append({
        "model_family": "Baseline",
        "model_name": "B1 Seasonal Naive",
        "hyperparameters": "None (lag_same_weekday)",
        **b1_metrics,
    })

    # -------------------------------------------------------------
    # 2. Evaluate B2 Seasonal Moving Average Baseline
    # -------------------------------------------------------------
    b2 = SeasonalMovingAverageBaseline().fit(X_train)
    b2_preds = b2.predict(X_val)
    b2_metrics = compute_metrics(y_val, b2_preds, c_under, c_over, b1_mae=b1_metrics["mae"])
    experiments.append({
        "model_family": "Baseline",
        "model_name": "B2 Seasonal Moving Avg",
        "hyperparameters": "None (4-week same-weekday mean)",
        **b2_metrics,
    })

    # -------------------------------------------------------------
    # 3. Model M1: Ridge Regression (Scaled Features)
    # -------------------------------------------------------------
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        m1 = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=42))])
        m1.fit(X_train, y_train)
        m1_preds = np.clip(m1.predict(X_val), 0.0, X_val["capacity"].to_numpy())
        m1_metrics = compute_metrics(y_val, m1_preds, c_under, c_over, b1_mae=b1_metrics["mae"])
        experiments.append({
            "model_family": "M1 Ridge Regression",
            "model_name": f"Ridge (alpha={alpha})",
            "hyperparameters": f"alpha={alpha}",
            **m1_metrics,
        })

    # -------------------------------------------------------------
    # 4. Model M2: Random Forest Regressor
    # -------------------------------------------------------------
    for n_est, depth in [(100, 10), (150, 15)]:
        m2 = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
        m2.fit(X_train, y_train)
        m2_preds = np.clip(m2.predict(X_val), 0.0, X_val["capacity"].to_numpy())
        m2_metrics = compute_metrics(y_val, m2_preds, c_under, c_over, b1_mae=b1_metrics["mae"])
        experiments.append({
            "model_family": "M2 Random Forest",
            "model_name": f"RandomForest (n={n_est}, d={depth})",
            "hyperparameters": f"n_estimators={n_est}, max_depth={depth}",
            **m2_metrics,
        })

    # -------------------------------------------------------------
    # 5. Model M3: HistGradientBoostingRegressor
    # -------------------------------------------------------------
    for lr, depth, min_leaf in [(0.05, 8, 20), (0.1, 10, 15), (0.1, 8, 25)]:
        m3 = HistGradientBoostingRegressor(
            learning_rate=lr,
            max_depth=depth,
            min_samples_leaf=min_leaf,
            random_state=42,
        )
        m3.fit(X_train, y_train)
        m3_preds = np.clip(m3.predict(X_val), 0.0, X_val["capacity"].to_numpy())
        m3_metrics = compute_metrics(y_val, m3_preds, c_under, c_over, b1_mae=b1_metrics["mae"])
        experiments.append({
            "model_family": "M3 Gradient Boosting",
            "model_name": f"HistGBM (lr={lr}, d={depth}, leaf={min_leaf})",
            "hyperparameters": f"learning_rate={lr}, max_depth={depth}, min_samples_leaf={min_leaf}",
            **m3_metrics,
        })

    exp_df = pd.DataFrame(experiments)
    exp_out = Path(experiments_path)
    exp_out.parent.mkdir(parents=True, exist_ok=True)
    exp_df.to_csv(exp_out, index=False)
    print(f"Logged {len(exp_df)} experimental runs to {exp_out}")

    print("\n" + "=" * 80)
    print("MODEL SELECTION & VALIDATION BENCHMARK SUMMARY")
    print("=" * 80)
    summary_cols = ["model_name", "mae", "wape_pct", "rmse", "bias", "total_business_cost", "skill_score_vs_b1"]
    print(exp_df[summary_cols].to_string(index=False))
    print("=" * 80)

    return exp_df, split_info


if __name__ == "__main__":
    run_training_pipeline()
