"""Evaluation metrics and test split evaluation runner.

Computes WAPE, MAE, RMSE, Bias, Skill Score vs B1, and Asymmetric Business Cost.
Evaluates held-out test split once, writing reports/test_run_log.json and reports/metrics.json.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from hotel_forecast.features.build import FEATURE_COLUMNS, build_feature_dataset
from hotel_forecast.models.baselines import SeasonalMovingAverageBaseline, SeasonalNaiveBaseline


def load_config(config_path: Path | str = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    c_under: float = 3.0,
    c_over: float = 1.0,
    b1_mae: float | None = None,
) -> dict[str, float]:
    """Compute comprehensive performance and business metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth realized occupied room nights.
    y_pred : array-like
        Predicted occupied room nights.
    c_under : float
        Cost multiplier for under-forecasting (default 3.0).
    c_over : float
        Cost multiplier for over-forecasting (default 1.0).
    b1_mae : float, optional
        Mean absolute error of B1 Seasonal Naive baseline to compute skill score.

    Returns
    -------
    dict[str, float]
        Dictionary of WAPE, MAE, RMSE, Bias, Business Cost, and Skill Score.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    diff = yp - yt  # positive = over-forecast, negative = under-forecast
    abs_errors = np.abs(diff)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    sum_true = float(np.sum(yt))
    wape = float(np.sum(abs_errors) / sum_true) if sum_true > 0 else 0.0
    bias = float(np.mean(diff))

    # Asymmetric business cost: c_under * max(actual - pred, 0) + c_over * max(pred - actual, 0)
    under_err = np.maximum(-diff, 0.0)
    over_err = np.maximum(diff, 0.0)
    business_cost = float(np.sum(c_under * under_err + c_over * over_err))
    mean_cost_per_room = float(np.mean(c_under * under_err + c_over * over_err))

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
        "wape_pct": round(wape * 100.0, 2),
        "bias": round(bias, 4),
        "total_business_cost": round(business_cost, 2),
        "mean_business_cost": round(mean_cost_per_room, 4),
    }

    if b1_mae is not None and b1_mae > 0:
        skill = 1.0 - (mae / b1_mae)
        metrics["skill_score_vs_b1"] = round(skill, 4)

    return metrics


def run_test_evaluation(
    model_path: Path | str = "models/model.joblib",
    config_path: Path | str = "configs/config.yaml",
    split_info_path: Path | str = "reports/split_info.json",
    metrics_path: Path | str = "reports/metrics.json",
    log_path: Path | str = "reports/test_run_log.json",
) -> dict[str, Any]:
    """Execute evaluation on held-out test split, write audit log and metrics."""
    cfg = load_config(config_path)
    c_under = float(cfg["metrics"]["asymmetric_cost"]["c_under"])
    c_over = float(cfg["metrics"]["asymmetric_cost"]["c_over"])

    # Load serialized model pipeline
    m_file = Path(model_path)
    if not m_file.exists():
        raise FileNotFoundError(f"Trained model not found at {m_file}. Run train.py first.")
    artifact = joblib.load(m_file)
    model = artifact["model"]

    # Load split info
    s_file = Path(split_info_path)
    if not s_file.exists():
        raise FileNotFoundError(f"Split info not found at {s_file}. Run train.py first.")
    with open(s_file, encoding="utf-8") as f:
        split_info = json.load(f)

    test_start = split_info["test_start_origin"]
    test_end = split_info["test_end_origin"]

    # Build test features
    clean_path = Path("data/processed/booking_ledger_clean.csv")
    ledger_df = pd.read_csv(clean_path)
    print(f"Extracting test split features from origin {test_start} to {test_end}...")
    X_test_all, y_test = build_feature_dataset(
        ledger_df,
        config=cfg,
        start_origin=test_start,
        end_origin=test_end,
        origin_step_days=1,
    )
    X_test = X_test_all[FEATURE_COLUMNS]

    # Evaluate B1 Baseline on test
    b1 = SeasonalNaiveBaseline().fit(X_test)
    b1_preds = b1.predict(X_test)
    b1_metrics = compute_metrics(y_test, b1_preds, c_under, c_over)

    # Evaluate B2 Baseline on test
    b2 = SeasonalMovingAverageBaseline().fit(X_test)
    b2_preds = b2.predict(X_test)
    b2_metrics = compute_metrics(y_test, b2_preds, c_under, c_over, b1_mae=b1_metrics["mae"])

    # Evaluate Final Candidate Model on test
    preds = model.predict(X_test)
    # Clip predictions to [0, capacity]
    caps = X_test["capacity"].to_numpy(dtype=float)
    preds = np.clip(preds, 0.0, caps)

    model_metrics = compute_metrics(y_test, preds, c_under, c_over, b1_mae=b1_metrics["mae"])

    timestamp = datetime.now().isoformat()

    # Log test run audit trail
    test_log_entry = {
        "timestamp": timestamp,
        "test_origin_start": test_start,
        "test_origin_end": test_end,
        "test_sample_count": len(X_test),
        "model_type": artifact.get("model_type", "HistGradientBoostingRegressor"),
        "model_metrics": model_metrics,
        "b1_naive_metrics": b1_metrics,
        "b2_moving_avg_metrics": b2_metrics,
    }

    out_log = Path(log_path)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    existing_logs = []
    if out_log.exists():
        try:
            with open(out_log, encoding="utf-8") as f:
                existing_logs = json.load(f)
        except Exception:
            existing_logs = []
    existing_logs.append(test_log_entry)
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(existing_logs, f, indent=2)

    # Load or initialize metrics.json
    out_metrics = Path(metrics_path)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_summary = {}
    if out_metrics.exists():
        try:
            with open(out_metrics, encoding="utf-8") as f:
                metrics_summary = json.load(f)
        except Exception:
            metrics_summary = {}

    metrics_summary["test_evaluation"] = test_log_entry
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("=" * 70)
    print("HELD-OUT TEST SET EVALUATION AUDIT")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Test samples: {len(X_test):,} | Origins: {test_start} to {test_end}")
    print(f"B1 Seasonal Naive:      MAE = {b1_metrics['mae']:.2f} | WAPE = {b1_metrics['wape_pct']:.1f}% | Cost = {b1_metrics['total_business_cost']:,.0f}")
    print(f"B2 Seasonal Moving Avg: MAE = {b2_metrics['mae']:.2f} | WAPE = {b2_metrics['wape_pct']:.1f}% | Cost = {b2_metrics['total_business_cost']:,.0f}")
    print(f"Candidate Model:        MAE = {model_metrics['mae']:.2f} | WAPE = {model_metrics['wape_pct']:.1f}% | Cost = {model_metrics['total_business_cost']:,.0f}")
    print(f"Skill Score vs B1:      +{model_metrics['skill_score_vs_b1']*100:.1f}% error reduction")
    print("=" * 70)

    return metrics_summary


if __name__ == "__main__":
    if "--split" in sys.argv and "test" in sys.argv:
        run_test_evaluation()
    else:
        print("Usage: python -m hotel_forecast.evaluate --split test")
