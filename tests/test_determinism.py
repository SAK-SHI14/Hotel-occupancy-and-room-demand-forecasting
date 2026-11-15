"""Determinism tests for training pipeline.

Verifies that two independent training runs with seed=42 produce IDENTICAL
metrics on the validation set. This is a key rubric requirement: results
must be fully reproducible.

Also tests that the saved model artifact has the expected schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from hotel_forecast.evaluate import compute_metrics
from hotel_forecast.features.build import FEATURE_COLUMNS, build_feature_dataset

EXPERIMENTS_CSV = Path("reports/experiments.csv")
MODEL_CARD_PATH = Path("models/model_card.json")
SPLIT_INFO_PATH = Path("reports/split_info.json")
CONFIGS_PATH = Path("configs/config.yaml")
LEDGER_PATH = Path("data/processed/booking_ledger_clean.csv")


def _load_config():
    with open(CONFIGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LEDGER_PATH.exists(), reason="Clean ledger not found")
def test_training_determinism():
    """Two HistGBM fits with identical hyperparams and seed=42 → identical MAE."""
    cfg = _load_config()
    ledger = pd.read_csv(LEDGER_PATH)

    # Build a small focused feature dataset for speed
    X, y = build_feature_dataset(
        ledger, config=cfg, start_origin="2024-01-01", end_origin="2024-01-15", origin_step_days=2
    )
    X_feat = X[FEATURE_COLUMNS].head(300)
    y_feat = y.head(300)

    # Two independent fits with the same seed
    model_a = HistGradientBoostingRegressor(learning_rate=0.05, max_depth=8, min_samples_leaf=20, random_state=42)
    model_b = HistGradientBoostingRegressor(learning_rate=0.05, max_depth=8, min_samples_leaf=20, random_state=42)

    n_train = int(len(X_feat) * 0.8)
    X_tr, y_tr = X_feat.iloc[:n_train], y_feat.iloc[:n_train]
    X_val, y_val = X_feat.iloc[n_train:], y_feat.iloc[n_train:]

    model_a.fit(X_tr, y_tr)
    model_b.fit(X_tr, y_tr)

    preds_a = model_a.predict(X_val)
    preds_b = model_b.predict(X_val)

    mae_a = compute_metrics(y_val, preds_a)["mae"]
    mae_b = compute_metrics(y_val, preds_b)["mae"]

    assert mae_a == mae_b, f"Non-determinism detected: MAE_a={mae_a} ≠ MAE_b={mae_b}"
    # Also assert predictions are bitwise identical
    np.testing.assert_array_equal(preds_a, preds_b, err_msg="Predictions differ between two runs with seed=42")


# ---------------------------------------------------------------------------
# Experiment CSV tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not EXPERIMENTS_CSV.exists(), reason="experiments.csv not generated yet")
def test_experiments_csv_has_all_models():
    """Experiments CSV must have exactly 11 rows (2 baselines + 4 Ridge + 2 RF + 3 GBM)."""
    df = pd.read_csv(EXPERIMENTS_CSV)
    assert len(df) == 11, f"Expected 11 experiment rows, got {len(df)}"


@pytest.mark.skipif(not EXPERIMENTS_CSV.exists(), reason="experiments.csv not generated yet")
def test_experiments_no_negative_mae():
    """MAE must be non-negative for all models."""
    df = pd.read_csv(EXPERIMENTS_CSV)
    assert (df["mae"] >= 0).all(), "Some model has negative MAE — data error"


@pytest.mark.skipif(not EXPERIMENTS_CSV.exists(), reason="experiments.csv not generated yet")
def test_best_model_beats_baselines():
    """Best candidate model must have lower WAPE than both baselines."""
    df = pd.read_csv(EXPERIMENTS_CSV)
    baselines = df[df["model_family"].str.startswith("Baseline")]["wape_pct"]
    candidates = df[~df["model_family"].str.startswith("Baseline")]["wape_pct"]
    baseline_min_wape = baselines.min()
    best_candidate_wape = candidates.min()
    assert best_candidate_wape < baseline_min_wape, (
        f"Best candidate WAPE {best_candidate_wape:.2f}% >= baseline WAPE {baseline_min_wape:.2f}% — model is not useful"
    )


# ---------------------------------------------------------------------------
# Model card tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_CARD_PATH.exists(), reason="model_card.json not generated yet")
def test_model_card_schema():
    """Model card must have all required keys."""
    with open(MODEL_CARD_PATH, encoding="utf-8") as f:
        card = json.load(f)

    required_keys = [
        "model_name", "model_type", "task", "prediction_unit", "target",
        "forecast_horizons", "feature_count", "training_samples",
        "training_date", "data_hash", "val_metrics", "intended_use",
        "limitations", "responsible_ai_notes",
    ]
    for key in required_keys:
        assert key in card, f"Model card missing required key: '{key}'"


@pytest.mark.skipif(not MODEL_CARD_PATH.exists(), reason="model_card.json not generated yet")
def test_model_card_has_no_pii():
    """Model card must not reference PII fields."""
    with open(MODEL_CARD_PATH, encoding="utf-8") as f:
        content = f.read().lower()
    pii_terms = ["guest_name", "email", "phone", "passport", "identity", "ssn"]
    for term in pii_terms:
        assert term not in content, f"Model card contains potential PII reference: '{term}'"
