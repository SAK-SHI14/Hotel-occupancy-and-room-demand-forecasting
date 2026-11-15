"""Tests for chronological time-series splitting and purge gap verification."""

import json
from datetime import date, timedelta
from pathlib import Path

from hotel_forecast.models.train import calculate_chronological_splits


def test_chronological_splits_and_purge_gap():
    """Verify chronological ordering, proportions, and purge gaps."""
    # Generate 250 consecutive daily origins (sufficient timeline to fit splits + 2x 14d purge gaps)
    start_d = date(2024, 1, 1)
    origins = [start_d + timedelta(days=i) for i in range(250)]

    split_info = calculate_chronological_splits(
        unique_origins=origins,
        train_ratio=0.60,
        val_ratio=0.20,
        purge_gap_days=14,
    )

    t_start = date.fromisoformat(split_info["train_start_origin"])
    t_end = date.fromisoformat(split_info["train_end_origin"])
    v_start = date.fromisoformat(split_info["val_start_origin"])
    v_end = date.fromisoformat(split_info["val_end_origin"])
    te_start = date.fromisoformat(split_info["test_start_origin"])
    te_end = date.fromisoformat(split_info["test_end_origin"])

    # 1. Strictly chronological sequence
    assert t_start <= t_end < v_start <= v_end < te_start <= te_end

    # 2. Purge gaps must be at least 15 days from end of previous origin (14 gap days + 1)
    assert (v_start - t_end).days >= 15, f"Train-Val purge gap violation: {(v_start - t_end).days} days"
    assert (te_start - v_end).days >= 15, f"Val-Test purge gap violation: {(te_start - v_end).days} days"

    # 3. Horizon safety: maximum horizon is 14 days.
    # The latest stay date forecasted from t_end is t_end + 14.
    # This must strictly precede v_start!
    latest_train_stay_date = t_end + timedelta(days=14)
    assert latest_train_stay_date < v_start, (
        f"Leakage hazard: train stay date {latest_train_stay_date} overlaps validation start {v_start}!"
    )

    latest_val_stay_date = v_end + timedelta(days=14)
    assert latest_val_stay_date < te_start, (
        f"Leakage hazard: val stay date {latest_val_stay_date} overlaps test start {te_start}!"
    )


def test_persisted_split_info_file():
    """Verify persisted split_info.json if it exists."""
    root = Path(__file__).resolve().parent.parent
    split_file = root / "reports" / "split_info.json"
    if split_file.is_file():
        with open(split_file, encoding="utf-8") as f:
            data = json.load(f)
        assert "train_start_origin" in data
        assert "val_start_origin" in data
        assert "test_start_origin" in data
        assert data["purge_gap_1_days"] == 14
        assert data["purge_gap_2_days"] == 14
