"""Sound hospitality benchmark baselines.

Implements B1 Seasonal Naive and B2 Seasonal 4-Week Moving Average baselines
strictly adhering to the scikit-learn estimator interface.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class SeasonalNaiveBaseline(BaseEstimator, RegressorMixin):
    """B1 Seasonal Naive Benchmark.

    Predicts realized occupancy at the most recent same-weekday date
    observable at origin T (lag_same_weekday).
    """

    def __init__(self, lag_col: str = "lag_same_weekday", capacity_col: str = "capacity"):
        self.lag_col = lag_col
        self.capacity_col = capacity_col

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SeasonalNaiveBaseline":
        """Fit baseline (stateless heuristic; validates schema)."""
        if self.lag_col not in X.columns:
            raise ValueError(f"Required lag column '{self.lag_col}' not found in X.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict occupancy using same-weekday historical realized actuals."""
        preds = X[self.lag_col].to_numpy(dtype=float)
        # Clip to [0, capacity]
        if self.capacity_col in X.columns:
            caps = X[self.capacity_col].to_numpy(dtype=float)
            preds = np.clip(preds, 0.0, caps)
        else:
            preds = np.maximum(0.0, preds)
        return preds


class SeasonalMovingAverageBaseline(BaseEstimator, RegressorMixin):
    """B2 Seasonal 4-Week Moving Average Benchmark.

    Predicts the mean realized occupancy across the last 4 same-weekdays
    observable at origin T (same_weekday_4w_mean).
    """

    def __init__(self, mean_col: str = "same_weekday_4w_mean", capacity_col: str = "capacity"):
        self.mean_col = mean_col
        self.capacity_col = capacity_col

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SeasonalMovingAverageBaseline":
        """Fit baseline (stateless heuristic; validates schema)."""
        if self.mean_col not in X.columns:
            raise ValueError(f"Required moving average column '{self.mean_col}' not found in X.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict occupancy using 4-week same-weekday average."""
        preds = X[self.mean_col].to_numpy(dtype=float)
        if self.capacity_col in X.columns:
            caps = X[self.capacity_col].to_numpy(dtype=float)
            preds = np.clip(preds, 0.0, caps)
        else:
            preds = np.maximum(0.0, preds)
        return preds
