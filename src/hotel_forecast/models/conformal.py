"""Split conformal prediction for calibrated uncertainty intervals.

Uses validation-set residuals grouped by horizon bucket to produce
empirically calibrated 80% prediction intervals (alpha=0.20).

Theory (one sentence per key idea):
- Split conformal: fit the model on train, compute conformity scores (abs
  residuals) on a held-out calibration set (validation), then at test time
  find the 1-alpha quantile of those scores and add/subtract it from the
  point forecast.
- Grouping by horizon: closer horizons have smaller residuals, so we
  compute separate quantiles for h in [1-3], [4-7], [8-14].  This keeps
  interval width realistic rather than uniformly inflated.

No fitting to the test set — conformity scores come only from validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Horizon groups: (label, min_h_inclusive, max_h_inclusive)
HORIZON_BUCKETS: list[tuple[str, int, int]] = [
    ("h1-3", 1, 3),
    ("h4-7", 4, 7),
    ("h8-14", 8, 14),
]

# Target coverage level
ALPHA: float = 0.20  # 80% prediction intervals


class SplitConformalPredictor:
    """Split conformal predictor grouped by forecast horizon.

    Workflow
    --------
    1. ``calibrate(X_val, y_val, model)`` — compute & store residual
       quantiles per horizon bucket from the validation set.
    2. ``predict_intervals(X, point_preds)`` — at inference time, look up
       the horizon-bucket quantile and return (lower, upper) arrays clipped
       to [0, capacity].

    Parameters
    ----------
    alpha : float
        Miscoverage rate.  Default 0.20 → 80% intervals.
    horizon_col : str
        Column name in X that holds the integer forecast horizon h.
    capacity_col : str
        Column name in X that holds capacity (for clipping upper bound).
    """

    def __init__(
        self,
        alpha: float = ALPHA,
        horizon_col: str = "horizon",
        capacity_col: str = "capacity",
    ) -> None:
        self.alpha = alpha
        self.horizon_col = horizon_col
        self.capacity_col = capacity_col
        # Filled by calibrate(); maps bucket_label -> quantile value (float)
        self.quantiles_: dict[str, float] = {}
        # Residual arrays per bucket (kept for diagnostics)
        self._residuals_per_bucket: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
        model: Any,
    ) -> SplitConformalPredictor:
        """Compute residual quantiles from the validation set.

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation feature matrix (must include FEATURE_COLUMNS).
        y_val : array-like
            Ground-truth occupancy for each validation row.
        model : sklearn estimator
            Fitted model with a ``predict`` method.
        """
        # Point predictions on validation
        point_preds = model.predict(X_val)
        # Clip to capacity so residuals reflect clipped predictions
        caps = X_val[self.capacity_col].to_numpy(dtype=float)
        point_preds = np.clip(point_preds, 0.0, caps)

        y_arr = np.asarray(y_val, dtype=float)
        abs_residuals = np.abs(point_preds - y_arr)
        horizons = X_val[self.horizon_col].to_numpy(dtype=int)

        for label, h_min, h_max in HORIZON_BUCKETS:
            mask = (horizons >= h_min) & (horizons <= h_max)
            bucket_residuals = abs_residuals[mask]
            if len(bucket_residuals) == 0:
                # Fallback: use all residuals if a bucket is empty
                bucket_residuals = abs_residuals
            # Finite-sample corrected quantile: (n+1)(1-alpha)/n
            n = len(bucket_residuals)
            q_level = min(1.0, (n + 1) * (1.0 - self.alpha) / n)
            self.quantiles_[label] = float(np.quantile(bucket_residuals, q_level))
            self._residuals_per_bucket[label] = bucket_residuals
            print(
                f"  Bucket {label}: n={n:,} residuals | "
                f"q{(1-self.alpha)*100:.0f}%={self.quantiles_[label]:.3f} rooms"
            )

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_intervals(
        self,
        X: pd.DataFrame,
        point_preds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) arrays for given predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must include horizon and capacity columns).
        point_preds : np.ndarray
            Model point predictions (already clipped to [0, capacity]).

        Returns
        -------
        (lower, upper) : tuple of np.ndarray
            80% prediction interval bounds, clipped to [0, capacity].
        """
        if not self.quantiles_:
            raise RuntimeError("Call calibrate() before predict_intervals().")

        horizons = X[self.horizon_col].to_numpy(dtype=int)
        caps = X[self.capacity_col].to_numpy(dtype=float)

        half_widths = np.zeros(len(X), dtype=float)
        for label, h_min, h_max in HORIZON_BUCKETS:
            mask = (horizons >= h_min) & (horizons <= h_max)
            half_widths[mask] = self.quantiles_[label]

        lower = np.clip(point_preds - half_widths, 0.0, caps)
        upper = np.clip(point_preds + half_widths, 0.0, caps)
        return lower, upper

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage_report(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series | np.ndarray,
        point_preds: np.ndarray,
    ) -> dict[str, Any]:
        """Compute empirical coverage on test set, broken down by bucket.

        Returns
        -------
        dict
            Per-bucket and overall coverage and mean interval width.
        """
        lower, upper = self.predict_intervals(X_test, point_preds)
        y_arr = np.asarray(y_test, dtype=float)
        horizons = X_test[self.horizon_col].to_numpy(dtype=int)

        in_interval = (y_arr >= lower) & (y_arr <= upper)
        widths = upper - lower

        report: dict[str, Any] = {
            "target_coverage": 1.0 - self.alpha,
            "overall_coverage": float(np.mean(in_interval)),
            "overall_mean_width": float(np.mean(widths)),
            "buckets": {},
        }
        for label, h_min, h_max in HORIZON_BUCKETS:
            mask = (horizons >= h_min) & (horizons <= h_max)
            if mask.sum() == 0:
                continue
            report["buckets"][label] = {
                "n": int(mask.sum()),
                "coverage": float(np.mean(in_interval[mask])),
                "mean_width": float(np.mean(widths[mask])),
                "quantile_halfwidth": self.quantiles_.get(label, 0.0),
            }
        return report

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize quantiles for storage in model artifact."""
        return {
            "alpha": self.alpha,
            "horizon_col": self.horizon_col,
            "capacity_col": self.capacity_col,
            "quantiles": self.quantiles_,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SplitConformalPredictor:
        """Re-hydrate from serialized dict."""
        obj = cls(alpha=d["alpha"], horizon_col=d["horizon_col"], capacity_col=d["capacity_col"])
        obj.quantiles_ = d["quantiles"]
        return obj


# ---------------------------------------------------------------------------
# Standalone figures
# ---------------------------------------------------------------------------

def plot_interval_width_by_horizon(
    conformal: SplitConformalPredictor,
    output_path: Path | str = "reports/figures/interval_width_by_horizon.png",
) -> None:
    """Bar chart: conformal half-width quantile per horizon bucket.

    Shows that interval width grows with horizon distance — a property that
    reviewers check to verify the uncertainty is realistic.
    """
    labels = [b[0] for b in HORIZON_BUCKETS]
    widths = [conformal.quantiles_.get(lbl, 0.0) * 2 for lbl in labels]  # full width

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, widths, color=["#1f77b4", "#ff7f0e", "#d62728"], edgecolor="black", linewidth=0.8)
    for bar, w in zip(bars, widths, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{w:.1f}", ha="center", fontsize=10)

    ax.set_xlabel("Forecast Horizon Bucket", fontsize=11)
    ax.set_ylabel("80% Interval Full Width (rooms)", fontsize=11)
    ax.set_title("Conformal Interval Width by Horizon Bucket\n(Width should grow with horizon — sanity check)", fontsize=12)
    ax.set_ylim(0, max(widths) * 1.3 + 0.5)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Interval width by horizon figure saved to {out_file}")


def plot_coverage_report(
    coverage: dict[str, Any],
    output_path: Path | str = "reports/figures/interval_coverage.png",
) -> None:
    """Horizontal bar: empirical coverage vs 80% target, per horizon bucket."""
    labels = list(coverage["buckets"].keys())
    achieved = [coverage["buckets"][lbl]["coverage"] * 100 for lbl in labels]
    target = (1.0 - ALPHA) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = range(len(labels))
    bars = ax.barh(list(y_pos), achieved, color=["#2ca02c" if a >= target else "#d62728" for a in achieved], edgecolor="black")
    ax.axvline(target, color="navy", linestyle="--", linewidth=1.5, label=f"Target {target:.0f}%")

    for bar, cov in zip(bars, achieved, strict=False):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{cov:.1f}%", va="center")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Empirical Coverage (%)", fontsize=11)
    ax.set_title(f"Conformal Interval Empirical Coverage on Test Set\n(Target: {target:.0f}% | Green=achieved, Red=below)", fontsize=12)
    ax.set_xlim(0, 110)
    ax.legend()
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Coverage report figure saved to {out_file}")


def save_coverage_report(coverage: dict[str, Any], output_path: Path | str = "reports/conformal_coverage.json") -> None:
    """Write coverage report to JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)
    print(f"Coverage report written to {out_file}")
