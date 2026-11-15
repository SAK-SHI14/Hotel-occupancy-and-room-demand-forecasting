"""Leakage Canary Demonstration.

Demonstrates the danger of look-ahead bias by injecting a deliberate leaky feature,
showing the suspiciously perfect offline score, and illustrating the immediate
collapse back to honest performance once the leakage guard purges it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from hotel_forecast.features.build import FEATURE_COLUMNS, build_feature_dataset


def run_leakage_canary():
    """Run the deliberate leakage experiment and plot comparative results."""
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    clean_path = Path("data/processed/booking_ledger_clean.csv")
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean ledger not found at {clean_path}")

    ledger_df = pd.read_csv(clean_path)

    # Build feature dataset with origin_step_days=2 for fast execution
    print("Extracting point-in-time feature matrix...")
    X_full, y = build_feature_dataset(ledger_df, origin_step_days=2)

    # Chronological train / validation split (80% train, 20% val)
    n = len(X_full)
    split_idx = int(n * 0.80)

    X_clean = X_full[FEATURE_COLUMNS].copy()

    # Create deliberate leaky feature: final stay date occupancy + tiny noise
    np.random.seed(42)
    leaky_feature = y.copy() + np.random.normal(0, 0.05, size=len(y))

    X_leaky = X_clean.copy()
    X_leaky["LEAKY_final_occupancy_stay_date"] = leaky_feature

    # Train Model A: Clean Leakage-Safe Model
    X_train_clean, X_val_clean = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model_clean = HistGradientBoostingRegressor(random_state=42)
    model_clean.fit(X_train_clean, y_train)
    preds_clean = model_clean.predict(X_val_clean)

    mae_clean = mean_absolute_error(y_val, preds_clean)
    r2_clean = r2_score(y_val, preds_clean)

    # Train Model B: Leaky Model (Artificial Cheat)
    X_train_leaky, X_val_leaky = X_leaky.iloc[:split_idx], X_leaky.iloc[split_idx:]
    model_leaky = HistGradientBoostingRegressor(random_state=42)
    model_leaky.fit(X_train_leaky, y_train)
    preds_leaky = model_leaky.predict(X_val_leaky)

    mae_leaky = mean_absolute_error(y_val, preds_leaky)
    r2_leaky = r2_score(y_val, preds_leaky)

    # Run Leakage Guard: Detect correlation with target > 0.95 or forbidden tokens
    guard_detected = []
    for col in X_leaky.columns:
        corr = np.corrcoef(X_leaky[col], y)[0, 1]
        if corr > 0.95 or "leaky" in col.lower() or "final" in col.lower():
            guard_detected.append((col, corr))

    print("=" * 70)
    print("LEAKAGE CANARY EXPERIMENTAL RESULTS")
    print("=" * 70)
    print(f"1. Leaky Model (With Look-Ahead Leak):      MAE = {mae_leaky:.4f} rooms | R² = {r2_leaky:.4f}")
    print(f"2. Leakage Guard Alert:                     Flagged {guard_detected}")
    print(f"3. Honest Guard-Protected Model:            MAE = {mae_clean:.4f} rooms | R² = {r2_clean:.4f}")
    print("=" * 70)

    # Generate Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    models = ["Leaky Model\n(Look-Ahead Leak)", "Honest Model\n(Guard Enforced)"]
    maes = [mae_leaky, mae_clean]
    r2s = [r2_leaky, r2_clean]
    colors = ["#d62728", "#2ca02c"]

    # MAE comparison
    bars1 = axes[0].bar(models, maes, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    axes[0].set_title("Validation MAE (Lower is Better)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Mean Absolute Error (Rooms)", fontsize=11)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.08,
            f"{yval:.2f} rooms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(maes) * 1.25)

    # R² comparison
    bars2 = axes[1].bar(models, r2s, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    axes[1].set_title("Validation R² Score (Suspicious Perfection)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("R² Coefficient of Determination", fontsize=11)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[1].set_ylim(0, 1.15)

    plt.suptitle("Leakage Canary: Detecting Look-Ahead Bias vs. Honest Offline Generalization", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_fig = figures_dir / "leakage_canary.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Leakage canary figure saved to {out_fig}")


if __name__ == "__main__":
    run_leakage_canary()
