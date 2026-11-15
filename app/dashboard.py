"""Streamlit interactive operations dashboard for Hotel Room Demand Forecasting.

Decision-support tool for hotel revenue and operations managers:
- Point-in-time forecast generator with 80% conformal uncertainty intervals
- Real-time overlay of actuals when viewing historical origins
- Multi-category occupancy heatmap across 14-day horizon
- Live model comparison and error analysis benchmarks
- Human-in-the-loop review alerts for high-uncertainty or capacity-constrained dates
- Transparent responsible AI panel with model limitations
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from hotel_forecast.features.build import compute_daily_realized_occupancy
from hotel_forecast.predict import load_artifact, predict_horizon

# Page Configuration
st.set_page_config(
    page_title="Hotel Demand Forecasting & Decision Support",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling - Warm Beige and Sky Blue Palette
st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
        background-color: #FDFBF7;
    }

    /* Header typography */
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #0369A1; /* Deep Sky Blue */
        margin-bottom: 0.2rem;
        letter-spacing: -0.01em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #57534E; /* Warm Stone / Taupe */
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E0F2FE; /* Sky Blue accent border */
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #F5EFEB; /* Warm sand beige */
        border: 1px solid #BAE6FD; /* Sky Blue border */
        border-left: 5px solid #0284C7; /* Sky blue accent */
        padding: 0.8rem 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(2, 132, 199, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #57534E !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0369A1 !important;
    }

    /* Custom cards */
    .metric-card {
        background-color: #F5EFEB;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0284C7;
        box-shadow: 0 1px 3px rgba(2, 132, 199, 0.08);
    }
    .alert-card {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #D97706;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F5EFEB;
        border-right: 1px solid #E2D9CF;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #0369A1;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F5EFEB;
        padding: 6px 8px;
        border-radius: 10px;
        border: 1px solid #E2D9CF;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #57534E;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_and_ledger():
    """Load and cache model artifact and processed ledger."""
    model_path = Path("models/model.joblib")
    ledger_path = Path("data/processed/booking_ledger_clean.csv")

    if not model_path.exists():
        st.error(f"Model artifact not found at {model_path}. Please train the model first.")
        st.stop()
    if not ledger_path.exists():
        st.error(f"Clean ledger not found at {ledger_path}.")
        st.stop()

    artifact = load_artifact(model_path)
    ledger_df = pd.read_csv(ledger_path)
    # Precompute internal date series once to avoid repeated conversions
    ledger_df["_b_date"] = pd.to_datetime(ledger_df["booking_date"]).dt.date
    ledger_df["_a_date"] = pd.to_datetime(ledger_df["arrival_date"]).dt.date
    ledger_df["_d_date"] = pd.to_datetime(ledger_df["departure_date"]).dt.date
    ledger_df["_c_date"] = pd.to_datetime(ledger_df["cancellation_date"]).dt.date
    realized_occ = compute_daily_realized_occupancy(ledger_df)
    ledger_df.attrs["realized_occ_df"] = realized_occ
    return artifact, ledger_df, realized_occ


def load_metrics_and_card():
    """Load metrics summary and model card if available."""
    metrics_path = Path("reports/metrics.json")
    card_path = Path("models/model_card.json")

    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics_data = json.load(f)

    card_data = {}
    if card_path.exists():
        with open(card_path, encoding="utf-8") as f:
            card_data = json.load(f)

    return metrics_data, card_data


# Load Resources
artifact, ledger_df, realized_occ = get_model_and_ledger()
metrics_data, card_data = load_metrics_and_card()

# Sidebar: Controls
st.sidebar.image("https://img.icons8.com/color/96/hotel-star.png", width=64)
st.sidebar.title("Forecast Controls")

all_categories = ["Standard", "Deluxe", "Family", "Suite"]
selected_categories = st.sidebar.multiselect(
    "Room Categories",
    options=all_categories,
    default=all_categories,
    help="Select categories to forecast.",
)

# Date Range Limits based on ledger
min_ledger_date = pd.to_datetime(ledger_df["booking_date"]).min().date() + timedelta(days=60)
max_ledger_date = pd.to_datetime(ledger_df["arrival_date"]).max().date() - timedelta(days=15)

default_origin = date(2025, 9, 15)  # Historical date in test window for rich visualization
if default_origin < min_ledger_date or default_origin > max_ledger_date:
    default_origin = min_ledger_date + timedelta(days=180)

origin_date = st.sidebar.date_input(
    "Forecast Origin Date (T)",
    value=default_origin,
    min_value=min_ledger_date,
    max_value=max_ledger_date,
    help="Cutoff time: Only bookings made on or before this date are known to the model.",
)

horizon_days = st.sidebar.slider(
    "Forecast Horizon (days ahead)",
    min_value=1,
    max_value=14,
    value=14,
    help="Operational planning window (1 to 14 days ahead).",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Operational Settings")
st.sidebar.info(
    "**Asymmetric Costs**\n"
    "- Under-forecasting ($3.0x): Stockouts / walk costs / overtime\n"
    "- Over-forecasting ($1.0x): Empty room holding cost\n"
    "**Uncertainty**\n"
    "- 80% Split Conformal Intervals calibrated on validation residuals."
)

# Main Header
st.markdown('<div class="main-header">🏨 Hotel Occupancy & Room-Demand Forecasting</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Decision Support System | Forecast Origin: <b>{origin_date}</b> | Horizon: <b>1-{horizon_days} Days</b></div>',
    unsafe_allow_html=True,
)

# Tab Navigation
tabs = st.tabs(["📈 Operational Forecast", "🔥 Occupancy Heatmap", "📊 Model Performance", "🛡️ Responsible AI & Model Card"])

@st.cache_data(show_spinner=False)
def get_cached_forecast(origin_date_str: str, horizon: int, categories: tuple[str, ...]):
    """Cached forecast generation for instant UI responsiveness."""
    return predict_horizon(
        origin_date=origin_date_str,
        horizon_days=horizon,
        room_categories=list(categories) if categories else all_categories,
        ledger_df=ledger_df,
        artifact=artifact,
    )


# Compute Forecasts (sub-second execution + instant cache hits)
cats_tuple = tuple(selected_categories) if selected_categories else tuple(all_categories)
forecast_results = get_cached_forecast(str(origin_date), horizon_days, cats_tuple)
fc_df = pd.DataFrame(forecast_results)

# TAB 1: OPERATIONAL FORECAST
with tabs[0]:
    if fc_df.empty or "error" in fc_df.columns and fc_df["prediction"].isna().all():
        st.warning("No forecasts generated. Please select valid categories.")
    else:
        # High-level KPI summary cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        total_pred_rooms = fc_df["prediction"].sum()
        total_capacity = fc_df["capacity"].sum()
        avg_occ_rate = (total_pred_rooms / total_capacity * 100) if total_capacity > 0 else 0
        human_review_count = fc_df["needs_human_review"].sum()

        with kpi_col1:
            st.metric("Total Forecasted Room Nights", f"{total_pred_rooms:,.1f}")
        with kpi_col2:
            st.metric("Expected Occupancy Rate", f"{avg_occ_rate:.1f}%")
        with kpi_col3:
            st.metric("Model Architecture", artifact.get("model_type", "HistGBM"))
        with kpi_col4:
            if human_review_count > 0:
                st.metric("Human Review Alerts", f"{human_review_count}", delta="-Action Req", delta_color="inverse")
            else:
                st.metric("Review Alerts", "0", delta="Normal")

        if human_review_count > 0:
            st.warning(
                f"⚠️ **Attention Manager:** {human_review_count} forecast windows flagged for human review "
                "due to wide conformal prediction intervals or extreme occupancy (>95% or <5%). Inspect flagged rows below."
            )

        # Plot Forecast Curves with Confidence Intervals
        st.subheader("14-Day Demand Trajectory with 80% Conformal Prediction Intervals")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#FDFBF7")
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")

        category_colors = {
            "Standard": "#0284C7",  # Sky Blue
            "Deluxe": "#0891B2",    # Deep Ocean Cyan
            "Family": "#D97706",    # Warm Sand / Amber
            "Suite": "#6366F1",     # Indigo Slate
        }

        for cat in selected_categories:
            cat_data = fc_df[fc_df["room_category"] == cat].copy()
            if cat_data.empty:
                continue
            cat_data["stay_date_dt"] = pd.to_datetime(cat_data["stay_date"])
            cat_data = cat_data.sort_values("stay_date_dt")

            color = category_colors.get(cat, "#374151")
            ax.plot(
                cat_data["stay_date_dt"],
                cat_data["prediction"],
                marker="o",
                label=f"{cat} (Forecast)",
                color=color,
                linewidth=2.2,
            )
            ax.fill_between(
                cat_data["stay_date_dt"],
                cat_data["lower"],
                cat_data["upper"],
                color=color,
                alpha=0.18,
                label=f"{cat} 80% Interval",
            )

            # Overlay Ground Truth Actuals if stay_date exists in historical realized occupancy
            actual_vals = []
            actual_dates = []
            for _, row in cat_data.iterrows():
                s_date = pd.to_datetime(row["stay_date"]).date()
                if s_date in realized_occ.index and cat in realized_occ.columns:
                    actual_dates.append(pd.to_datetime(s_date))
                    actual_vals.append(realized_occ.loc[s_date, cat])

            if actual_vals:
                ax.plot(
                    actual_dates,
                    actual_vals,
                    linestyle="--",
                    marker="s",
                    color=color,
                    alpha=0.85,
                    label=f"{cat} (Realized Actual)",
                )

        ax.set_xlabel("Stay Date", fontsize=11, fontweight="bold", color="#1E293B")
        ax.set_ylabel("Occupied Room Nights", fontsize=11, fontweight="bold", color="#1E293B")
        ax.tick_params(colors="#475569")
        ax.grid(True, linestyle=":", alpha=0.6, color="#CBD5E1")
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            facecolor="#FDFBF7",
            edgecolor="#BAE6FD",
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Detailed Forecast Data Table
        st.subheader("Operational Planning Table")
        display_cols = [
            "stay_date",
            "room_category",
            "horizon",
            "prediction",
            "lower",
            "upper",
            "occupancy_rate",
            "capacity",
            "needs_human_review",
        ]
        styled_df = fc_df[display_cols].copy()
        styled_df["occupancy_pct"] = (styled_df["occupancy_rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(styled_df, use_container_width=True)

# TAB 2: OCCUPANCY HEATMAP
with tabs[1]:
    st.subheader("Expected Occupancy Rate (%) Heatmap across Horizon")
    if not fc_df.empty:
        pivot_occ = fc_df.pivot(index="room_category", columns="stay_date", values="occupancy_rate") * 100
        fig_heat, ax_heat = plt.subplots(figsize=(12, 4))
        fig_heat.patch.set_facecolor("#FDFBF7")
        ax_heat.set_facecolor("#FDFBF7")

        # Custom Beige-to-Sky-Blue gradient colormap
        beige_skyblue_cmap = LinearSegmentedColormap.from_list(
            "beige_skyblue",
            ["#F5EFEB", "#E0F2FE", "#7DD3FC", "#38BDF8", "#0284C7", "#0369A1"],
        )
        sns.heatmap(
            pivot_occ,
            annot=True,
            fmt=".1f",
            cmap=beige_skyblue_cmap,
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Occupancy Rate (%)"},
            ax=ax_heat,
            linewidths=1.0,
            linecolor="#FDFBF7",
        )
        ax_heat.set_xlabel("Stay Date", fontsize=11, fontweight="bold", color="#1E293B")
        ax_heat.set_ylabel("Room Category", fontsize=11, fontweight="bold", color="#1E293B")
        ax_heat.tick_params(colors="#475569")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

        st.caption(
            "💡 **Interpretation:** Cells in dark sky blue (>80%) indicate high occupancy pressure where "
            "overbooking thresholds should be strictly enforced and discount rates closed."
        )

# TAB 3: MODEL PERFORMANCE
with tabs[2]:
    st.subheader("Model Benchmark & Validation Performance")
    exp_csv_path = Path("reports/experiments.csv")
    if exp_csv_path.exists():
        exp_df = pd.read_csv(exp_csv_path)
        st.dataframe(exp_df, use_container_width=True)

    col_bench1, col_bench2 = st.columns(2)
    fig_horizon_path = Path("reports/figures/error_by_horizon.png")
    if fig_horizon_path.exists():
        with col_bench1:
            st.image(str(fig_horizon_path), caption="Error Degradation across 1-14 Day Horizons")

    fig_cat_path = Path("reports/figures/error_by_category.png")
    if fig_cat_path.exists():
        with col_bench2:
            st.image(str(fig_cat_path), caption="Validation Error by Room Category")

    col_bench3, col_bench4 = st.columns(2)
    fig_split_path = Path("reports/figures/split_timeline.png")
    if fig_split_path.exists():
        with col_bench3:
            st.image(str(fig_split_path), caption="Chronological Split with 14-Day Purge Gaps")

    fig_conf_path = Path("reports/figures/interval_width_by_horizon.png")
    if fig_conf_path.exists():
        with col_bench4:
            st.image(str(fig_conf_path), caption="Conformal Interval Width Growth by Horizon Bucket")

# TAB 4: RESPONSIBLE AI & MODEL CARD
with tabs[3]:
    st.subheader("Model Governance & Responsible AI Card")
    st.markdown(
        """
        ### System Purpose & Intended Use
        This system is a **human-in-the-loop decision-support tool** for hotel managers to optimize staffing,
        inventory allocations, and pricing over a 14-day operational horizon.
        It is **NOT** an automated autonomous pricing bot.

        ### Key Assumptions & Safeguards
        1. **Strict Point-in-Time Features:** Features are constructed exclusively using bookings and cancellations known at origin $T$.
        2. **Conformal Uncertainty:** 80% coverage guarantees via split conformal prediction on validation residuals.
        3. **Asymmetric Risk Optimization:** Under-forecasting ($3.0x cost) is penalized more heavily than over-forecasting ($1.0x cost).
        4. **Zero PII:** Dataset is strictly non-PII and aggregated by (stay_date, room_category).
        """
    )
    if card_data:
        st.json(card_data)
