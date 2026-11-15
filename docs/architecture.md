# System Architecture: Hotel Occupancy & Room-Demand Forecasting

## 1. End-to-End System Flow

```mermaid
flowchart TD
    subgraph DataEngine ["Data Engine (Synthetic Booking Ledger)"]
        A1["Booking Generator<br/>(Seasonality, Trends, Events, Holidays)"] --> A2["Raw Ledger CSV<br/>(40,891 rows)"]
        A2 --> A3["Cleaning & Quality Gate<br/>(Drops duplicates, corrupted dates)"]
        A3 --> A4["Clean Ledger CSV<br/>(40,571 rows)"]
    end

    subgraph FeaturePipeline ["Leakage-Safe Feature Engineering"]
        A4 --> B1["build_features(origin_T, stay_date, category)"]
        B1 --> B2["OTB & Pace Features<br/>(as-of origin T)"]
        B1 --> B3["Horizon-Aware Lags<br/>(stay_date - 7*ceil(h/7))"]
        B1 --> B4["Calendar & Holiday Features<br/>(Indian Calendar + Events)"]
        B1 --> B5["Booking Dynamics<br/>(OTA share, Avg lead time)"]
        B2 & B3 & B4 & B5 --> B6["Point-in-Time Feature Matrix (23 features)"]
    end

    subgraph SplitStrategy ["Chronological Splitting & Validation"]
        B6 --> C1["Timeline Partitioning"]
        C1 --> C2["Train Split (60% / 18,480 samples)<br/>2023-02-15 to 2024-12-04"]
        C1 --> C3["Purge Gap 1 (14 Days)"]
        C1 --> C4["Validation Split (20% / 6,160 samples)<br/>2024-12-20 to 2025-07-26"]
        C1 --> C5["Purge Gap 2 (14 Days)"]
        C1 --> C6["Held-Out Test Split (20% / 10,696 samples)<br/>2025-08-11 to 2026-02-17"]
    end

    subgraph Modeling ["Model Training & Benchmarking"]
        C2 --> D1["B1 Seasonal Naive Baseline"]
        C2 --> D2["B2 Seasonal Moving Average Baseline"]
        C2 --> D3["M1 Ridge Regression (Scaled)"]
        C2 --> D4["M2 Random Forest Regressor"]
        C2 --> D5["M3 HistGradientBoostingRegressor<br/>(Hyperparameter Search)"]
        D1 & D2 & D3 & D4 & D5 --> D6["Validation Evaluation & Model Selection"]
        D6 --> D7["Winner: HistGBM (lr=0.05, max_depth=8, min_leaf=20)"]
    end

    subgraph ConformalCalibration ["Split Conformal Prediction"]
        D7 --> E1["Compute Absolute Residuals on Validation Set"]
        E1 --> E2["Bucket by Horizon: h1-3, h4-7, h8-14"]
        E2 --> E3["80% Conformal Quantiles<br/>(h1-3: ±0.74, h4-7: ±1.71, h8-14: ±2.65)"]
    end

    subgraph Deployment ["Serving & Interfaces"]
        D7 & E3 --> F1["Serialized Artifact<br/>(models/model.joblib)"]
        F1 --> F2["FastAPI Serving Layer<br/>(src/hotel_forecast/api/main.py)"]
        F1 --> F3["Streamlit Operations Dashboard<br/>(app/dashboard.py)"]
        F2 --> G1["REST Endpoints: /health, /model-info, /forecast"]
        F3 --> G2["Manager UI: Trajectories, Intervals, Heatmap, Alerts"]
    end
```

---

## 2. Information Horizon and Point-in-Time Constraint

```mermaid
sequenceDiagram
    autonumber
    actor Manager as Hotel Operations Manager
    participant Sys as Forecasting System
    participant Ledger as Point-in-Time Ledger
    participant Model as HistGBM + Conformal Predictor

    Note over Manager,Ledger: Forecast Origin T = End of Day T
    Manager->>Sys: Request 1-14 day forecast for origin T
    Sys->>Ledger: Query bookings strictly with booking_date <= T
    Note over Ledger: Filter out cancellations occurring after T
    Ledger-->>Sys: Return observable on-the-books (OTB) bookings
    Sys->>Sys: Compute lag actuals observable at T (lag_same_weekday)
    Sys->>Sys: Assemble 23 point-in-time features
    Sys->>Model: Predict point estimate & calculate 80% interval
    Model-->>Sys: [prediction, lower_bound, upper_bound]
    Sys->>Manager: Display forecast trajectory & human-review alerts
```

---

## 3. Data Flow and Guardrails

1. **Zero Future Leakage:** All feature computation is centralized in `src/hotel_forecast/features/build.py:build_features()`. Every sub-query filters on `booking_date <= origin_T` and `cancellation_date <= origin_T`.
2. **Purge Gaps:** 14-day gap between training, validation, and test splits prevents overlapping multi-step horizon targets from contaminating adjacent partitions.
3. **Capacity Bounds:** All point predictions and conformal bounds are clipped to $[0, \text{capacity}]$ for physical realism.
4. **Asymmetric Loss Function:** Objective weights under-forecasting ($c_{\text{under}}=3.0$) $3\times$ higher than over-forecasting ($c_{\text{over}}=1.0$).
