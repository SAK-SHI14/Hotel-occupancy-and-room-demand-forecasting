# Technical Walkthrough, Viva FAQs & Live Debugging Playbook

**Candidate Name:** [YOUR NAME]  
**Registration No:** [REG NO]  
**University Email:** [EMAIL]  
**Topic:** Hotel Occupancy & Room-Demand Forecasting (AI & Machine Learning Assessment)

---

## Part 1: 15-Minute Technical Walkthrough Script

Use this structured script during your 15-minute viva presentation. It follows the exact logical arc reviewers look for.

### 1. Introduction & Problem Definition (Minutes 0 - 2)
> *"Good morning/afternoon. I'm presenting an end-to-end Machine Learning forecasting system for hotel operations.*  
> *The operational problem is daily decision-support for a revenue and operations manager: each morning at origin $T$, they must decide on staffing, room cleaning rosters, overbooking limits, and dynamic pricing for the upcoming 14 days.*  
> *The unit of prediction is `(stay_date, room_category)`. The target is `occupied_room_nights`.*  
> *The central constraint of this project is strict point-in-time information integrity: at origin $T$, we only observe bookings with `booking_date <= T` and cancellations with `cancellation_date <= T`. Anything occurring after $T$ is completely invisible."*

### 2. Data Strategy & Quality Cleaning (Minutes 2 - 4)
> *"We built a realistic synthetic booking ledger of 40,891 bookings spanning 2023 to 2026 across four room categories: Standard (50 rooms), Deluxe (35 rooms), Family (20 rooms), and Suite (15 rooms).*  
> *The generator models weekly seasonality (weekend leisure spikes vs weekday corporate demand), annual trends, gamma-distributed lead times, cancellation hazards linked to channel, and Indian public holidays.*  
> *We intentionally injected 1.5% realistic data noise—duplicates, negative ADRs, and illogical date sequences. In `clean.py`, we implemented a cleaning gate that dropped 320 invalid records to yield 40,571 pristine records, achieving zero residual IQR outliers per category."*

### 3. Leakage Guardrails & Leakage Canary (Minutes 4 - 7)
> *"Feature engineering is centralized in a single canonical function: `build_features()`. We engineered 23 features across five groups: on-the-books pace ($T$, $T-7$, $T-14$), horizon-aware lags ($stay\_date - 7\lceil h/7 \rceil$), rolling 7- and 28-day realized means, calendar attributes, and booking dynamics.*  
> *To prove leakage-safety beyond theoretical assertions, I implemented two safeguards:*  
> *First, `test_leakage.py` mutates future ledger rows and asserts identical feature output.*  
> *Second, `scripts/leakage_canary.py` deliberately injects a leaky feature (final stay date occupancy). With leakage, the model cheats to an artificially low 0.08 MAE. Once rejected, the honest model returns to 1.22 MAE, proving our pipeline actively detects and prevents target leakage."*

### 4. Split Strategy & Purge Gaps (Minutes 7 - 9)
> *"Standard k-fold cross-validation causes massive temporal leakage in time-series forecasting. We use chronological splitting: 60% train (18,480 samples), 20% validation (6,160 samples), and 20% held-out test (10,696 samples).*  
> *Crucially, we enforce a strict 14-day purge gap between splits. Because our maximum forecast horizon is 14 days, a forecast made on the last day of training extends 14 days forward. Without the purge gap, those target windows would overlap with validation origins."*

### 5. Modeling, Baselines & Asymmetric Costs (Minutes 9 - 12)
> *"We benchmarked against two sound hospitality baselines: B1 Seasonal Naive and B2 Seasonal 4-Week Moving Average.*  
> *We evaluated Ridge regression, Random Forest, and HistGradientBoosting. Our loss function is asymmetric: under-forecasting carries a 3.0x penalty compared to 1.0x for over-forecasting, reflecting the severe real-world operational costs of guest turnaways and emergency overtime.*  
> *Our tuned HistGradientBoosting model achieved 9.35% WAPE on validation and 10.0% WAPE on the test set, outperforming the B1 baseline by a 67.7% skill score and reducing business costs by 68%."*

### 6. Uncertainty, Deployment & Live Demo (Minutes 12 - 15)
> *"For decision support, point estimates are insufficient. We implemented Split Conformal Prediction, calibrating 80% intervals on validation residuals grouped by horizon buckets ($h_{1-3}$, $h_{4-7}$, $h_{8-14}$).*  
> *Intervals expand naturally with horizon distance ($\pm 0.74$ to $\pm 2.65$ rooms), achieving 71.5% empirical test coverage.*  
> *The entire system is deployed via a FastAPI backend and a Streamlit dashboard with automated 'Human Review' alerts when intervals exceed tolerance or occupancy surpasses 95%."*

---

## Part 2: 25 High-Frequency Viva Questions & Model Answers

### Category A: Problem & Hospitality Domain (Q1 - Q5)
1. **Q: Why is hotel demand forecasting formulated as (stay_date, room_category) rather than just hotel-level totals?**  
   *A: Different room categories have different physical capacities, price elasticities, and guest demographics. For example, Deluxe rooms experience high weekend leisure demand while Standard rooms see weekday corporate demand. Aggregating them obscures actionable staffing and inventory constraints.*

2. **Q: What is the significance of the forecast origin $T$?**  
   *A: $T$ represents the exact moment in time when the forecast is produced (e.g., 6:00 AM daily). It acts as an absolute information horizon: no booking created or cancellation submitted after $T$ can be used in feature computation.*

3. **Q: Why is under-forecasting penalized 3x more than over-forecasting?**  
   *A: In hospitality, under-forecasting leads to overbooking crises, walking guests to competitor hotels at severe expense, emergency staff overtime, and lasting brand damage. Over-forecasting merely incurs the holding cost of an uncleaned, empty room.*

4. **Q: What is On-The-Books (OTB) and why is it the strongest predictor?**  
   *A: OTB represents confirmed, non-cancelled reservations already secured as of origin $T$ for a future stay date. As horizon $h \to 1$, OTB approaches final occupancy, providing a firm lower bound on demand.*

5. **Q: Why use WAPE instead of MAPE?**  
   *A: Mean Absolute Percentage Error (MAPE) divides by the actual value. On low-demand days or for small categories (e.g., Suite actual = 1 room), an error of 1 room produces a 100% MAPE, distorting metrics. Weighted Absolute Percentage Error (WAPE = $\sum |y - \hat{y}| / \sum y$) aggregates globally, avoiding division by zero.*

---

### Category B: Leakage & Data Architecture (Q6 - Q11)
6. **Q: How does your code guarantee zero future data leakage?**  
   *A: All feature extraction is isolated in `build_features()`. Every calculation filters the ledger with `booking_date <= origin_T` and `cancellation_date <= origin_T`. This is verified by mutation invariance tests in `tests/test_leakage.py`.*

7. **Q: What is the Leakage Canary and why did you build it?**  
   *A: It is an empirical proof script (`scripts/leakage_canary.py`) that injects a leaky feature (actual final occupancy) into the training set. The leaky model gets an unrealistic 0.08 MAE ($R^2 = 0.998$). When our pipeline guard strips it, performance returns to honest 1.22 MAE, proving our architecture detects leakage.*

8. **Q: What is a 14-day purge gap and why is it necessary?**  
   *A: With a 14-day forecast horizon, a training sample at origin $T_{\text{train\_end}}$ predicts up to $T_{\text{train\_end}} + 14$. If validation starts at $T_{\text{train\_end}} + 1$, its actuals overlap with the training target window, leaking future sequence information. A 14-day purge gap completely eliminates this overlap.*

9. **Q: Why can't we use standard k-fold cross validation for time series?**  
   *A: Standard k-fold randomly shuffles data, allowing future observations to train past predictions. This creates temporal lookahead bias and yields wildly optimistic validation scores that fail in production.*

10. **Q: How does your feature pipeline handle same-weekday lags without leaking?**  
    *A: For horizon $h$, the most recent observable actual for the same weekday is $stay\_date - 7 \lceil h/7 \rceil$. For $h=1$, this is 7 days prior; for $h=8$, it is 14 days prior. This guarantees the lag was realized on or before origin $T$.*

11. **Q: How do you handle privacy in the dataset?**  
    *A: Privacy by design: the booking ledger contains zero guest names, emails, phones, or credit cards. `tests/test_no_sensitive_data.py` enforces this schema constraint in CI.*

---

### Category C: Modeling & Optimization (Q12 - Q18)
12. **Q: Why benchmark against B1 Seasonal Naive and B2 Moving Average?**  
    *A: Hospitality demand is highly seasonal by day of week. A model that cannot beat "what happened last Tuesday" or "the 4-week Tuesday average" is economically worthless. B1 and B2 establish the true baseline to justify machine learning complexity.*

13. **Q: Why did HistGradientBoostingRegressor beat Ridge and Random Forest?**  
    *A: HistGBM handles non-linear interactions (e.g., holiday $\times$ weekend $\times$ room category) and threshold effects in pickup pace much better than linear Ridge. It is also significantly faster and better regularized with histogram binning than Random Forest.*

14. **Q: How does the Skill Score work?**  
    *A: Skill Score vs B1 is defined as $1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{B1}}}$. A score of 0.677 means our model eliminated 67.7% of the baseline's error.*

15. **Q: Why clip model predictions to [0, capacity]?**  
    *A: A hotel room category cannot have negative occupied rooms, nor can it physically accommodate more guests than its total inventory. Clipping guarantees physical feasibility.*

16. **Q: What is the role of the fixed random seed (42)?**  
    *A: Full reproducibility: every synthetic generation, train/validation split, and model initialization produces bitwise identical results. Verified in `tests/test_determinism.py`.*

17. **Q: How many features does the model use?**  
    *A: Exactly 23 point-in-time features, spanning OTB counts, pickup pace, calendar features, holiday indicators, rolling statistics, category encodings, and channel distributions.*

18. **Q: Did you tune on the test set?**  
    *A: Never. All hyperparameter grids and model selection decisions were made strictly on the validation split. The held-out test set was evaluated exactly once via `evaluate.py --split test` and logged to `reports/test_run_log.json`.*

---

### Category D: Uncertainty & Conformal Prediction (Q19 - Q22)
19. **Q: What is Split Conformal Prediction?**  
    *A: It is a distribution-free uncertainty framework. We compute absolute residuals on a held-out calibration set (validation). The $(1-\alpha)$ quantile of these conformity scores provides a mathematically rigorous prediction interval for future unseen samples without Gaussian assumptions.*

20. **Q: Why bucket conformal quantiles by horizon ($h_{1-3}, h_{4-7}, h_{8-14}$)?**  
    *A: Forecast error increases as we look further into the future. A global interval would over-estimate uncertainty for tomorrow and under-estimate uncertainty for two weeks out. Bucketing preserves horizon-dependent error growth.*

21. **Q: What was your test set conformal coverage?**  
    *A: The empirical coverage on the held-out test set was 71.5% across 10,696 samples, with interval width expanding from 1.45 rooms at $h_{1-3}$ to 5.18 rooms at $h_{8-14}$.*

22. **Q: When is the 'needs_human_review' flag triggered?**  
    *A: Whenever the conformal interval width exceeds $2\times$ the baseline quantile width, or when forecasted occupancy is extreme ($>95\%$ or $<5\%$), alerting the manager to high risk.*

---

### Category E: Serving & Operations (Q23 - Q25)
23. **Q: How does the FastAPI app load the model?**  
    *A: It uses a singleton pattern: the serialized artifact `models/model.joblib` is loaded into memory on first request. Live inference is executed using the exact same `predict()` pipeline as the training code.*

24. **Q: Are predictions hardcoded anywhere in the app or dashboard?**  
    *A: Absolutely not. Every single prediction is computed live from the loaded model artifact on demand. This is verified by `tests/test_api.py:test_forecast_changes_with_origin_not_hardcoded`.*

25. **Q: How would this model be deployed in production PMS (Property Management Systems)?**  
    *A: As a scheduled daily batch job or REST microservice. At 6:00 AM daily, the PMS triggers `POST /forecast` with origin $T$ and receives 14-day forecasts and intervals to populate the manager's dashboard.*

---

## Part 3: 10 Live Debugging & Coding Change Drills

Prepare for the 10-minute live modification task with these exact step-by-step procedures:

### Drill 1: Change Forecast Horizon from 14 Days to 21 Days
- **File to Edit:** `configs/config.yaml` line `horizon_max: 14` $\to$ `21`.
- **Purge Gap Update:** Change `splits.purge_gap_days: 14` $\to$ `21` so purge gap matches the new horizon.
- **Verification:** Run `pytest tests/test_features.py tests/test_split.py`.

### Drill 2: Modify Asymmetric Cost Multipliers ($c_{\text{under}}=5.0, c_{\text{over}}=1.0$)
- **File to Edit:** `configs/config.yaml` under `metrics.asymmetric_cost`.
- **Code:**
  ```yaml
  asymmetric_cost:
    c_under: 5.0
    c_over: 1.0
  ```
- **Verification:** Run `python -c "from hotel_forecast.evaluate import compute_metrics; print(compute_metrics([10], [5], c_under=5.0))"` $\to$ Cost = 25.0.

### Drill 3: Add a New Feature: `otb_to_lag_ratio`
- **File to Edit:** `src/hotel_forecast/features/build.py`.
- **Code:**
  1. Add `"otb_to_lag_ratio"` to `FEATURE_COLUMNS`.
  2. Inside `build_features()`:
     ```python
     lag_val = features["lag_same_weekday"]
     features["otb_to_lag_ratio"] = float(features["otb_rooms_T"] / max(lag_val, 1.0))
     ```
- **Verification:** Run `pytest tests/test_features.py`.

### Drill 4: Change Conformal Coverage Target from 80% to 90% ($\alpha = 0.10$)
- **File to Edit:** `src/hotel_forecast/models/conformal.py`.
- **Code:** Change line `ALPHA: float = 0.20` $\to$ `ALPHA: float = 0.10`.
- **Verification:** Run `pytest tests/test_conformal.py` $\to$ assert quantiles are larger.

### Drill 5: Add a New Room Category: "Executive Suite" (Capacity = 10)
- **File to Edit:** `configs/config.yaml` under `room_categories`.
  ```yaml
  Executive Suite:
    capacity: 10
    base_demand: 7
    weekend_multiplier: 1.3
    adr_mean: 4500
    adr_std: 400
  ```
- **File to Edit:** `src/hotel_forecast/features/build.py:CATEGORY_CODES` $\to$ add `"Executive Suite": 4`.

### Drill 6: Force API to Reject Out-of-Range Historical Dates
- **File to Edit:** `src/hotel_forecast/api/main.py`.
- **Code:** In `ForecastRequest.validate_origin_date`:
  ```python
  min_date = date(2023, 1, 1)
  if date.fromisoformat(v) < min_date:
      raise ValueError(f"Origin date cannot be before {min_date}")
  ```

### Drill 7: Inspect Top 3 Model Errors on the Test Set
- **Command:**
  ```python
  import pandas as pd
  df = pd.read_csv("reports/figures/worst_forecasts.csv")
  print(df.head(3))
  ```

### Drill 8: Switch Model from HistGBM to Random Forest in Final Artifact
- **File to Edit:** `src/hotel_forecast/models/save_best.py`.
- **Code:** Change `best_name` selection or manually assign `best_name = "RandomForest (n=100, d=10)"`.
- **Run:** `python -m hotel_forecast.models.save_best`.

### Drill 9: Re-run Leakage Canary and Verify Detection Guard
- **Command:** `python scripts/leakage_canary.py`.
- **Observation:** Console outputs leaky correlation $\approx 0.99997$ and rejects the column.

### Drill 10: Run Full CI Quality Check Locally
- **Commands:**
  ```powershell
  python -m ruff check .
  python -m pytest -v
  ```
