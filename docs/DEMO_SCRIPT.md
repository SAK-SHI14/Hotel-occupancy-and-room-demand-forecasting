# 3 to 5-Minute Video Walkthrough Script

**Video Title:** Hotel Occupancy & Room-Demand Forecasting: End-to-End ML Pipeline  
**Target Time:** 4 Minutes (240 Seconds)  
**Presenter:** [YOUR NAME] (Reg No: [REG NO])  
**Company Assessment:** Algions AI Labs — AI & Machine Learning Track

---

## Visual & Spoken Storyboard

```
00:00 - 00:35  Problem Definition & Information Rigor
00:35 - 01:15  Data Engine, Cleaning & Leakage Canary Proof
01:15 - 02:00  Leakage-Safe Feature Engineering & Split Strategy
02:00 - 02:45  Model Benchmarks & Asymmetric Loss Optimization
02:45 - 03:30  Conformal Uncertainty & Live API / Dashboard Demo
03:30 - 04:00  Responsible AI, Limitations & Defense Summary
```

---

### [00:00 - 00:35] Problem Definition & Business Context
**Visual:** Show `docs/problem_definition.md` and the Streamlit dashboard banner.  
**Speech:**
> *"Hello! Today I am presenting my solution for the Algions AI Labs Machine Learning assessment: Hotel Occupancy & Room-Demand Forecasting.*  
> *In hotel operations, the revenue manager makes critical daily decisions on staffing, overbooking thresholds, and dynamic pricing for the upcoming 14 days.*  
> *Our goal is to forecast `occupied_room_nights` for each `(stay_date, room_category)` pair. Most importantly, we enforce a strict information cutoff at forecast origin $T$: at time $T$, we only know bookings and cancellations created on or before $T$. No future data can ever leak."*

---

### [00:35 - 01:15] Data Engine & Leakage Canary Proof
**Visual:** Show `reports/data_quality_report.md` and `reports/figures/leakage_canary.png`.  
**Speech:**
> *"We built a synthetic booking-level ledger with over 40,000 bookings across three years, encoding realistic seasonality, day-of-week demand, lead-time curves, and Indian public holidays.*  
> *To prove that our pipeline is truly leakage-free, I engineered a Leakage Canary experiment. When a leaky feature—the final occupancy of the stay date—is injected, the model reaches an artificially perfect 0.08 MAE. But when our pipeline guard rejects it, the honest model scores an MAE of 1.22 rooms. We also have unit tests that randomly mutate future ledger rows and prove feature invariance."*

---

### [01:15 - 02:00] Feature Engineering & Chronological Split Strategy
**Visual:** Show `docs/feature_list.md` and `reports/figures/split_timeline.png`.  
**Speech:**
> *"All 23 features are computed in a single function, `build_features()`. Key features include on-the-books rooms as-of origin $T$, pickup pace over 7 and 14 days, horizon-aware same-weekday lags, and OTA booking shares.*  
> *For our split, we strictly avoid random k-fold cross-validation. Instead, we use chronological splitting: 60% train, 20% validation, and 20% held-out test, separated by 14-day purge gaps. This ensures a 14-day multi-step forecast from the training period never overlaps with validation or test origins."*

---

### [02:00 - 02:45] Model Benchmarking & Asymmetric Cost
**Visual:** Show `reports/experiments.csv` and the model comparison table.  
**Speech:**
> *"We benchmarked against two sound hospitality baselines: B1 Seasonal Naive and B2 Seasonal 4-Week Moving Average. We evaluated Ridge regression, Random Forests, and HistGradientBoosting across hyperparameter grids.*  
> *Hospitality errors are asymmetric: under-forecasting costs 3x more due to emergency staffing and guest turnaways. Our tuned HistGradientBoosting model achieved 9.35% WAPE on validation and 10.0% WAPE on the test set, beating the seasonal naive baseline with a 67.7% skill score and slashing total business cost from $104,000 to $33,200."*

---

### [02:45 - 03:30] Conformal Intervals & Interactive Dashboard
**Visual:** Live screen recording of `streamlit run app/dashboard.py`. Select origin date, show confidence band, hover over occupancy heatmap.  
**Speech:**
> *"Because point forecasts are never enough for high-stakes operational planning, we implemented Split Conformal Prediction. Using validation residuals, we calibrated 80% prediction intervals across horizon buckets.*  
> *Notice how the interval grows realistically from ±0.74 rooms at day 1 to ±2.65 rooms at day 14. In our interactive Streamlit dashboard, managers can pick any origin date, view expected occupancy heatmaps, and immediately see automated 'Needs Human Review' flags whenever uncertainty spikes."*

---

### [03:30 - 04:00] Responsible AI & Viva Defense
**Visual:** Show FastAPI Swagger UI at `http://localhost:8000/docs` and `docs/responsible_ai.md`.  
**Speech:**
> *"Finally, our system is production-ready with a FastAPI serving layer, zero guest PII, determinism guaranteed via fixed random seed 42, and an automated audit trail logging every test evaluation.*  
> *The entire codebase is modular, fully tested with 45 pytest tests, and ready for technical walkthrough. Thank you!"*
