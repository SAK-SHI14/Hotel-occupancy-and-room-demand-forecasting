# Responsible AI, Governance & Fairness Card

## 1. System Identity & Mission

- **Model Name:** Hotel Demand Forecaster (HistGradientBoostingRegressor + Split Conformal Intervals)
- **Primary Domain:** Hospitality Revenue Management & Operational Planning
- **Decision Scope:** Staffing schedules, inventory allocation, overbooking limits, and dynamic pricing guidance.
- **Human-in-the-Loop Philosophy:** The system is exclusively a **decision-support tool**. It does **not** autonomously adjust room rates or confirm staff cuts without manager oversight.

---

## 2. Privacy & Data Minimization (Privacy by Design)

- **Zero Guest Identifiers:** The raw booking ledger schema contains **no guest personal identifiable information (PII)**:
  - No names, emails, phone numbers, passport numbers, home addresses, or IP addresses.
  - Opaque `booking_id` hashes are generated purely for referential integrity during deduplication.
- **Automated PII Guardrail:** Verified automatically in CI via `tests/test_no_sensitive_data.py`. Any schema change that introduces PII keywords immediately fails builds.
- **GDPR / DPDP Compliance:** Since data is synthesized and anonymized at creation, no personal privacy rights or data subject requests are implicated.

---

## 3. Asymmetric Loss Function & Fairness

Hospitality operations face fundamentally asymmetric economic and service penalties:

$$\text{Loss}(y, \hat{y}) = c_{\text{under}} \cdot \max(y - \hat{y}, 0) + c_{\text{over}} \cdot \max(\hat{y} - y, 0)$$

- $c_{\text{under}} = 3.0$: Under-forecasting results in emergency overtime, staff burnout, guest turnaways (walk costs), brand reputation loss, and lost revenue.
- $c_{\text{over}} = 1.0$: Over-forecasting results in carrying empty room preparation costs and slight over-staffing.
- **Impact:** The objective explicitly protects operational staff against sudden, unpredicted demand surges while minimizing costly customer service failures.

---

## 4. Uncertainty Quantification & Human-in-the-Loop Escalation

- **Split Conformal Prediction:** Validated 80% empirical prediction intervals ensure managers have realistic error bounds for every horizon:
  - $h \in [1, 3]$: $\pm 0.74$ rooms (narrow window, high OTB confidence)
  - $h \in [4, 7]$: $\pm 1.71$ rooms (medium window)
  - $h \in [8, 14]$: $\pm 2.65$ rooms (long window, higher variance)
- **Automated Human Review Flag (`needs_human_review = True`):**
  - Triggered whenever the conformal interval spans $> 2\times$ the baseline quantile width, or when expected occupancy exceeds $95\%$ (high risk of overbooking) or falls below $5\%$ (unusual slump requiring marketing review).
  - Prompts operations managers with a prominent alert on the Streamlit dashboard and API response payload.

---

## 5. Failure Modes & Known Limitations

| Failure Mode | Root Cause | System Behavior & Mitigation |
|---|---|---|
| **Black Swan Demand Shocks** (Pandemic, natural disaster, air traffic strike) | Unprecedented exogenous disruptions outside historical training distribution. | The model will under-forecast sudden collapses or surges. Conformal intervals will fail to cover. Mitigation: Emergency manual override switch in dashboard. |
| **New Local Competitor Opening** | Unobserved supply increase in the immediate geographic submarket. | Historical pickup pace will gradually overestimate actual bookings. System adapts within 28 days as rolling mean lags update. |
| **Data Drift from Changing Cancellation Policies** | OTA channels introduce non-refundable policy changes. | Feature `ota_share_otb` captures channel shifts; monthly retraining re-calibrates conformal quantiles. |
| **Synthetic-to-Real Domain Gap** | Models trained on synthetic ledger generators may miss micro-correlations present in specific property PMS ledgers. | Public dataset adapter (`scripts/fetch_public_data.py`) provides an out-of-domain sanity pipeline. |

---

## 6. Model Maintenance & Monitoring Plan

1. **Drift Detection:** Compare weekly realized WAPE and coverage against the $80\%$ benchmark. If coverage falls below $70\%$ over a rolling 14-day window, trigger automated model recalibration.
2. **Monthly Retraining:** Roll forward the chronological split window by 30 days, retaining the 14-day purge gap.
3. **Audit Trail Logging:** All held-out test evaluations write timestamped entries to `reports/test_run_log.json` to detect performance regressions over time.
