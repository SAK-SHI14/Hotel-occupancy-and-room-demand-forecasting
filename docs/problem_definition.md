# Problem Definition: Hotel Occupancy & Room-Demand Forecasting

## 1. Executive Summary & Business Context

Hotel revenue management and hotel operations require forward-looking demand estimates to make critical daily operational and commercial decisions. Every morning, a hotel revenue manager and operations lead review forecasts for the subsequent 14 days ($h \in \{1, 2, \dots, 14\}$).

Key daily operational decisions driven by this forecast include:
1. **Housekeeping & Front-Desk Staffing**: Scheduling variable shift labor based on anticipated guest volume.
2. **Overbooking Limits**: Establishing risk-calibrated overbooking buffers to compensate for anticipated cancellations without causing costly walk-outs or brand erosion.
3. **Dynamic Pricing & Distribution**: Opening or closing discounted rate fences, managing inventory on Online Travel Agencies (OTAs), and setting minimum length-of-stay restrictions.
4. **Food & Beverage / Consumable Procurement**: Estimating perishable ingredient quantities and amenities based on confirmed and projected guest count.

---

## 2. Formal Problem Formulation

### 2.1 Prediction Unit
The fundamental unit of observation and prediction is the tuple:
$$(\text{stay\_date } s, \text{room\_category } c)$$
where:
- $s$ is a calendar date representing the guest's night of stay.
- $c \in \{\text{Standard}, \text{Deluxe}, \text{Suite}, \text{Family}\}$ represents the room category with distinct capacities and customer profiles.

### 2.2 Target Variables
1. **Primary Target (`occupied_room_nights`):**
   The total count of rooms within category $c$ occupied on stay date $s$ by active, non-cancelled reservations:
   $$y(s, c) = \sum_{b \in \mathcal{B}} n\_rooms(b) \cdot \mathbb{I}\Big(category(b) = c \wedge arrival(b) \le s < departure(b) \wedge \neg is\_cancelled(b)\Big)$$
   where $y(s, c) \in [0, \text{capacity}_c]$.

2. **Derived Operational Metric (`occupancy_rate`):**
   The ratio of occupied rooms relative to total physical capacity of that category:
   $$\text{occupancy\_rate}(s, c) = \frac{y(s, c)}{\text{capacity}_c} \in [0.0, 1.0]$$

### 2.3 Forecast Origin $T$ and Forecast Horizon $h$
- **Forecast Origin $T$**: The end of day $T$ (midnight cutoff). The forecasting system runs at the start of operational day $T+1$ before morning operational briefings.
- **Forecast Horizon $h$**: Daily steps forward from the origin, $h \in \{1, 2, \dots, 14\}$, where the target stay date is:
  $$s = T + h$$
- **Model Topology**: A unified direct multi-horizon regression framework parameterized with $h$ explicitly encoded as a continuous and discrete feature, enabling the model to learn horizon-dependent decay of pickup signals.

---

## 3. The Information Horizon & Strict Anti-Leakage Rule

In time-series forecasting, historical backtesting frequently suffers from subtle look-ahead bias (data leakage). To guarantee real-world fidelity, we enforce a strict mathematical point-in-time boundary:

> **Information Boundary Condition:**
> At forecast origin $T$, any transaction, modification, or cancellation that occurred after $T$ is strictly unobservable.
> 
> For all historical booking records $b$:
> 1. A booking is known if and only if:
>    $$\text{booking\_date}(b) \le T$$
> 2. A booking is known to be cancelled if and only if:
>    $$\text{is\_cancelled}(b) = \text{True} \quad \wedge \quad \text{cancellation\_date}(b) \le T$$
> 3. If a booking has $\text{cancellation\_date}(b) > T$, it **must be treated as an active, confirmed booking** as of origin $T$.

Any model feature violating this boundary constitutes temporal data leakage, artificially inflating offline backtest performance while causing catastrophic failure in live deployment.

---

## 4. Evaluation Framework & Success Metrics

To balance mathematical rigor with operational interpretability, model performance is evaluated across a complementary suite of metrics:

### 4.1 Primary Metric: Weighted Absolute Percentage Error (WAPE)
$$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}$$
*Why WAPE?* Traditional MAPE suffers from division-by-zero or extreme skew when actual room nights are low (common in low-capacity suites). WAPE weights errors by volume, providing revenue executives with an intuitive, volume-weighted percentage error.

### 4.2 Standard Accuracy Metrics
- **Mean Absolute Error (MAE):** Average error in absolute room units.
  $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE):** Penalizes large forecast misses more heavily.
  $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
- **Mean Bias Error:** Quantifies persistent over- or under-forecasting tendencies.
  $$\text{Bias} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)$$

### 4.3 Skill Score vs. Seasonal Benchmark
To prove genuine predictive value above naive heuristics:
$$\text{Skill Score} = 1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{Seasonal Naive B1}}}$$
A positive skill score confirms the machine learning pipeline extracts genuine demand signals beyond calendar seasonality.

### 4.4 Asymmetric Business Cost
In hotel operations, under-predicting demand is asymmetric to over-predicting demand:
- **Under-forecasting ($y > \hat{y}$):** Leads to understaffing, turning away high-paying walk-in guests, or premature rate discounting. Cost weight: $c_{\text{under}} = 3.0$.
- **Over-forecasting ($\hat{y} > y$):** Leads to surplus labor costs and holding empty rooms at non-competitive rates. Cost weight: $c_{\text{over}} = 1.0$.

$$\text{Business Cost} = \sum_{i=1}^N \Big[ c_{\text{under}} \cdot \max(y_i - \hat{y}_i, 0) + c_{\text{over}} \cdot \max(\hat{y}_i - y_i, 0) \Big]$$

---

## 5. Uncertainty Quantification & Human-in-the-Loop Protocol

No point forecast is 100% certain, especially 14 days ahead. Operational decisions require calibrated uncertainty bounds:
1. **Conformal Prediction Intervals**: Using split conformal prediction on validation set residuals, we generate 80% prediction intervals $[\hat{y}^{\text{lower}}, \hat{y}^{\text{upper}}]$ guaranteed to widen realistically as the forecast horizon $h$ increases.
2. **Bounds Enforcement**: Predictions are strictly bounded within physical reality: $[0, \text{capacity}_c]$.
3. **Human-in-the-Loop Flag (`needs_human_review`):**
   The forecasting tool is strictly a **decision-support copilot**, not an autonomous agent. A forecast automatically triggers an operational alert requiring human manager review when:
   - Relative interval width exceeds $40\%$ of category capacity:
     $$\frac{\hat{y}^{\text{upper}} - \hat{y}^{\text{lower}}}{\text{capacity}_c} > 0.40$$
   - The stay date coincides with an unprecedented demand shock or an anomalous holiday/event cluster.
