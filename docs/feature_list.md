# Feature Catalog: Leakage-Safe Hotel Demand Features

## 1. Overview & Information Horizon Guarantee

Every feature utilized in the forecasting models is strictly evaluated as-of the **forecast origin $T$**.
Any booking transaction where $\text{booking\_date} > T$ or cancellation where $\text{cancellation\_date} > T$ is strictly omitted or treated as active.

Target stay date: $s = T + h$, where $h \in \{1, 2, \dots, 14\}$.

---

## 2. Comprehensive Feature Table

| Feature Name | Group | Data Type | Mathematical Definition / Formula | Known at Origin $T$ Justification |
| :--- | :--- | :--- | :--- | :--- |
| `otb_rooms_T` | OTB / Pickup | Float | $\sum_{b \in \mathcal{B}_c} n\_rooms(b) \cdot \mathbb{I}(\text{arr} \le s < \text{dep} \wedge \text{book} \le T \wedge (\neg \text{canc} \vee \text{canc\_date} > T))$ | Evaluates only reservations registered $\le T$ that have not been cancelled as of $T$. |
| `otb_rooms_T_minus_7` | OTB / Pickup | Float | Same formula as `otb_rooms_T` evaluated as-of $T - 7$ | Historical snapshot of the ledger 7 days prior; strictly $\le T - 7 < T$. |
| `otb_rooms_T_minus_14`| OTB / Pickup | Float | Same formula as `otb_rooms_T` evaluated as-of $T - 14$ | Historical snapshot of the ledger 14 days prior; strictly $\le T - 14 < T$. |
| `pickup_pace_7d` | OTB / Pickup | Float | `otb_rooms_T - otb_rooms_T_minus_7` | Velocity of new bookings added over the last 7 calendar days before $T$. |
| `pickup_pace_14d` | OTB / Pickup | Float | `otb_rooms_T - otb_rooms_T_minus_14` | Velocity of new bookings added over the last 14 calendar days before $T$. |
| `otb_capacity_share` | OTB / Pickup | Float | `otb_rooms_T / capacity` | Proportion of category capacity committed as-of origin $T$. |
| `lag_same_weekday` | Realized Lags | Float | Realized occupancy at $s_{\text{lag}} = s - 7 \cdot \lceil h/7 \rceil$ | Since $h \le 7 \cdot \lceil h/7 \rceil$, $s_{\text{lag}} \le T$; this is the most recent completed stay on the same weekday. |
| `rolling_mean_7d` | Realized Lags | Float | $\frac{1}{7}\sum_{d=0}^6 Y(T - d, c)$ | Mean realized occupancy across the 7 most recent days ending at $T$; all stay dates $\le T$. |
| `rolling_mean_28d` | Realized Lags | Float | $\frac{1}{28}\sum_{d=0}^{27} Y(T - d, c)$ | Mean realized occupancy across the 28 most recent days ending at $T$; all stay dates $\le T$. |
| `same_weekday_4w_mean`| Realized Lags | Float | $\frac{1}{4}\sum_{j=0}^3 Y(s_{\text{lag}} - 7j, c)$ | Average realized occupancy of the past 4 same-weekdays prior to $T$; all dates $\le T$. |
| `horizon` | Horizon | Integer | $h = (s - T)$ in days | Operational parameter defining the forward forecast horizon ($1 \le h \le 14$). |
| `day_of_week` | Calendar | Integer | $s.\text{weekday}() \in [0, 6]$ (0=Monday, 6=Sunday) | Calendar property of the target stay date $s$. |
| `month` | Calendar | Integer | $s.\text{month} \in [1, 12]$ | Calendar property of the target stay date $s$. |
| `week_of_year` | Calendar | Integer | $s.\text{isocalendar}()[1] \in [1, 53]$ | ISO calendar week of the target stay date $s$. |
| `is_weekend` | Calendar | Binary | $\mathbb{I}(s.\text{weekday}() \in \{4, 5\})$ (Fri/Sat night) | Static calendar classification for hotel leisure weekend. |
| `is_holiday` | Calendar | Binary | $\mathbb{I}(s \in \text{Indian Gazetted Holidays})$ | Official public holiday calendar known in advance. |
| `days_to_next_holiday`| Calendar | Integer | Calendar days from $s$ to next upcoming holiday | Fixed calendar distance known in advance. |
| `days_since_last_holiday`| Calendar| Integer | Calendar days from $s$ to most recent past holiday | Fixed calendar distance known in advance. |
| `is_event_period` | Calendar | Binary | $\mathbb{I}(s \in \text{Scheduled Event Window})$ | Pre-scheduled annual conference or festival dates known in advance. |
| `room_category_code` | Identity | Integer | Categorical ordinal: Standard=0, Deluxe=1, Family=2, Suite=3 | Known physical category classification. |
| `capacity` | Identity | Integer | Category physical capacity (Standard=50, Deluxe=35, Family=20, Suite=15) | Fixed hotel property infrastructure attribute. |
| `avg_lead_time_otb` | Dynamics | Float | Mean $(s - \text{booking\_date})$ for active OTB bookings as-of $T$ | Computed strictly over confirmed reservations on the books as of $T$. Defaults to 0 if $OTB=0$. |
| `ota_share_otb` | Dynamics | Float | $\frac{\text{OTB}_{\text{OTA}}(T)}{\text{OTB}(T)}$ | Proportion of current OTB coming from OTA channel as-of $T$. Defaults to 0.0 if $OTB=0$. |

---

## 3. Strict Target Specification

- **Target Variable (`occupied_room_nights`):**
  Final realized rooms occupied on stay date $s$ by confirmed, non-cancelled guests:
  $$y(s, c) = \sum_{b \in \mathcal{B}_c} n\_rooms(b) \cdot \mathbb{I}(\text{arrival} \le s < \text{departure} \wedge \neg \text{is\_cancelled})$$
- **Target Variable (`occupancy_rate`):**
  $$\text{occupancy\_rate}(s, c) = \frac{y(s, c)}{\text{capacity}_c}$$
- **Zero Leakage Rule:** No feature column utilizes or references $y(s, c)$ or any post-$T$ transaction.
