# Data Card: Hotel Booking Demand Ledger

## 1. Dataset Overview & Provenance

- **Dataset Name:** Synthetic Hotel Booking Ledger (`booking_ledger.csv` / `booking_ledger_clean.csv`)
- **Version:** 1.0.0
- **Primary Generator Script:** `src/hotel_forecast/data/generate.py`
- **Cleaning & Validation:** `src/hotel_forecast/data/clean.py` & `src/hotel_forecast/data/validate.py`
- **Record Count:** 40,891 raw bookings $\rightarrow$ 40,571 validated transactions retained.
- **Temporal Horizon:** Over 3 full calendar years (2023-01-01 to 2026-03-01).
- **Primary Use Case:** Point-in-time training and evaluation of 14-day forward hotel room demand forecasting models.

> [!IMPORTANT]
> **Synthetic Data Statement & Generalization Limits:**
> This dataset is synthetically generated via a parametric stochastic process. While explicitly calibrated to reflect empirical hospitality dynamics (seasonality, lead times, channel cancellation rates, price elasticity), synthetic data cannot capture unobserved human behavioral shifts, macroeconomic black-swan events, or property-specific operational frictions. Real-world deployment requires retraining and calibrating on actual Property Management System (PMS) / Central Reservation System (CRS) event logs.

---

## 2. Privacy-by-Design & Ethical Considerations

- **Personally Identifiable Information (PII):** 0% (Strictly zero guest identifiers).
- **Opaque Identifiers:** The only identifier is a surrogate alphanumeric string `booking_id` (e.g. `BK-100001`).
- **Data Protection Compliance:** Complies with privacy-by-design guidelines (GDPR, DPDP Act) as no personal, contact, financial, or demographic guest attributes exist in the data schema.

---

## 3. Schema & Column Specifications

| Column Name | Data Type | Description | Example | Nullable |
| :--- | :--- | :--- | :--- | :--- |
| `booking_id` | String | Opaque unique booking transaction identifier | `"BK-100042"` | No |
| `booking_date` | String (ISO) | Date transaction was created in CRS | `"2024-03-12"` | No |
| `arrival_date` | String (ISO) | Check-in stay date | `"2024-04-05"` | No |
| `departure_date` | String (ISO) | Check-out date | `"2024-04-08"` | No |
| `room_category` | Categorical | Room category: `Standard`, `Deluxe`, `Suite`, `Family` | `"Deluxe"` | No |
| `n_rooms` | Integer | Number of physical rooms booked in this transaction ($\ge 1$) | `1` | No |
| `channel` | Categorical | Distribution channel: `Direct`, `OTA`, `Corporate` | `"OTA"` | No (Imputed) |
| `is_cancelled` | Boolean | True if booking was ultimately cancelled | `True` | No |
| `cancellation_date`| String (ISO) | Date cancellation was registered (null if active) | `"2024-03-28"` | Yes |
| `adr` | Float | Average Daily Rate (nightly room price in INR) | `4350.00` | No |

---

## 4. Property Profile & Category Capacities

The simulated property is a 120-room full-service hotel operating in India:

| Category | Room Capacity | Base ADR (INR) | Primary Guest Segment | Demand Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | 50 | ₹2,500 | Business / Budget travelers | High corporate share, short lead time, high turnover |
| **Deluxe** | 35 | ₹4,200 | Upscale leisure & corporate | Balanced weekly demand, high sensitivity to events |
| **Family** | 20 | ₹5,500 | Vacationing families & groups | Weekend leisure surge, multi-night stays (2-5 nights) |
| **Suite** | 15 | ₹8,500 | Luxury travelers & VIPs | Peak holiday/wedding demand, longer planning horizon |
| **Total** | **120** | - | - | - |

---

## 5. Generator Mathematical Assumptions

### 5.1 Seasonality & Secular Trend
Daily arrival expectations follow:
$$\lambda_t = \lambda_{\text{base}} \cdot M_{\text{season}}(t) \cdot M_{\text{trend}}(t) \cdot M_{\text{holiday}}(t) \cdot M_{\text{event}}(t)$$
- **Annual Seasonality ($M_{\text{season}}$):**
  - High Season (Nov–Feb): $1.30\times$ (Winter tourism and weddings).
  - Shoulder Season (Mar, Oct): $1.15\times$.
  - Low Season (Jun–Aug): $0.75\times$ (Monsoon lull).
- **Secular Growth ($M_{\text{trend}}$):** $+4.0\%$ compounding per annum over the 3-year baseline.
- **Weekly Dynamics:**
  - Corporate bookings cluster Mon–Thu.
  - Leisure bookings (Family, Deluxe, Suite) peak Fri–Sat.

### 5.2 Holiday & Event Calendar
- **Public Holidays:** Integrated with the official Indian gazetted calendar via `holidays.country_holidays('IN')`, injecting a $+25\%$ arrival lift.
- **Synthetic Local Events:** Multi-day demand surges (Annual Tech Summit in Feb: $+45\%$ corporate; Cultural Festival in Nov: $+45\%$ leisure; Medical Congress in Aug: $+45\%$ corporate).

### 5.3 Lead-Time Dynamics
Lead times are sampled from channel- and segment-specific Gamma distributions:
$$\text{lead\_time} \sim \text{Gamma}(\alpha, \beta)$$
- Corporate: $\alpha=2.0, \beta=4.0$ ($\mu \approx 8$ days, highly concentrated).
- Leisure / Standard: $\alpha=2.5, \beta=9.0$ ($\mu \approx 22.5$ days, right-skewed).
- Holiday getaways: $\alpha=3.5, \beta=8.0$ ($\mu \approx 28$ days).

### 5.4 Cancellation Dynamics
Cancellation probability is modeled as a function of channel vulnerability and lead time:
$$P(\text{cancel}) = \min\left(0.65, P_{\text{base}}(\text{channel}) \cdot (1.0 + 0.012 \cdot \text{lead\_time})\right)$$
where $P_{\text{base}}(\text{OTA}) = 0.28$, $P_{\text{base}}(\text{Corporate}) = 0.14$, and $P_{\text{base}}(\text{Direct}) = 0.10$.
When cancelled, cancellation timestamps are sampled between booking date and arrival date via a Beta distribution ($\alpha=2.0, \beta=1.5$), reflecting that cancellation velocity accelerates as the check-in date approaches.

### 5.5 Injected Anomalies (Controlled Noise)
To ensure the data cleaning and validation suite is genuine and robust, controlled anomalies (~1.5% noise rate) were injected:
- Missing channel values ($N = 119$).
- Duplicate booking IDs ($N = 127$).
- Chronological date inversions ($\text{arrival} < \text{booking}$ or $\text{departure} \le \text{arrival}$; $N = 250$).
- Outlier ADR pricing (negative prices or $99,999.00$ test values; $N = 117$).

---

## 6. External Public Benchmark Dataset Comparison

For comparative domain analysis, the prominent open-access hotel demand dataset is:
- **Reference:** Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49.
- **Source:** Two Portuguese hotels (H1: Resort Hotel, H2: City Hotel) spanning 2015–2017 (~119,390 records).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Why a synthetic ledger was designed as the primary engine:**
  While the Antonio et al. dataset is valuable, it provides aggregated lead times without explicit microsecond/daily transaction ledger timestamps for point-in-time state reconstruction. Developing our dedicated booking-level ledger enables mathematically provable, point-in-time leakage guarantees at any arbitrary forecast origin $T$.
