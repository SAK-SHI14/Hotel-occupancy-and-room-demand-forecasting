# Data Quality & Integrity Audit Report

## 1. Executive Summary
- **Raw Ledger Records:** 40,891
- **Cleaned Valid Records:** 40,571 (Dropped 320 corrupted records)
- **Audit Timeline:** Synthetic booking ledger multi-year assessment (2023–2026).
- **Privacy & PII Assessment:** Passed. Zero guest personal identifiers detected ([]).

---

## 2. Anomaly Detection & Cleaning Verification

| Anomaly Category | Detected (Raw) | Cleaned Ledger | Remediation Strategy | Action Status |
| :--- | :--- | :--- | :--- | :--- |
| **Missing Channel** | 123 (0.30%) | 0 | Impute with modal channel (`OTA`) | Resolved |
| **Duplicate Booking IDs** | 123 (0.30%) | 0 | Drop duplicate records (keep first) | Resolved |
| **Arrival < Booking Date** | 123 (0.30%) | 0 | Drop corrupted chronological records | Resolved |
| **Departure <= Arrival Date** | 122 (0.30%) | 0 | Drop non-positive stay durations | Resolved |
| **Negative / Zero ADR** | 61 (0.15%) | 0 | Impute with category median ADR | Resolved |
| **ADR Outliers (> 3x IQR)** | 122 (0.30%) | 0 | Impute with category median ADR | Resolved |
| **Invalid Categories** | 0 (0.00%) | 0 | Schema validation guard | Verified |
| **Invalid Room Counts** | 0 (0.00%) | 0 | Physical constraints guard | Verified |

---

## 3. Justification of Cleaning Actions

1. **Chronological Inconsistencies (Impossible Dates):**
   - *Issue:* Reservations where arrival date precedes booking date or departure date is on or before arrival.
   - *Decision:* Drop records entirely. Such records represent corrupted transmission payloads; attempting to guess true dates would introduce artificial assumptions.
2. **Duplicate Booking Identifiers:**
   - *Issue:* Identical opaque IDs assigned to multiple booking transactions.
   - *Decision:* Retain the first observed transaction and purge subsequent duplicates.
3. **Missing Distribution Channels:**
   - *Issue:* Null channel attributes prevent channel-specific cancellation modeling.
   - *Decision:* Impute with the majority distribution channel (`OTA`), preserving sample size without skewing distribution parameters.
4. **ADR Pricing Anomalies:**
   - *Issue:* Extreme values (negative rates or erroneous test rates like 99999.0).
   - *Decision:* Replace invalid non-positive or outlier values with the median ADR of the corresponding room category.
