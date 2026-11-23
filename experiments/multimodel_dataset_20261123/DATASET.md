# Phase 2 — Historical Dataset Preparation

Isolated experiment directory. No production, API, database, or artifact changes.

## Verified dataset summary

| Field | Value |
|---|---|
| Days | 356 |
| Date coverage | 2025-12-01 through 2026-11-21 |
| Bookings | 519 |
| Guests | 6,669 |
| Metadata `input_sha256` | `0af4b7c020769453ef748f2302ca8176c9ced891e77d86db4f33ec5eb2a77cf1` |
| Reconstruction source | `baseline_history` inside `bookings_model.joblib` and `guests_model.joblib` |

**Identity limitation:** The reconstructed `daily_demand.csv` matches the verified metadata totals and continuous calendar embedded in the trusted artifacts. It is **not** proven byte-for-byte identical to the original training export file. The original export file is not present in this repository; only its SHA-256 fingerprint is recorded in `forecast_models_metadata.json`.

## Target definitions

- `demand_date` = reservation arrival / check-in date
- `bookings` = count of eligible arriving reservations that day
- `guests` = sum of arriving guests that day
- Zero-demand days are retained
- High-demand days are retained

## Feature definitions

| Feature group | Columns | Leakage rule |
|---|---|---|
| Calendar | `day_of_week`, `is_weekend`, `month`, `dow_sin`, `dow_cos` | May use the forecast date |
| Lags | `{bookings,guests}_lag_{1,7,14,28}` | Source date must be strictly before the feature `as_of` date |
| Rolling means | `{bookings,guests}_roll_mean_{7,14,28}` | Mean of the `w` days ending the day before `as_of` |
| Trend | `{bookings,guests}_trend_14` | Recent 14-day mean / preceding 14-day mean, both before `as_of` |

### One-step supervised table

`supervised_features.csv` predicts same-day bookings/guests using only prior days. Warm-up: first 28 days omitted.

### Static seven-day horizon table

`horizon_features_static.csv` builds features for each Monday cutoff’s next seven days using only history before that cutoff. Short lags that would require intra-horizon actuals are left null. Phase 3 may fill them recursively with predictions, never with future actuals.

### ARIMA inputs

`arima_bookings_series.csv` and `arima_guests_series.csv` are univariate arrival series indexed by date.

## Evaluation periods

- Development: 24 weeks, cutoffs 2026-04-06 through 2026-09-14 (weeks ending 2026-09-20)
- Reserved: 8 weeks, cutoffs 2026-09-21 through 2026-11-09 (weeks ending 2026-11-15)
- Reserved period was previously inspected; it is not untouched independent validation

## 519 eligible arrivals vs 595 total reservation records

Existing evidence does **not** contain a reconciled 595-row ledger inside this repository or inside the reconstructed daily series.

What is established:

1. The forecasting snapshot counts **519 eligible arrival bookings** on the continuous daily aggregate used for training/experiments.
2. Eligibility in `predictionTrainingService.js` keeps arrivals with statuses `checked-in`, `checked-out`, or `completed`, and may optionally include past `confirmed` check-ins. Future/pending/cancelled and null check-in dates are outside that arrival demand definition.
3. The Phase 1 product note refers to **491 imported** historical reservations as legitimate pre-ReserVision operations records. That figure is about import provenance, not a direct subtraction from 595.
4. No local artifact enumerates a audited total of **595** reservations or maps the **76** (=595−519) difference to cancelled, pending, future, excluded, or non-arrival rows.

**Unresolved limitation:** Without the original reservation-level audit table or a read-only database count (forbidden in Phase 2), this experiment cannot prove why 595 previously audited reservation records differ from 519 eligible arrivals. The difference must **not** be interpreted as “invalid imports.” Valid imported demand that passed eligibility remains inside the 519/6,669 aggregate.

## Files in this directory

| Path | Role |
|---|---|
| `periods.py` | Snapshot constants and week cutoffs |
| `features.py` | Leakage-safe feature builders |
| `reconstruct.py` | Artifact reconstruction + hash guard |
| `prepare.py` | Writes `prepared/` outputs |
| `test_dataset.py` | Focused integrity/leakage tests |
| `DATASET.md` | This document |
| `prepared/` | Generated dataset and manifests |
