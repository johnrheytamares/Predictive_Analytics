# ReserVision — Final Offline ML Evaluation Freeze

**Package:** `experiments/final_ml_evaluation_freeze_20261123`  
**Status:** Frozen consolidation of verified Phases 2–5 only  
**Graphs / Chapter IV presentation:** not included (numerical package only)  
**Retraining / production changes:** none

This document consolidates reproducible offline evaluation results for arrival-day **bookings** and **guests**. All figures are taken from the Phase 2–5 experiment directories. No new models were trained for this package.

---

## 1. Scope and sources

| Phase | Directory | Role |
|-------|-----------|------|
| 2 | `experiments/multimodel_dataset_20261123` | Leakage-safe 356-day dataset and evaluation frame |
| 3 | `experiments/multimodel_compare_20261123` | Raw model comparison (baseline, Ridge, Random Forest) |
| 3b | `…/results/arima` | Raw ARIMA comparison (isolated venv) |
| 5 | `experiments/postprocess_phase5_20261123` | Operational post-processing + robust median guests recipe |

**Not in scope of this freeze:** Phase 6 production integration, deployment, database writes, Chapter IV figures.

---

## 2. Evaluation protocol (fixed)

| Item | Definition |
|------|------------|
| Dataset | 356 continuous days, 2025-12-01 → 2026-11-21 |
| Totals | 519 bookings, 6,669 guests |
| Daily series SHA-256 | `7017de3fb24a1105e3a09872526367aa40ce1b05384e28065de99a1464ca1cf7` |
| Origins | 32 Monday cutoffs; 7-day arrival horizons |
| Development | 24 weeks, 2026-04-06 → 2026-09-20 (**primary selection period**) |
| Reserved | 8 weeks, 2026-09-21 → 2026-11-15 (**previously inspected**; confirmatory only) |
| WAPE | \(100 \times \sum\|e\| / \sum \text{actual}\). Weekly WAPE uses week totals first. **WAPE is error, not accuracy.** |
| Target | 10–20% weekly WAPE — **not achieved** by any configuration in this study |

---

## 3. Distinguishing result layers

| Layer | Meaning |
|-------|---------|
| **Raw model performance** | Predictions before production post-processing |
| **Operational post-processing performance** | After Variant A (production rules) or Variant B (no guest floor) |
| **Development-period results** | Used for model / recipe selection |
| **Reserved-period results** | Confirmatory only; period was already examined earlier → selection-bias risk |

---

## 4. Robust median is not a newly trained ML model

**Robust median guests** is a **deterministic forecasting recipe**:

- **Bookings:** same four-week equal-weight weekday mean as the production baseline  
- **Guests:** median of the four most recent matching-weekday guest counts **including zeros**

It does **not** introduce a newly fitted Ridge, Random Forest, ARIMA, or Prophet parameter set. It is an alternative historical aggregation rule evaluated offline under the same origins and post-processing rules as other pairs.

---

## 5. Operational bookings WAPE improvement — cause

On the **development** period, raw bookings forecasts for production baseline and robust-median pair are **identical** (weekly WAPE **28.76**).

After Variant A post-processing:

| Configuration | Raw bookings weekly WAPE | Operational bookings weekly WAPE | Booking-minimum activations |
|---------------|--------------------------|----------------------------------|-----------------------------|
| Production baseline + A | 28.76 | **30.11** | **21** |
| Robust median guests + A | 28.76 | **27.60** | **8** |

The operational bookings WAPE improvement comes from **fewer booking-minimum activations** (the rule that forces bookings ≥ 1 when predicted guests > 0), **not** from a better raw bookings forecast.

Supporting table: `tables/bookings_wape_decomposition.csv`.

---

## 6. Raw model performance (development)

Primary weekly WAPE (development). Lower is better. Full metrics: `tables/raw_development.csv`, `tables/raw_development_primary.csv`.

### Bookings (weekly WAPE %)

| Model | Weekly WAPE | Daily WAPE |
|-------|-------------|------------|
| ARIMA(2,1,1) | 28.59 | 68.11 |
| Four-week weekday baseline | 28.76 | 61.02 |
| Ridge (α=100) | 28.95 | 61.57 |
| ARIMA(1,1,1) | 29.12 | 69.00 |
| Random Forest | 30.37 | 59.51 |

### Guests (weekly WAPE %)

| Model | Weekly WAPE | Daily WAPE |
|-------|-------------|------------|
| Four-week weekday baseline | **33.77** | 81.38 |
| Ridge (α=100) | 38.78 | 91.10 |
| ARIMA(1,1,1) | 40.25 | 107.34 |
| Random Forest | 40.49 | 90.00 |
| ARIMA(2,1,1) | 41.16 | 108.95 |

**Interpretation (raw):**

- No trained model clearly beat the four-week weekday baseline on **guests**.  
- ARIMA(2,1,1) had a tiny edge on **bookings weekly** WAPE but worse daily WAPE than baseline.  
- Complex models did not justify replacing the weekday baseline on raw guest skill.

---

## 7. Raw reserved-period results (confirmatory)

Reserved weeks were previously inspected; treat as confirmatory, not untouched holdout. Full table: `tables/raw_reserved.csv`.

### Bookings (weekly WAPE %, reserved)

| Model | Weekly WAPE | Daily WAPE |
|-------|-------------|------------|
| Four-week weekday baseline | 54.03 | 84.68 |
| ARIMA(2,1,1) | 55.82 | 88.33 |
| ARIMA(1,1,1) | 56.21 | 88.80 |
| Random Forest | 70.22 | 98.09 |

### Guests (weekly WAPE %, reserved)

| Model | Weekly WAPE | Daily WAPE |
|-------|-------------|------------|
| Four-week weekday baseline | 83.55 | 149.73 |
| ARIMA(1,1,1) | 96.33 | 161.38 |
| ARIMA(2,1,1) | 97.03 | 162.53 |
| Random Forest | 103.63 | 158.21 |

Ridge rows for reserved were not re-exported in the Phase 3 reserved comparison file used here; use development Ridge figures for trained linear comparison. Exact floats for Chapter IV must come from the CSVs.

---

## 8. Operational post-processing performance

Variant A = production rules (round/clamp; bookings ≥ 1 if guests > 0; guests ≥ max(guests, bookings × typical party size ≈ 7.0, bookings)).  
Variant B = same without the party-size guest floor.

Full tables: `tables/operational_development.csv`, `tables/operational_reserved.csv`.

### Development — selected comparison (Variant A)

| Role | Pair | Bookings weekly WAPE | Guests weekly WAPE |
|------|------|----------------------|--------------------|
| **Current production baseline** | equal-weight baseline + A | 30.11 | 35.20 |
| **Best offline candidate** | robust median guests + A | **27.60** | **32.09** |

### Reserved — confirmatory (Variant A)

| Role | Bookings weekly WAPE | Guests weekly WAPE |
|------|----------------------|--------------------|
| Current production baseline + A | 70.97 | 94.18 |
| Best offline candidate + A | 67.74 | 80.00 |

Reserved errors are much larger (sparse late-period demand). They **do not** establish that the 10–20% weekly WAPE target is met.

---

## 9. Production baseline vs best offline candidate

| | Current production baseline | Best offline candidate |
|--|----------------------------|------------------------|
| **Role** | Live serving reference | Offline study winner only |
| **Bookings forecast** | Four-week weekday mean | Same four-week weekday mean |
| **Guests forecast** | Four-week weekday mean | Median of four matching weekdays (incl. zeros) |
| **Post-process** | Variant A (production) | Variant A (evaluated offline) |
| **Model type** | Deterministic baseline | Deterministic recipe (**not** a new trained ML model) |
| **Dev operational WAPE** | 30.11 / 35.20 | 27.60 / 32.09 |
| **Production status** | **In production** (`weekday_baseline` artifacts) | **Not integrated**; offline candidate only |

If production were changed **only** in post-processing while keeping live equal-weight baseline forecasts, Phase 5 evidence favored **Variant B** (no guest floor) for that pair—not Variant A. That is separate from the robust-median offline candidate above.

---

## 10. Numerical package for later Chapter IV

All machine-readable tables live under `tables/`:

| File | Contents |
|------|----------|
| `raw_development.csv` | All consolidated raw methods, development |
| `raw_development_primary.csv` | Primary models for Chapter IV raw comparison |
| `raw_reserved.csv` | Raw reserved (confirmatory) |
| `operational_development.csv` | All operational pairs × variants, development |
| `operational_reserved.csv` | Operational reserved (confirmatory) |
| `production_vs_candidate.csv` | Production baseline vs robust candidate, both periods |
| `bookings_wape_decomposition.csv` | Raw vs operational bookings + booking-min counts |
| `dataset_provenance.json` | Dataset SHA, periods, source experiments |

Also: `protocol.json`, `serving_integrity.json`.

**No graphs** are generated in this freeze.

---

## 11. Evaluation limitations (must retain)

1. **356-day historical dataset** — limited calendar coverage; seasonal and year-over-year effects are under-observed.  
2. **Sparse and irregular resort demand** — many low/zero days; high relative error even when absolute error is small.  
3. **Previously inspected reserved period** — reserved weeks were examined during earlier adaptive work; they are **not** a clean untouched final test.  
4. **Selection-bias risk** — robust median guests was identified after prior inspection that included reserved outcomes. Development numbers remain the selection criterion, but optimism bias is possible.  
5. **Unresolved 519 vs 595 reservation reconciliation** — daily demand totals 519 bookings from reconstructed history; an alternate 595 figure remains unresolved without a reservation-level ledger / DB audit (not performed in this offline freeze).  
6. **10–20% weekly WAPE target not achieved** — neither raw nor operational configurations reached the stated accuracy goal on development or reserved periods.  
7. **Reconstruction caveat** — Phase 2 series content-matches trusted `baseline_history` in joblib artifacts; original export bytes (`0af4b7c0…`) were not re-verified byte-identical in this freeze because that export file is absent from the working tree.  
8. **WAPE is error** — lower is better; do not report WAPE as “accuracy %”.

---

## 12. Serving / production integrity

At freeze creation, SHA-256 hashes of production serving files were recorded in `serving_integrity.json`. Comparison before/after writing this package:

**Serving hashes unchanged: true**

Checked files include:

- `bookings_model.joblib`
- `guests_model.joblib`
- `forecast_models_metadata.json`
- `bookings_model_metadata.json`
- `api.py`
- `predict_real.py`
- `train_real_model.py`

No database access, no retraining, no production integration, and no deployment were performed for this package.

---

## 13. Freeze decision

| Decision | Value |
|----------|--------|
| Offline evaluation | **Frozen** as of this package |
| Production baseline | Remain **four-week weekday baseline + Variant A** |
| Best offline candidate | **Robust median guests + Variant A** (deterministic recipe) |
| Immediate production replace | **Not recommended** solely from this offline package (selection-bias + target miss + unresolved 519/595) |
| Next work requiring new approval | Chapter IV graphs/tables presentation; any production integration (former Phase 6) |

---

## 14. Citation note for Chapter IV

When building tables later, cite:

- Dataset SHA `7017de3…`  
- Development as selection period  
- Reserved as previously inspected confirmatory period  
- Robust median as deterministic recipe  
- Operational bookings WAPE gain as booking-minimum activation effect  
- Explicit non-achievement of 10–20% weekly WAPE  

End of freeze package.
