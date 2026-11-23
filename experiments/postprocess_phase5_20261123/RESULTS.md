# Phase 5 — Operational Post-Processing Evaluation

## Verification

- Phase 2 daily SHA-256: `7017de3fb24a1105e3a09872526367aa40ce1b05384e28065de99a1464ca1cf7`
- Phase 3 / ARIMA saved prediction actuals match the demand series
- Prediction files were read-only inputs; not rewritten
- Serving artifact hashes unchanged
- Ridge reserved predictions were not invented or used

## Variants

| Code | Meaning |
|---|---|
| A | Production rules: round/clamp, booking≥1 when guests>0, party-size floor 7.0, guests≥bookings |
| B | Same without party-size floor |
| C | Forecast pair `robust_median_guests` regenerated from the verified series (prior adaptive recipe) |

## Development operational comparison (weekly WAPE)

| Pair | Variant | Bookings | Guests | Avg | Floor days changed | Floor guest +total |
|---|---|---:|---:|---:|---:|---:|
| baseline_baseline | A | 30.11 | 35.20 | 32.65 | 71 | +251 |
| baseline_baseline | B | 30.11 | 33.84 | 31.97 | — | — |
| arima211_baseline | A | 32.26 | 40.26 | 36.26 | 82 | +555 |
| arima211_baseline | B | 32.26 | 34.01 | 33.13 | — | — |
| rf_baseline | A | 30.11 | 37.68 | 33.89 | 75 | +354 |
| rf_baseline | B | 30.11 | 33.91 | 32.00 | — | — |
| **robust_median_guests** | **A** | **27.60** | **32.09** | **29.84** | 81 | +337 |
| robust_median_guests | B | 27.60 | 34.90 | 31.25 | — | — |

## Reserved confirmatory (selected vs production)

| Configuration | Bookings weekly WAPE | Guests weekly WAPE |
|---|---:|---:|
| Production reference (`baseline_baseline` + A) | 70.97 | 94.18 |
| Selected (`robust_median_guests` + A) | 67.74 | 80.00 |
| Same pair without floor (`robust_median_guests` + B) | 67.74 | 65.45 |

Reserved was not used for selection.

## Guest-floor impact

On the current production pair (`baseline_baseline`), the party-size floor:

- does not change bookings metrics
- raises guests weekly WAPE (35.20 vs 33.84 development; 94.18 vs 84.73 reserved)
- added +251 guest-count units across 71 development days

So the floor is a post-processing penalty for the equal-weight baseline pair.

For `robust_median_guests`, the floor helps development guests weekly WAPE because that raw guest model under-forecasts; the floor offsets some of that bias.

## Selection

Development complete-pair rule selected:

**`robust_median_guests` + variant A**

Improvement sources:

1. **Model/recipe change:** median-of-four weekday guests instead of equal-weight 4-week guest mean  
2. **Post-processing:** production floor retained for this pair because it improved development guests weekly WAPE versus B

## Recommendation

**Candidate for later integration review:** `robust_median_guests` with production post-process A.

If only a post-processing change is allowed on the current live weekday baseline pair, prefer **variant B (no guest floor)** over A. That improves guests without changing bookings.

Do **not** replace production yet without an explicit integration phase. Remaining limits:

- Sparse reserved demand still yields high absolute WAPE
- 10–20% weekly WAPE target is not achieved
- Config C was regenerated from the verified series (deterministic), not loaded from a frozen Phase 3 ML file
- No Ridge reserved operational comparison

## Tests

`python -m unittest test_postprocess.py -v` → **6/6 passed**
