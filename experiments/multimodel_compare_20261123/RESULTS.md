# Phase 3 — Multimodel Comparison (Raw Forecasts)

## Dataset verification

- Phase 2 `daily_demand.csv` SHA-256: `7017de3fb24a1105e3a09872526367aa40ce1b05384e28065de99a1464ca1cf7`
- Totals: 356 days, 519 bookings, 6,669 guests
- Serving artifact hashes unchanged after the run

## WAPE definition

`WAPE = 100 * sum(|predicted - actual|) / sum(actual)`

- Weekly: aggregate each Monday–Sunday week first, then score week totals
- Daily: score every daily point in the period
- WAPE is an error measure, not an accuracy percentage
- Raw forecasts only; no guest-per-booking floor

## Development selection (24 weeks)

| Target | Selected method | Weekly WAPE | Daily WAPE | Weekly MAE |
|---|---|---:|---:|---:|
| Bookings | weekday_baseline | 28.76 | 61.02 | 3.34 |
| Guests | weekday_baseline | 33.77 | 81.38 | 42.54 |

Closest Ridge challenger for bookings: `ridge_alpha_100` (weekly WAPE 28.95). Random Forest did not beat the baseline on weekly WAPE for either target.

## ARIMA status

Completed in an isolated `.venv` (statsmodels 0.14.4). See `results/arima/ARIMA_RESULTS.md`.

- Selected: bookings `arima_211` (dev weekly WAPE 28.59), guests `arima_111` (40.25)
- Guests remain worse than weekday baseline (33.77)
- Bookings weekly WAPE edges baseline by ~0.17 points but daily WAPE is worse

## Reserved confirmatory check (8 previously inspected weeks)

| Method | Bookings weekly WAPE | Guests weekly WAPE |
|---|---:|---:|
| weekday_baseline | 54.03 | 83.55 |
| random_forest | 70.22 | 103.63 |

Reserved demand is sparse; high WAPE here is expected and not an independent validation claim.
