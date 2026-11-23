# Phase 4: saved raw forecast error analysis

No models fitted, no dependencies installed, no production edits, database access, artifact replacement, deployment, or operational post-processing. Phase 5 was not run.

## Integrity and scope

Validated 3,920 saved daily predictions across all available Phase 3 configurations. Selected comparison contains 1,680 daily rows. Dates, cutoff, horizon offsets 0-6, actual values, saved errors, complete available periods, and weekly aggregation match. Recomputed 112 saved Phase 3 metric values match within 1e-7. Input and serving source/artifact hashes unchanged. Seven tests passed.

Development: April 6-September 20, 2026 (24 weeks, 279 bookings, 3,023 guests). Previously examined reserved: September 21-November 15 (8 weeks, 31 bookings, 275 guests). Reserved is not independent validation.

Ridge family development choices: alpha100 bookings; alpha10 guests. Phase 3 selected the baseline overall and saved no reserved Ridge predictions. Reserved Ridge comparison is unavailable; none generated. ARIMA choices remain Phase 3's (2,1,1) bookings and (1,1,1) guests. No new configuration search performed.

## Numerical results

WAPE in percent; bias is mean signed predicted-minus-actual error per week.

| Period | Target | Method | Weekly WAPE | Daily WAPE | Weekly MAE | Weekly RMSE | Weekly bias |
|---|---|---|---:|---:|---:|---:|---:|
| Development | Bookings | Baseline | 28.76 | 61.02 | 3.34 | 4.27 | +0.99 |
| Development | Bookings | Ridge alpha100 | 28.95 | 61.57 | 3.37 | 4.43 | +1.30 |
| Development | Bookings | Random Forest | 30.37 | 59.51 | 3.53 | 4.39 | +1.06 |
| Development | Bookings | ARIMA(2,1,1) | 28.59 | 68.11 | 3.32 | 4.34 | +1.44 |
| Development | Guests | Baseline | 33.77 | 81.38 | 42.54 | 58.85 | +6.54 |
| Development | Guests | Ridge alpha10 | 38.14 | 94.34 | 48.04 | 67.02 | +2.15 |
| Development | Guests | Random Forest | 40.49 | 90.00 | 51.00 | 65.08 | +22.77 |
| Development | Guests | ARIMA(1,1,1) | 40.25 | 107.34 | 50.70 | 65.37 | +31.31 |
| Reserved | Bookings | Baseline | 54.03 | 84.68 | 2.09 | 2.68 | +1.47 |
| Reserved | Bookings | Random Forest | 70.22 | 98.09 | 2.72 | 3.21 | +1.92 |
| Reserved | Bookings | ARIMA(2,1,1) | 55.82 | 88.33 | 2.16 | 2.70 | +1.61 |
| Reserved | Guests | Baseline | 83.55 | 149.73 | 28.72 | 33.95 | +16.16 |
| Reserved | Guests | Random Forest | 103.63 | 158.21 | 35.62 | 41.05 | +25.44 |
| Reserved | Guests | ARIMA(1,1,1) | 96.33 | 161.38 | 33.11 | 38.79 | +27.69 |

All daily MAE, RMSE, bias, over/under counts and amounts are in results/metrics.csv. Demand and calendar strata for every compared configuration are in results/daily_strata.csv; all weekly predictions/errors and top ten rankings are saved separately.

## Why the ARIMA bookings advantage does not generalize

Development weekly absolute error is 79.7619 ARIMA versus 80.25 baseline: a reduction of only 0.4881 bookings across 24 weeks, or 0.175 percentage points WAPE. Daily absolute error increases from 170.25 to 190.0213. Within-week error cancellation increases from 90 to 110.2594. Weekly aggregation hides day-level timing errors; it is not a corresponding improvement in daily forecasting. Weekly RMSE and positive bias are also worse.

ARIMA development zero-booking-day MAE is 1.84 versus baseline 1.10. On high-booking days (>=3), MAE is 1.85 versus 1.37. Development weekday/weekend booking bias is +0.67/-0.96 ARIMA versus +0.11/+0.22 baseline. This supports a loss of day-of-week allocation accuracy. Source models.py uses nonseasonal ARIMA orders; it does not establish a causal holiday explanation.

Reserved ARIMA overpredicts 49/56 booking days and 7/8 weeks, versus baseline 28/56 days and 5/8 weeks (17 daily ties and one weekly tie). Its weekly bookings WAPE rises to 55.82 versus 54.03 baseline. This small development edge is not evidence for promotion.

## Demand strata and calendar findings

Descriptive daily strata (not model inputs): zero=0; low-positive bookings=1 and guests=1-10; high bookings>=3 and guests>=30; middle covers the remainder. Weekly low demand <=5 bookings or <=35 guests; high >=20 bookings or >=150 guests. No observations excluded or thresholds tuned for promotion.

Development baseline booking MAE: zero1.10, low0.67, high1.37. Guest MAE: zero9.32, low7.97, high32.58. High guest days have mean bias -29.00, while zero guest days have +9.32. Underestimation of spikes and overestimation of quiet dates coexist.

Development baseline weekday/weekend MAE is 0.98/1.10 bookings and 10.59/24.77 guests. Guest actuals total 1,243 weekdays and 1,780 weekends across 120 and 48 days respectively. Larger weekend errors coincide with larger actual demand; these aggregate comparisons do not prove a holiday effect.

Reserved baseline weekday/weekend MAE is 0.49/0.42 bookings and 7.19/7.75 guests. Reserved zero-day false-positive means are 0.63 bookings and 5.94 guests. WAPE on all-zero subsets is undefined and stored as null; MAE and predicted totals retain those errors. Reserved has 29 zero days and only four high-guest days, limiting inference.

## Error concentration

Baseline top three weekly errors account for 31.78% of development booking error and 36.31% of development guest error; reserved shares are 70.15% and 61.15%. Reserved low-demand weeks account for 92.54% of booking error and 63.00% of guest error. Development high-guest weeks account for 45.00% of guest error. Errors are not solely one type of week.

Largest baseline examples (week starts; predicted vs actual):

- Development bookings June8:17.25 vs8; May25:16.75 vs25; June15:15 vs7.
- Development guests April20:160.25 vs310; May4:225.75 vs104; June22:157.25 vs58.
- Reserved bookings October12:6.75 vs2; October26:4.5 vs0; October19:5.5 vs3.
- Reserved guests October12:69.5 vs4; October19:51.75 vs8; October26:31.25 vs0.

Saved numerical outputs support these observed patterns. They do not identify holidays, source-record completeness, causal resort changes, or original training information availability independently. Imported records are retained in actuals and are not blamed for errors.

## Decision

Retain the baseline as reference. ARIMA's development weekly bookings edge is too small and does not persist in daily accuracy or reserved results; Random Forest's development daily bookings gain also does not persist in reserved results. Ridge reserved performance cannot be assessed from saved predictions. No below20% claim is supported. Stop after Phase 4; Phase 5 operational processing requires separate approval.

Run tests: `python -B -m unittest discover -s experiments/phase4_errors_20261123 -p test_analysis.py` from the Predictive_Analytics root. Analyzer refuses to overwrite existing results. supplementary_checks.json records final complete-period alignment, metric reproduction, hashes, and tests.
