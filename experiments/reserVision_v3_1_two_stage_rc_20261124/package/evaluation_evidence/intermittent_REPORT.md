# Bookings intermittent-demand experiment

Two-stage probability-times-positive-size forecasting is the best tested candidate on the shared rolling evaluation. It is not ready for production: it loses on the original sparse 28-day holdout, and the 10–20% WAPE target was not achieved.

## Data and cause

Used the existing continuous daily historical snapshot, 2025-12-01–2026-11-21: 356 days, 519 bookings. Source: `../multimodel_dataset_20261123/prepared/data/daily_demand.csv`, reconstructed previously from model histories. Existing provenance records say this matches a user-confirmed historical server aggregate; the source reservation register was not independently audited in this experiment. No synthetic CSV, database access, imputation, zero removal, or date edits were used.

The saved selected-model WAPE of 203.125% reproduced exactly on 2026-10-25–2026-11-21. There were eight bookings, 14.25 predicted bookings, and 16.25 total absolute error: 16.25 / 8 × 100 = 203.125%. Errors on zero-demand days contribute 11.25 (69.23%) of absolute error. Positive-day errors contribute 5.0. Signed mean bias is +0.223 bookings/day. The baseline continues forecasting demand through a pronounced recent decline; the small denominator magnifies this mismatch.

| Pattern | Earlier 328 days | Last 28 days | Entire history |
|---|---:|---:|---:|
| Zero-demand days | 75 (22.86%) | 22 (78.57%) | 97 (27.25%) |
| Days per positive observation (ADI) | 1.30 | 4.67 | 1.37 |
| Mean positive-day bookings | 2.02 | 1.33 | 2.00 |
| Positive-size CV squared | 0.425 | 0.125 | 0.426 |
| Longest consecutive zero run | 6 | 8 | 10 |

Both occurrence frequency and positive demand size fell. Calendar-only predictors cannot establish whether this is a persistent demand shift or an operational recording/coverage change. Zeros are preserved as recorded observations; their business provenance needs a separate read-only review.

## Protocol

Reused `evaluate_forecasts.load_history`, `metrics`, and `weekday_forecast`. All methods use identical windows and raw nonnegative expected counts; no integer rounding or cross-target adjustments.

- Original fixed-origin holdout: train through 2026-10-24, forecast 28 days without observing any holdout actuals.
- Rolling evaluation: 28 non-overlapping seven-day horizons, 2026-05-04–2026-11-15, expanding training history. The first four origins in the previous 32-week pipeline were omitted for every method to permit four inner folds with at least 120 training days.
- At each outer origin, tune independently on four preceding non-overlapping inner horizons of the same length. Selection minimizes pooled daily MAE, equivalent to WAPE ranking when the actual denominator is shared. Outer labels never enter tuning. Earlier outer outcomes become available only at subsequent chronological origins.
- Optimized weekday: 2/4/8/12-week history, equal or 0.7 exponential weekday weights, 0 or 0.5 shrinkage to the recent overall mean.
- Two-stage hurdle: empirical positive-demand probability over 14/28/56 days, weekday/global shrinkage 0/0.5/1, multiplied by conditional positive size over 28/84/168 days. Probability estimation includes every zero day; conditional size appropriately uses positive observations only.
- TSB: positive-size alpha 0.05/0.1/0.3/0.5; probability beta 0.02/0.05/0.1/0.3, with beta <= alpha. Probability updates every observed day, size only on positive observations. Future unobserved days are not treated as known zeros. Method reference: [SAP TSB documentation](https://help.sap.com/docs/SAP_HANA_PLATFORM/319d36de4fd64ac3afbf91b1fb3ce8de/croston-tsb-method-8203847.html).
- Weekly scores aggregate seven forecast days within an origin; in the 28-day holdout these are four Sunday–Saturday blocks, not rolling weekly refits. Pooled WAPE uses summed absolute errors divided by summed actuals; all-zero individual periods remain present and their standalone WAPE is undefined.
- Periods have been inspected in prior experiments. This is retrospective development evidence, not untouched validation. Do not pool the overlapping holdout and rolling protocols.

## Results

Numbers in each metric column are WAPE %, MAE, RMSE, signed Bias (prediction minus actual). Daily units are bookings/day; weekly units are bookings/week.

| Rolling candidate | Daily metrics | Weekly metrics |
|---|---|---|
| Current weekday | 69.62 / 0.842 / 1.164 / +0.214 | 37.55 / 3.179 / 4.108 / +1.500 |
| Optimized weekday | 67.11 / 0.812 / 1.169 / +0.171 | 36.50 / 3.089 / 4.195 / +1.197 |
| Two-stage | **64.94 / 0.785 / 1.142 / +0.134** | **32.72 / 2.770 / 3.909 / +0.940** |
| TSB | 74.83 / 0.905 / 1.285 / +0.243 | 39.63 / 3.354 / 4.598 / +1.702 |
| Always-zero diagnostic | 100.00 / 1.209 / 1.800 / -1.209 | 100.00 / 8.464 / 9.948 / -8.464 |

| Original 28-day holdout candidate | Daily metrics | Weekly metrics |
|---|---|---|
| Current weekday | 203.13 / 0.580 / 0.972 / +0.223 | 103.13 / 2.063 / 2.589 / +1.563 |
| Optimized weekday | 233.48 / 0.667 / 0.715 / +0.393 | 143.75 / 2.875 / 3.326 / +2.750 |
| Two-stage | 234.65 / 0.670 / 0.703 / +0.393 | 143.75 / 2.875 / 3.326 / +2.750 |
| TSB | 240.98 / 0.689 / 0.723 / +0.419 | 148.36 / 2.967 / 3.480 / +2.934 |
| Always-zero diagnostic | 100.00 / 0.286 / 0.655 / -0.286 | 100.00 / 2.000 / 2.739 / -2.000 |

Two-stage improves rolling daily WAPE by 4.68 percentage points (6.73% relative) and weekly WAPE by 4.83 points (12.86% relative). It improves all four rolling metrics, but cannot be recommended as a production replacement given the original holdout regression. The always-zero diagnostic illustrates sparse WAPE incentives and is not a useful automatic serving recommendation.

## Recommended next implementation step

Keep two-stage as an experimental challenger. Freeze its tuning protocol, then run prospective shadow forecasts over new complete weeks, comparing both seven-day and 28-day horizons against the current baseline. First verify read-only that recent recorded zeros represent genuine demand rather than incomplete booking capture. Track zero-day false demand and positive-day underprediction alongside daily/weekly metrics. Only consider serving integration after the challenger consistently beats the baseline on new data. No deployment is recommended from these results.

## Artifacts and verification

`results/results.json` records provenance, input SHA-256, every selected configuration, metrics, profiles, error decomposition and before/after serving-file hashes. `results/daily_predictions.csv` contains all predictions; `results/comparison.csv` contains flat metric summaries. Production Python, model and metadata hashes were unchanged. No Node.js, Vue, production database or Dokploy changes were made.

Run `python -B experiments/intermittent_bookings_20261124/test_experiment.py` for hand-calculated TSB, zero-preserving hurdle, temporal boundary, undefined zero-total WAPE, dataset and exact original baseline checks. Reproduce in a new isolated results directory: the experiment refuses to overwrite existing results.
