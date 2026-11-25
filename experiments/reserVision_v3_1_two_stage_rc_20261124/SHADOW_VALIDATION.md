# Prospective shadow validation — ReserVision Version 3.1

**Status:** Procedure only. Not started. No production serving of two-stage or floor_round.

## Frozen configuration (do not retune)

| Target | Model / mode |
|--------|----------------|
| Bookings | Original **two_stage** (`intermittent_bookings_20261124`) |
| Guests | **robust_median_guests** (cutoff-anchored; preserved) |
| Post-process | **floor_round** (`v3_1_postprocessing_optimization_20261124`) |

## Retrospective baseline (reference only)

Do not treat these as prospective proof:

- Raw weekly WAPE **32.72%** (selection protocol).
- floor_round weekly WAPE **29.96%**, MAE **2.536**, RMSE **3.868**, Bias **−0.464**.
- Nested past-only ops selection **35.02%** (evidence; not serving mode).

Prospective weeks must be scored separately after they complete.

## Cadence

1. Each Monday (or each new complete arrival week), freeze history through Sunday.
2. Generate raw two-stage bookings for the next 7 days using nested past-only tune with the **original** intermittent grid only (no new optimization grids).
3. Generate Robust Median guests for the same dates (cutoff-anchored).
4. Apply **floor_round** for the operational shadow track; also retain raw floats for continuity with the 32.72% protocol.
5. After the week completes, score actual arrivals.

## Metrics to log every week

- Weekly and daily WAPE / MAE / RMSE / Bias for **raw** bookings (experimental continuity).
- Same metrics after **floor_round** (operational shadow — primary serving candidate).
- Guests weekly/daily WAPE under Robust Median + floor_round.
- Consistency violations (should be zero after floor_round): `guests < bookings`, or `guests > 0` with `bookings == 0`.
- Zero-day false demand and positive-day underprediction.

## Decision gates (before any serving discussion)

- At least several new complete weeks **not** used in the retrospective 28-origin study.
- Raw two-stage must not regress badly versus weekday baseline on those new weeks across **all four** weekly metrics (not WAPE alone).
- Operational floor_round metrics reviewed explicitly; do not claim the frozen 29.96% for unseen weeks until measured.
- Guests remain Robust Median; no guest-model or ops-mode swap during this shadow.
- No further model or postprocess optimization during shadow.

## Out of scope

Production artifact replacement, Dokploy deploy, Node/Vue activation, database writes, further two-stage or postprocess optimization.
