# ReserVision v3.2 RC — guests_per_booking_sumratio

Isolated release candidate. **Production v3.1 untouched** (rollback preserved).

## Configuration

- Bookings: Two-Stage (identical v3.1 artifacts / forward parameters)
- Guests: past-84-day Σguests/Σbookings × operational bookings
- Postprocess: floor_round consistency only (no party-size floor, no holiday multipliers)
- Demographics: period-preserving allocation

Booking integers match v3.1 because `floor_round` consistency still consults the rolled-back robust-median guest path as an **auxiliary** input. Served guest **points and intervals** are sum-ratio only (robust-median guest totals are not returned).

## 7-day weekly metrics (28 origins, frozen Two-Stage bookings)

| Target | MAE | RMSE | Bias | WAPE % |
|---|---:|---:|---:|---:|
| Guests (sumratio) | 29.18 | 40.94 | -1.82 | 34.11 |
| Bookings (locked) | 2.54 | 3.87 | -0.46 | 29.96 |

Daily guest MAE/RMSE/Bias/WAPE: 10.872 / 18.338 / -0.260 / 88.98%

## Uncertainty intervals (sumratio-specific)

Method: past-only party-size ratio quantiles q10–q90 × bookings (not robust-median intervals).
Daily coverage on evaluation days with demand signal: **51.4%** (n=177).

## Sample 2027-01-01 → 2027-02-08

- V3.2 RC: **12 bookings / 108 guests**
- V3.1 production path: **12 bookings / 18 guests**
- Booking vectors identical across v3.1 and v3.2 for this window.

## Horizons (pooled horizon-total)

| Horizon | Origins | Guest MAE | Guest RMSE | Guest Bias | Guest WAPE % | Booking WAPE % | Interval coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 28 | 29.18 | 40.94 | −1.82 | 34.11 | 29.96 | 82.1% (week totals) |
| 30 | 25 | 80.92 | 102.38 | +35.56 | 23.01 | 24.11 | 88.0% |
| 90 | 17 | 315.00 | 377.62 | +315.00 | 30.98 | 20.29 | 100% |

Daily interval coverage on 7-day evaluation days: **51.4%** (ratio q10–q90 bands are intentionally narrower than unused robust-median margins).

## Rollback

Keep serving `production_v31` → `reserVision_v3_1_two_stage_rc_20261124`.
This RC remains `production_approved=false` / `shadow_serving_only=true`.
