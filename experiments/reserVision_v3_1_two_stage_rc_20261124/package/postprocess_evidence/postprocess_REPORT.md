# V3.1 operational post-processing optimization

Frozen Two-Stage bookings + Robust Median guests unchanged. No retraining. Production artifacts untouched.

**Best operationally valid fixed policy:** `floor_round`  
**Rolling weekly bookings WAPE:** **29.96%** (vs raw 32.72%, vs Variant A 38.40%)

## What Variant A does to error

On the V3.1 pair, **bookings weekly WAPE moves 32.72 → 38.40** under Variant A.

Ablation on **bookings** weekly WAPE (same 28 origins):

| Ops rule set | Weekly WAPE | Weekly MAE | Weekly RMSE | Weekly Bias |
|---|---:|---:|---:|---:|
| Raw float (experimental headline) | 32.72 | 2.770 | 3.909 | +0.940 |
| Floor + consistency (`floor_round`) | **29.96** | **2.536** | **3.868** | **-0.464** |
| Aggregate-preserving round + party | 32.07 | 2.714 | 3.928 | +0.929 |
| Aggregate-preserving round | 34.60 | 2.929 | 4.097 | +1.286 |
| No book-min / soft suppress guest-only | 37.13 | 3.143 | 4.440 | +1.000 |
| Book-min only | **38.40** | 3.250 | 4.555 | +1.250 |
| Consistency only (round + book-min + g≥b) | **38.40** | 3.250 | 4.555 | +1.250 |
| Party-floor only (with consistency) | **38.40** | 3.250 | 4.555 | +1.250 |
| Variant B | **38.40** | 3.250 | 4.555 | +1.250 |
| **Variant A** | **38.40** | 3.250 | 4.555 | +1.250 |
| Ceil bookings | 49.79 | 4.214 | 5.230 | +3.429 |

### Rule contribution (bookings)

1. **Dominant harm:** inventing `bookings=1` when rounded guests &gt; 0 but rounded bookings = 0 (booking-minimum / consistency). Robust Median guests are often &gt; 0 on low two-stage booking days → false positive bookings.
2. **Party-size floor:** does **not** change bookings WAPE vs book-min alone (same 38.40); it mainly inflates **guest** forecasts.
3. **Ordinary round-half-up** feeds the book-min path more often than floor/aggregate alternatives.
4. **Ceil bookings** is worst (strong over-forecast).

## Best configuration

**Recommended ops for experimental V3.1 shadow integers:** `floor_round`

- Floor bookings and guests to nonnegative integers.
- Then enforce consistency: `guests >= bookings`; if `guests > 0` then `bookings >= 1`.
- No party-size floor; no extra booking-minimum beyond consistency after flooring.

### Before → after (Two-Stage bookings, rolling)

| | Weekly WAPE | MAE | RMSE | Bias | Daily WAPE |
|--|------------:|----:|-----:|-----:|-----------:|
| Raw (frozen headline) | 32.72 | 2.770 | 3.909 | +0.940 | 64.94 |
| Variant A | 38.40 | 3.250 | 4.555 | +1.250 | 67.09 |
| **floor_round** | **29.96** | **2.536** | **3.868** | **-0.464** | **59.49** |

Weekly WAPE target “preserve or improve 32.72%” after ops: **achieved with floor_round** (29.96%). MAE &lt; 2.50 not met (2.54). RMSE &lt; 3.50 not met (3.87).

## Production baseline vs Two-Stage (identical periods)

Weekly bookings WAPE:

| Ops | Production baseline (weekday+weekday) | Two-Stage + Robust Median |
|---|---:|---:|
| Raw float | 37.55 | 32.72 |
| Variant A | 42.19 | 38.40 |
| floor_round | 34.60 | **29.96** |
| no_book_min_party | **33.76** | 37.13 |

For the production baseline pair, `no_book_min_party` edges floor_round; for the V3.1 two-stage pair, **floor_round wins**.

## Past-only selection note

Per-origin nested selection among the predeclared SELECTABLE catalog most often chose `soft_book_min` (threshold 0) for two-stage; pooled `selected_past_only` weekly WAPE = **35.02%**, worse than fixed `floor_round`. Nested ops loss did **not** recover the best fixed policy—another reason to freeze a **predeclared** ops mode rather than adaptive ops search for serving.

## Tests

- `python -B experiments/v3_1_postprocessing_optimization_20261124/test_ops.py` — integer/consistency invariants.
- Experiment asserts raw two-stage weekly WAPE still **32.7223…** (original evaluation preserved).
- Production SHA-256 unchanged.

## Remaining limitations

1. Retrospective 28-origin period (previously inspected).
2. `floor_round` under-forecasts on average (bias −0.46 week).
3. Guest party-size semantics for admin UX are weaker without Variant A floor.
4. Past-only adaptive ops selection underperformed the best fixed rule.
5. Not deployed; serving still uses Variant A until a separate integration approval.
6. No further model changes; guests recipe unchanged.

## Artifacts

`results/comparison.csv`, `bookings_ops_ranking.csv`, `daily_predictions.csv`, `results.json`, `protocol.json`.
