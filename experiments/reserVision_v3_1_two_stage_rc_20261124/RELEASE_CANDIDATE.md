# ReserVision Version 3.1 — Two-Stage + floor_round experimental release candidate

**Release ID:** `reservision_v3_1_two_stage_rc_20261124`  
**Scope:** `release-candidate-not-deployed`  
**Production approved:** No  
**Prospective shadow validation:** Not started (required before any serving discussion)  
**Optimization in this step:** None (models frozen; ops mode selected from completed ablation only)

## Selected configuration (frozen)

| Target | Selection |
|--------|-----------|
| Bookings | **Original two-stage** from `experiments/intermittent_bookings_20261124` |
| Guests | **Robust Median** cutoff-anchored (`robust_release_candidate_20261124` artifacts preserved) |
| Post-process | **`floor_round`** from `experiments/v3_1_postprocessing_optimization_20261124` |
| Explicitly not selected | `two_stage_optimized`, adaptive two-stage, direct-weekly challengers, Variant A ops, nested past-only ops selection |

## Retrospective vs prospective

| Layer | Status |
|-------|--------|
| Retrospective rolling 28 origins (raw + ops) | Completed and reproduced below |
| Nested past-only ops selection evidence (35.02%) | Preserved; **not** the serving ops mode |
| Prospective shadow on unseen future weeks | **Not run** — procedure in `SHADOW_VALIDATION.md` |
| Production / staging deploy | **NO-GO** |

Do not treat retrospective WAPE as live prospective accuracy.

## Reproduced raw bookings (selection protocol)

Rolling 28 origins — exact match to frozen intermittent headline:

| Metric | Rounded | Exact |
|--------|--------:|------:|
| Weekly WAPE | 32.72% | 32.72234791049451 |
| Weekly MAE | 2.770 | 2.769713019566857 |
| Weekly RMSE | 3.909 | 3.909415622750741 |
| Weekly Bias | +0.940 | +0.9403568384502454 |

## Reproduced operational bookings (`floor_round`)

Same 28 origins after floor → consistency:

| Metric | Rounded | Exact |
|--------|--------:|------:|
| Weekly WAPE | **29.96%** | 29.957805907172997 |
| Weekly MAE | **2.536** | 2.5357142857142856 |
| Weekly RMSE | **3.868** | 3.8683699040145725 |
| Weekly Bias | **−0.464** | −0.4642857142857143 |

Vs raw: weekly WAPE improves by ≈2.76 pp. Vs Variant A (~38.40%): floor_round is substantially better. Nested past-only selected ops remains **35.02%** weekly WAPE (evidence only; not frozen serving mode).

## Guest forecasting impact

Robust Median guests under the same rolling protocol:

| Ops mode | Guests weekly WAPE |
|----------|-------------------:|
| raw_float | 38.83089770354906 |
| **floor_round** | **38.78914405010438** |
| variant_a | 43.0062630480167 |

**floor_round does not harm guest accuracy** versus raw (slightly lower WAPE). Variant A worsens guests.

## Consistency

All floor_round rows: nonnegative ints; `guests >= bookings`; if `guests > 0` then `bookings >= 1` (no zero-booking / positive-guest pairs after ops). Regenerated sample week matches saved predictions.

## Tests

| Suite | Result |
|-------|--------|
| `verify_floor_round_rc.py` | **8/8 passed** |
| Canonical production SHA-256 | **unchanged** |

## Horizon compatibility (structural, floor_round)

History through 2026-11-21; cutoff 2026-11-22; original nested tune; Robust Median guests; floor_round:

| Days | Operational bookings sum | Operational guests sum |
|-----:|-------------------------:|-----------------------:|
| 7 | 2 | 3 |
| 30 | 9 | 14 |
| 90 | 25 | 38 |

Finite nonnegative forecasts; consistency invariants hold. Compatibility evidence only — not a new accuracy claim.

## Release-candidate readiness

| Item | Status |
|------|--------|
| Frozen two-stage + Robust Median + floor_round | Ready |
| Raw 32.72% + ops 29.96/2.536/3.868/−0.464 reproduction | Ready |
| Nested 35.02% evidence preserved | Ready |
| Guest impact assessed (no harm vs raw) | Ready |
| Consistency + 7/30/90 structural checks | Ready |
| Shadow validation procedure | Documented (`SHADOW_VALIDATION.md`) |
| Isolated shadow FastAPI (`shadow_api.py`, port 8002) | Ready — does not replace production |
| Serving integration in production `predict_real.py` / `api.py` | **Not done** (production untouched) |
| Production / Dokploy / Node / Vue / DB | **Not modified** |
| Prospective shadow / staging / production | **NO-GO** until separate approval |

**Experimental RC readiness:** **GO for isolated shadow serving + prospective validation.**  
**Production readiness:** **NO-GO.**

## Remaining limitations

1. All headline metrics above are **retrospective**; prospective weeks are untested.
2. Sparse holdout still poor for two-stage (weekly WAPE 143.75% raw).
3. 10–20% weekly WAPE target unmet.
4. Two-stage + floor_round are only on the isolated shadow app (`shadow_api.py`); production FastAPI still serves the prior stack.
5. Zero-demand provenance remains unresolved from earlier audits.
6. No Dokploy staging of this configuration has been executed.
7. Nested ops selection (35.02%) is worse than fixed floor_round; fixed mode is frozen deliberately.

## Package layout

- `package/frozen_selection.json` — Version 3.1 complete freeze (bookings + guests + floor_round)
- `package/evaluation_evidence/` — preserved intermittent results
- `package/postprocess_evidence/` — floor_round ablation artifacts
- `package/robust_median_artifacts/` — copied Robust Median RC artifacts
- `verification_floor_round.json`, `verification_shadow_fastapi.json`
- `shadow_api.py`, `shadow_predict.py`, `artifacts/` (isolated shadow only)
- `PACKAGE_MANIFEST.json`, `SHADOW_VALIDATION.md`, `SHADOW_SERVING.md`
