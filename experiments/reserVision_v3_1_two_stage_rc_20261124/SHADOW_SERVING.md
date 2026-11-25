# Isolated FastAPI shadow serving — ReserVision Version 3.1

**Status:** Validated for a local isolated shadow process only. Canonical `api.py` and production artifacts are unchanged. Prospective validation has not run.

## Configuration (frozen)

| Target | Selection |
|--------|-----------|
| Bookings | original `two_stage` |
| Guests | `robust_median_guests` |
| Post-process | `floor_round` |

## How to run (isolated)

From `Predictive_Analytics` root (separate port from production):

```text
python -B -m uvicorn experiments.reserVision_v3_1_two_stage_rc_20261124.shadow_api:app --host 127.0.0.1 --port 8002
```

Endpoints (same paths as production):

- `GET /health`
- `GET /forecast?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
- Optional diagnostic: `&compare_variant_a=true`

Use an unused loopback port; do not change the production URL or Node configuration. Existing `PREDICTION_API_KEY` authentication and optional `CORS_ALLOW_ORIGINS` behavior are preserved. `/health` remains public. Do not run artifact preparation again: it now refuses an existing frozen artifact directory.

Forward inference reuses the already recorded configuration `prob_days=14`, `size_days=84`, `shrink=0`, from `verification_horizons_postprocess.json`. There is no runtime fitting, tuning, or history refresh. The saved history ends on 2026-11-21. `serving_snapshot.json` verifies 56 frozen evidence, artifact, and canonical dependency files. Historical replay with `allow_historical=true` is supported only at the 28 recorded seven-day origins; other historical starts return `FROZEN_PARAMETERS_UNAVAILABLE`, without fallback or optimization.

## Compatibility

- Top-level forecast keys and per-day row keys match production.
- Model identity differs: `model_version=3.1`, `two_stage`, `floor_round`, `shadow_serving_only=true`, `production_approved=false`.
- Additive fields: separate bookings/guests weekly metrics; optional Variant A comparison block.
- Weekly metric blocks now expose WAPE, MAE, RMSE, Bias, observation count and actual total for each target and policy, computed from the existing frozen prediction CSV. Original `model_metadata` remains byte-for-byte unchanged on disk and unchanged in the response. Bounds are not calibrated intervals; their status remains `Not evaluated`.
- The 7/30/90-day responses are structurally validated. Retrospective accuracy covers seven-day horizons only; no 30/90-day accuracy claim is made.

## Retrospective vs prospective

| Layer | Status |
|-------|--------|
| Retrospective weekly metrics in metadata | Preserved (raw 32.72%; floor_round 29.96%) |
| Prospective shadow weeks | Not run |
| Production / Dokploy / Node / Vue | Untouched |

## Guest totals vs Variant A

Variant A applies a ~7 guests-per-booking party floor after rounding. `floor_round` does not. On the forward cutoff 2026-11-22:

| Horizon | floor_round guests | Variant A guests |
|--------:|-------------------:|-----------------:|
| 7 | 3 | 14 |
| 30 | 14 | 63 |
| 90 | 38 | 175 |

The reduction is an **ops-policy** difference; the Robust Median guest model is unchanged. The comparison applies the two policies to the same selected raw forecasts; it is not a comparison with the current production service's different model outputs.

## Validation

Run `python -B experiments/reserVision_v3_1_two_stage_rc_20261124/test_shadow_serving.py` from the analytics root. The combined suite passed 17 tests, including all 28 frozen origin replays, API authentication/errors, integer and consistency checks, prefix stability, integrity failure handling, unchanged canonical API outputs, and a temporary loopback HTTP server startup/request/shutdown. No persistent server was left running. New test evidence is in `integration_validation/results.json`; prior verification files are preserved.
