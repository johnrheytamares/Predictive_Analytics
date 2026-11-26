ReserVision production_v32 — Sum-Ratio guest estimator (production-approved)

- Source RC: experiments/reserVision_v3_2_sumratio_guests_rc_20261126/
- Bookings: two_stage (identical to v3.1)
- Guests: guests_per_booking_sumratio (past-84-day Σg/Σb)
- Postprocess: floor_round
- Demographics: period-preserving allocate_guest_mix_period
- Intervals: past-only party-size ratio quantiles q10–q90 (observed coverage disclosed; not a validated confidence level)
- PRODUCTION_APPROVED = True (authorized 2026-11-27)
- Rollback: git 8892d29 / experiments/reserVision_v3_1_two_stage_rc_20261124/

Local serve:
  set PORT=8002
  set PREDICTION_API_KEY=<shared-key>
  python -m production_v32.serve

Wire Node to v3.2 (cutover):
  set PREDICTIVE_ANALYTICS_URL=http://127.0.0.1:8002
  set PREDICTION_API_KEY=<same-key>

Verify:
  python -m production_v32.verify_release
  python -m production_v32.test_serving

Docker:
  docker build -f Dockerfile.v32 -t reservision-predictive:3.2-prepared .
  powershell -File production_v32/validation/docker_smoke_test.ps1

Rollback: see production_v32/ROLLBACK.txt (git 8892d29 / production_v31).
