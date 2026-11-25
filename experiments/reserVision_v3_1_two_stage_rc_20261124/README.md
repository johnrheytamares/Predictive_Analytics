# ReserVision Version 3.1 — Two-Stage + floor_round experimental RC

Freezes **original** intermittent two-stage bookings, **Robust Median** guests, and **floor_round** operational postprocessing.

- Report: [`RELEASE_CANDIDATE.md`](RELEASE_CANDIDATE.md)
- Prospective procedure: [`SHADOW_VALIDATION.md`](SHADOW_VALIDATION.md)
- Isolated FastAPI: [`SHADOW_SERVING.md`](SHADOW_SERVING.md)
- Manifest: [`PACKAGE_MANIFEST.json`](PACKAGE_MANIFEST.json)
- Verification: [`verification_floor_round.json`](verification_floor_round.json), [`verification_shadow_fastapi.json`](verification_shadow_fastapi.json)

**Retrospective:** raw weekly WAPE 32.72%; floor_round 29.96% / MAE 2.536 / RMSE 3.868 / Bias −0.464.  
**Isolated shadow API:** ready on separate process (`shadow_api.py`). Production API untouched.  
**Prospective shadow weeks:** not started. **Shadow GO / Production NO-GO.**
