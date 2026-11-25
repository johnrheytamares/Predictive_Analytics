RESERVISION V3.1 PRODUCTION-SERVING PREPARATION — 25 November 2026

STATUS
Prepared and locally verified. Not deployed, production-approved, or proven
prospectively accurate. Docker image validation remains blocked by the unavailable
Docker Desktop Linux engine. No models were retrained or optimized.

FILES ADDED
Dockerfile.v31                 Separate image definition; original Dockerfiles unchanged.
Dockerfile.v31.dockerignore    Dedicated context: retains frozen evaluation CSVs.
production_v31/api.py          Compatible GET /health and GET /forecast entry point.
production_v31/runtime.py      Frozen inference adapter; no fallback or model search.
production_v31/serve.py        Container launcher honoring PORT.
production_v31/healthcheck.py  Requires model_ready=true and version 3.1, not just HTTP 200.
production_v31/verify_release.py  Portable integrity/readiness check used by image build.
production_v31/release_manifest.json  41 explicit runtime/frozen/rollback file hashes.
production_v31/test_serving.py  Focused production and original shadow regressions.
production_v31/validation/results.json  Actual local test results and dynamic metrics.

UNCHANGED FROZEN SELECTION
Original Two-Stage bookings, Robust Median guests, floor_round postprocessing.
Existing forward parameters come from recorded horizon verification, not a new
optimization. Historical replay uses only existing frozen origin parameters.
The adapter reuses shadow_predict.py and its prediction helpers unchanged. Its
Windows workspace integrity adapter is replaced IN MEMORY with a portable
manifest validator; none of the frozen source/evidence files are rewritten.
All 56 original serving_snapshot protected files were separately checked: zero
mismatches. Canonical api.py, predict_real.py, all original models/metadata and
Dockerfile remain unchanged for rollback. The new image also retains those
canonical files, but its default command selects production_v31.serve explicitly.

API AND ENVIRONMENT CONTRACT
GET /health remains public and HTTP 200 for liveness. model_ready is authoritative;
bad/missing/wrong-version artifacts result in false and never load canonical v3.0.
GET /forecast keeps start_date, end_date, allow_historical and X-API-Key, plus
the existing optional compare_variant_a diagnostic. Original required response
and daily-row fields remain. Authentication, validation status/error headers and
the CORS_ALLOW_ORIGINS policy are retained. An unset PREDICTION_API_KEY continues
to mean optional local authentication as in the existing service; deployment
must verify the intended existing key configuration before routing traffic.

Environment variables:
- PORT: honored by the new container launcher and readiness healthcheck (default 8001).
- PREDICTION_API_KEY: same shared-key header behavior; never logged in tests.
- CORS_ALLOW_ORIGINS and LOG_LEVEL: existing behavior preserved.
- PREDICTION_MAX_FORECAST_DAYS: existing limit retained; default 90, no silent increase.
- PREDICTION_ARTIFACT_DIR: optional exact v3.1 artifact directory; Docker defaults
  to /app/experiments/reserVision_v3_1_two_stage_rc_20261124/artifacts. A stale v3.0
  mount or a modified/new history bundle fails integrity. No fallback search.

Production adapter envelope identifies production_prepared rather than isolated
shadow. production_approved is activated only through the release authorization
constant in production_v31/runtime.py. Retrospective weekly metrics remain the
frozen evaluation evidence; prospective_accuracy_validated stays false. Frozen
source metadata files remain untouched. API metadata copies retain the historical
scores and enrich target selected/raw/operational weekly metrics from the existing
saved prediction CSV. No score is hardcoded. Complete separate weekly_performance
blocks are exposed on both endpoints. Bounds remain Not evaluated.

DATE/HORIZON SUPPORT
Current-day and later requested starts, each with 7/30/90-day horizons, are
supported without needing observations adjacent to the future target date.
Both models use actual saved history strictly before the requested cutoff;
future unobserved days are not treated as zeros. No recursive predictions enter
history. The saved history ends November 21, 2026 and is disclosed in lifecycle.
Health capabilities are anchored to the current Manila serving date (or history
end + one day, whichever is later), avoiding an obsolete original-cutoff date list.
Forecast capabilities are anchored to the actual requested start.
allow_historical requests retain the frozen rule: only the 28 saved seven-day
origins may replay; arbitrary historical starts are rejected rather than retuned.
Seven-day retrospective scores do NOT validate 30/90-day accuracy.

LIFECYCLE AND SYNCHRONIZATION FINDINGS
Source inspected read-only: backend predictionTrainingService.js, specifically
resolveRemoteTrainingStatus(), trainForecastModels(), syncAndRetrainModel().
- Synchronization rebuilds booking_demand_daily in the database.
- With PREDICTIVE_ANALYTICS_URL configured, local training is deliberately skipped.
- dataset_sync_refreshes_served_history remains false: Sync Training Data alone
  does not mutate served history. History refresh is a separate authenticated
  POST /history/refresh that validates booking_demand_daily-shaped rows, writes
  a runtime overlay (not hashed joblibs), snapshots the prior overlay, and rolls
  back on failure. Hyperparameters and retrospective weekly metrics stay frozen.
- Node still derives history_refresh_supported only from
  dataset_sync_refreshes_served_history and does not POST /history/refresh yet.
  End-to-end Sync → refresh requires that Node wiring (reported, not implemented).
- /health metadata and model_ready are the serving authority.
- With no remote URL, the unchanged Node production path can still invoke its
  canonical local v3.0 script. A later deployment must verify explicit remote
  routing to the new entry point; this Python-only change cannot control Node's
  external dispatch. No Node configuration was changed here.
- The regular Node mapper retains target metadata but drops additive full weekly
  performance blocks; the current Vue helper may then label selected scores as
  Holdout WAPE. Correct production propagation/labeling requires a separately
  authorized integration review. The shadow-specific adapter expects shadow-only
  flags and is not the production activation path.
No synchronization or database operation was executed: its implementation writes
database contents, which is outside this task. Compatibility here is API tests
plus source inspection, not a live sync/refresh or deployed end-to-end test.

ACTUAL TESTS
Command: python -B production_v31/test_serving.py
15 tests passed, zero failures/errors; final run 13.337 seconds.
Includes ten production cases and five original shadow API regression cases:
- Current Manila date and June 1, 2027 starts at 7/30/90 days, tuner prohibited.
- Exact original forward totals and Variant A diagnostic preservation.
- Existing parameter schemas, response keys and error-code compatibility.
- Shared key auth, public liveness and complete separate dynamic weekly metrics.
- Wrong v3.0 directory and missing artifacts fail closed, with no disk mutation.
- Actual Docker COPY destinations mirrored to a clean temporary tree; required
  evaluation CSV present, portable manifest/readiness/CORS verified in a subprocess.
- Temporary real Uvicorn HTTP startup, future 90-day response, PORT-aware
  healthcheck process, and clean shutdown.
- Source-only synchronization contract test; no database access or training.
Additional command: python -B -m production_v31.verify_release — passed.
No persistent process was left running.

DOCKER VALIDATION STATUS
Attempt: docker build -f Dockerfile.v31 -t reservision-predictive:3.1-prepared .
The sandbox attempt hit context permissions. A permitted unsandboxed retry reached
Docker but failed because //./pipe/dockerDesktopLinuxEngine does not exist.
No image was produced. Linux dependency installation, actual Docker context
handling and in-container startup/healthcheck therefore remain unverified.
Dockerfile-specific ignore precedence reference:
https://docs.docker.com/build/concepts/context/
Do not mistake the passing copied-layout test for a successful container build.

REMAINING GATES BEFORE PRODUCTION
1. Start the intended Linux Docker engine and build/test the separate image locally.
2. Verify existing deployment command uses the new production_v31 entry point,
   correct v3.1 artifact directory, shared key, port and backend remote URL; no
   stale command/mount may select canonical v3.0. Dokploy was not inspected/changed.
3. Resolve/accept the explicit frozen-history lifecycle and separately authorize
   any future versioned history refresh. A database sync alone is insufficient.
4. Review the existing Node sync version/readiness reporting and regular Node/Vue
   weekly-metric propagation before end-to-end production activation.
5. Keep the currently deployed image/configuration digest for rollback; repository
   rollback bytes are verified, but the current deployed image was not inspected.
6. Obtain explicit release approval. Prospective accuracy remains unvalidated.

No Node.js, Vue, database, database configuration, Dokploy, pushes or deployments
were changed. Canonical production and frozen candidate evidence remain intact.
