# Predictive Analytics API

Standalone FastAPI microservice that serves ReserVision booking/guest demand forecasts using the existing Phase 4 models (`bookings_model.joblib`, `guests_model.joblib`).

This is a **backend microservice** (port `8001`). The ReserVision Node.js backend should call it internally — do not call the models from the frontend.

## Local development

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8001
```

CLI forecast (unchanged):

```bash
python predict_real.py --start-date 2026-08-27 --end-date 2026-09-02
```

## Health check

```text
GET /health
```

Example:

```json
{
  "status": "ok",
  "service": "predictive-analytics",
  "model_ready": true,
  "metadata": {
    "model_version": "3.0",
    "trained_at": "2026-08-20T04:37:45Z",
    "training_days": 256,
    "date_from": "2025-12-01",
    "date_to": "2026-08-13",
    "data_source": "booking_demand_daily",
    "targets": { "...": "holdout metrics from training" },
    "safe_for_live_use": true
  }
}
```

`metadata` is the real `forecast_models_metadata.json` artifact (or `null` if missing). Metrics are never invented.

## Forecast

```text
GET /forecast?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
```

Optional:

```text
&allow_historical=true
```

Optional auth header (when `PREDICTION_API_KEY` is set):

```text
X-API-Key: <shared-secret>
```

Response fields match the existing Phase 4 payload (`success`, `model`, `forecast_range`, `guest_mix`, `forecast[]`, …).

## Docker

```bash
docker build -t predictive-analytics .
docker run --rm -p 8001:8001 predictive-analytics
```

With optional API key:

```bash
docker run --rm -p 8001:8001 -e PREDICTION_API_KEY=change-me predictive-analytics
```

## Dokploy

Use **Dockerfile** build (do not use Nixpacks / Node detection):

| Setting | Value |
|---------|--------|
| Build Type | `Dockerfile` |
| Build Path | `/` |
| Container Port | `8001` |
| Start command | from Dockerfile (`uvicorn api:app --host 0.0.0.0 --port 8001`) |

Do **not** set start command to `python predict_real.py` or `npm start`.

### Optional environment variables

| Variable | Purpose |
|----------|---------|
| `PREDICTION_API_KEY` | Shared secret; if set, `/forecast` requires `X-API-Key` |
| `CORS_ALLOW_ORIGINS` | Comma-separated origins (usually empty; backend-to-backend) |
| `PREDICTION_MAX_FORECAST_DAYS` | Max inclusive days per request (default `90`) |
| `LOG_LEVEL` | Logging level (default `INFO`) |

## Architecture

```text
ReserVision Frontend
        │
        ▼
ReserVision Backend (Node.js :8000)
        │  GET /forecast
        ▼
Predictive Analytics (FastAPI :8001)
        │
        ▼
*.joblib + forecast_models_metadata.json
```

## Note about `package.json`

`package.json` exists for older Node helper scripts (`load_bookings.js`) and is **not** used by this API. Deploy with the Dockerfile so Dokploy does not treat the repo as a Node app.
