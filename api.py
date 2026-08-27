"""
Predictive Analytics FastAPI microservice.

Wraps existing Phase 4 forecast logic from predict_real.py for Dokploy deployment.
Intended consumers: ReserVision Node.js backend (internal HTTP), not browsers.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from predict_real import (
    ForecastError,
    get_model_bundles,
    load_metadata,
    run_forecast,
    validate_readiness,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [predictive-analytics] %(message)s",
)
logger = logging.getLogger("predictive-analytics")
# Prophet optionally imports plotly for notebooks; missing plotly is harmless for API forecasts.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

app = FastAPI(
    title="ReserVision Predictive Analytics",
    version="1.0.0",
    description="Booking/guest demand forecast microservice",
)


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


_origins = _cors_origins()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """
    Optional shared-secret gate for Node → Python calls.
    When PREDICTION_API_KEY is unset, requests are allowed (local/dev).
    """
    expected = os.getenv("PREDICTION_API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.on_event("startup")
def warmup_models() -> None:
    """Load metadata + model bundles once at process start (fail soft for /health)."""
    try:
        # Mute optional plotly warning from Prophet during cold start.
        logging.getLogger().setLevel(logging.INFO)
        metadata = load_metadata(use_cache=True)
        validate_readiness(metadata)
        get_model_bundles(use_cache=True)
        logger.info("Forecast models loaded successfully")
    except ForecastError as exc:
        logger.error("Model warmup failed (%s): %s", exc.code, exc.message)
    except Exception:
        logger.exception("Unexpected model warmup failure")


@app.get("/health")
def health():
    """Lightweight readiness probe — does not run forecasts."""
    return {
        "status": "ok",
        "service": "predictive-analytics",
    }


@app.get("/forecast")
def forecast(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    allow_historical: bool = Query(
        False,
        description="Allow ranges that overlap training history",
    ),
    _: None = Depends(require_api_key),
):
    try:
        return run_forecast(
            start_date,
            end_date,
            allow_historical=allow_historical,
            use_cache=True,
        )
    except ForecastError as exc:
        detail = exc.message
        if exc.code == "INVALID_DATE":
            detail = "Invalid date format. Use YYYY-MM-DD."
        elif exc.code == "MODEL_MISSING":
            detail = "Required prediction model is missing."
        elif exc.code == "METADATA_MISSING":
            detail = "Required prediction model is missing."
        raise HTTPException(
            status_code=exc.status_code,
            detail=detail,
            headers={"X-Error-Code": exc.code},
        ) from None
    except Exception:
        logger.exception("Forecast generation failed")
        raise HTTPException(
            status_code=500,
            detail="Forecast generation failed.",
        ) from None
