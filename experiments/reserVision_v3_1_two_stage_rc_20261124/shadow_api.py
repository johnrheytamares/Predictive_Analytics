"""Isolated FastAPI app for ReserVision V3.1 shadow serving.

Compatible /health and /forecast surface with production api.py, but does not
replace or bind over the production service. Run separately, e.g.:

    uvicorn experiments.reserVision_v3_1_two_stage_rc_20261124.shadow_api:app --port 8002
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from predict_real import ForecastError  # noqa: E402
import shadow_predict as sp  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [reservision-v3.1-shadow] %(message)s",
)
logger = logging.getLogger("reservision-v3.1-shadow")

app = FastAPI(
    title="ReserVision Predictive Analytics — V3.1 Shadow",
    version="3.1.0-shadow",
    description=(
        "Isolated shadow microservice for two_stage + robust_median_guests + "
        "floor_round. Not production."
    ),
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
    expected = os.getenv("PREDICTION_API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.on_event("startup")
def warmup_models() -> None:
    try:
        metadata = sp.load_metadata(use_cache=True)
        sp.validate_readiness(metadata)
        bookings, guests = sp.get_bundles(use_cache=True)
        sp.validate_v31_configuration(bookings, guests, metadata)
        logger.info(
            "V3.1 shadow models loaded (two_stage + robust_median_guests + floor_round)"
        )
    except ForecastError as exc:
        logger.error("Shadow warmup failed (%s): %s", exc.code, exc.message)
    except Exception:
        logger.exception("Unexpected V3.1 shadow warmup failure")


def _model_ready(metadata: dict | None) -> bool:
    if not metadata or not isinstance(metadata, dict):
        return False
    return (
        metadata.get("ready") is True
        and metadata.get("safe_for_live_use") is True
        and metadata.get("is_demo") is not True
        and metadata.get("production_approved") is not True
        and metadata.get("shadow_serving_only") is True
    )


@app.get("/health")
def health():
    payload: dict = {
        "status": "ok",
        "service": "predictive-analytics-v3.1-shadow",
        "model_ready": False,
        "metadata": None,
        "shadow_serving_only": True,
        "production_approved": False,
    }
    try:
        metadata = sp.load_metadata(use_cache=True)
        if metadata:
            payload["metadata"] = metadata
            payload["model_ready"] = _model_ready(metadata)
            bookings, guests = sp.get_bundles(use_cache=True)
            sp.validate_v31_configuration(bookings, guests, metadata)
            cutoff = pd.Timestamp(metadata["date_to"]) + pd.Timedelta(days=1)
            payload["forecast_capabilities"] = sp.shadow_capabilities(guests, cutoff)
            payload["model_ready"] = payload["model_ready"] and bool(
                payload["forecast_capabilities"]["supported_dates"]
            )
            payload["selected_configuration"] = {
                "bookings": "two_stage",
                "guests": "robust_median_guests",
                "postprocess": "floor_round",
            }
            payload["weekly_performance"] = sp.weekly_performance()
            payload["evaluation_scope"] = metadata.get("evaluation_scope")
            payload["prospective_validation"] = metadata.get("prospective_validation")
    except Exception:
        payload["model_ready"] = False
        logger.exception("Failed to attach V3.1 shadow metadata to /health")
    return payload


@app.get("/forecast")
def forecast(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    allow_historical: bool = Query(
        False,
        description="Allow ranges that overlap training history",
    ),
    compare_variant_a: bool = Query(
        False,
        description="Include Variant A totals comparison (shadow diagnostics only)",
    ),
    _: None = Depends(require_api_key),
):
    try:
        return sp.run_forecast(
            start_date,
            end_date,
            allow_historical=allow_historical,
            use_cache=True,
            include_variant_a_comparison=compare_variant_a,
        )
    except ForecastError as exc:
        detail = exc.message
        if exc.code in {
            "ROBUST_RANGE_UNSUPPORTED",
            "ROBUST_HISTORY_INVALID",
            "MODEL_CONFIGURATION_MISMATCH",
        }:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.message, "code": exc.code, **(exc.extra or {})},
                headers={"X-Error-Code": exc.code},
            )
        if exc.code == "INVALID_DATE":
            detail = "Invalid date format. Use YYYY-MM-DD."
        elif exc.code in {"MODEL_MISSING", "METADATA_MISSING"}:
            detail = "Required prediction model is missing."
        raise HTTPException(
            status_code=exc.status_code,
            detail=detail,
            headers={"X-Error-Code": exc.code},
        ) from None
    except Exception:
        logger.exception("V3.1 shadow forecast generation failed")
        raise HTTPException(
            status_code=500,
            detail="Forecast generation failed.",
        ) from None
