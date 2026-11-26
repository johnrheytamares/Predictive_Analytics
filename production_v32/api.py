"""Production-compatible v3.2 entry point; production_v31 remains the active default."""
import logging, os, secrets
from typing import Any, Optional
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from . import runtime

logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO'))
app = FastAPI(title='ReserVision Predictive Analytics',version='3.2.0')
origins=[s.strip() for s in os.getenv('CORS_ALLOW_ORIGINS','').split(',') if s.strip()]
if origins:
    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,
        allow_methods=['GET','POST'],allow_headers=['Authorization','X-API-Key','Content-Type'])

def require_api_key(x_api_key:Optional[str]=Header(default=None,alias='X-API-Key')):
    expected=os.getenv('PREDICTION_API_KEY','').strip()
    if expected and (not x_api_key or not secrets.compare_digest(x_api_key.strip(),expected)):
        raise HTTPException(status_code=401,detail='Invalid or missing API key.')

class HistoryRefreshRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    source: Optional[str] = 'booking_demand_daily'

@app.on_event('startup')
def warmup():
    ready=runtime.health()
    if not ready['model_ready']:
        logging.error('V3.2 warmup failed; health remains live but not model-ready. No fallback.')
    if ready.get('production_approved') is not True:
        logging.error('V3.2 warmup: production_approved is not true after authorization.')

@app.get('/health')
def health():return runtime.health()

def _forecast_error_response(exc: runtime.ForecastError):
    detail='Invalid date format. Use YYYY-MM-DD.' if exc.code=='INVALID_DATE' else exc.message
    if exc.code in {'MODEL_MISSING','METADATA_MISSING'}:detail='Required prediction model is missing.'
    if exc.code in {
        'ROBUST_RANGE_UNSUPPORTED','ROBUST_HISTORY_INVALID','MODEL_CONFIGURATION_MISMATCH',
        'INVALID_HISTORY','HISTORY_REFRESH_FAILED','HISTORY_REFRESH_NOT_READY','HISTORY_SNAPSHOT_MISSING',
        'FROZEN_PARAMETERS_UNAVAILABLE',
    }:
        return JSONResponse(status_code=exc.status_code,content={'detail':detail,'code':exc.code,**(exc.extra or {})},
                            headers={'X-Error-Code':exc.code})
    raise HTTPException(status_code=exc.status_code,detail=detail,headers={'X-Error-Code':exc.code}) from None

@app.get('/forecast')
def forecast(start_date:str=Query(...,description='YYYY-MM-DD'),
             end_date:str=Query(...,description='YYYY-MM-DD'),
             allow_historical:bool=Query(False,description='Allow ranges that overlap training history'),
             _:None=Depends(require_api_key)):
    try:
        return runtime.forecast(start_date,end_date,allow_historical=allow_historical)
    except runtime.ForecastError as exc:
        return _forecast_error_response(exc)
    except Exception:
        logging.exception('V3.2 forecast failed')
        raise HTTPException(status_code=500,detail='Forecast generation failed.') from None

@app.post('/history/refresh')
def history_refresh(body:HistoryRefreshRequest,_:None=Depends(require_api_key)):
    """Validate and activate served-history overlay. Does not retrain or mutate hashed artifacts."""
    try:
        return runtime.refresh_history(rows=body.rows, source=body.source or 'booking_demand_daily')
    except runtime.ForecastError as exc:
        return _forecast_error_response(exc)
    except Exception:
        logging.exception('V3.2 history refresh failed')
        raise HTTPException(status_code=500,detail='History refresh failed.') from None
