"""Validated served-history refresh for production_v31.

Mutates only a writable overlay directory. Frozen joblib hashes, hyperparameters,
and retrospective evaluation evidence remain unchanged. No v3.0 fallback.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ACTIVE_NAME = 'active_history.json'
SNAPSHOT_NAME = 'snapshot_previous.json'
STAGING_NAME = 'staging_history.json'
MINIMUM_DAYS = 120

_lock = threading.RLock()
_overlay_cache: dict | None = None


class HistoryRefreshError(Exception):
    def __init__(self, message, *, code='HISTORY_REFRESH_FAILED', status_code=422, details=None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


def history_dir() -> Path:
    configured = os.getenv('PREDICTION_HISTORY_DIR', '').strip()
    if configured:
        return Path(configured).resolve()
    artifact_dir = Path(os.getenv('PREDICTION_ARTIFACT_DIR') or '').resolve() if os.getenv('PREDICTION_ARTIFACT_DIR') else None
    if artifact_dir is not None:
        return (artifact_dir.parent / 'runtime_history').resolve()
    return (Path(__file__).resolve().parents[1] / 'runtime_history').resolve()


def _active_path() -> Path:
    return history_dir() / ACTIVE_NAME


def _snapshot_path() -> Path:
    return history_dir() / SNAPSHOT_NAME


def _staging_path() -> Path:
    return history_dir() / STAGING_NAME


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def clear_overlay_cache() -> None:
    global _overlay_cache
    with _lock:
        _overlay_cache = None


def load_active_overlay(*, use_cache: bool = True) -> dict | None:
    global _overlay_cache
    with _lock:
        if use_cache and _overlay_cache is not None:
            return copy.deepcopy(_overlay_cache)
        payload = _read_json(_active_path())
        if use_cache:
            _overlay_cache = copy.deepcopy(payload) if payload else None
        return payload


def frozen_baseline_rows(bookings_bundle: dict, guests_bundle: dict) -> tuple[list[dict], list[dict]]:
    bookings = list(bookings_bundle.get('baseline_history') or [])
    guests = list(guests_bundle.get('baseline_history') or [])
    if not bookings or not guests:
        raise HistoryRefreshError('Frozen baseline_history is missing from model bundles.', code='MODEL_INVALID')
    return bookings, guests


def _to_frame(rows: list[dict], value_key: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or 'ds' not in frame.columns or value_key not in frame.columns:
        raise HistoryRefreshError(f'History rows must include ds and {value_key}.', code='INVALID_HISTORY')
    frame['ds'] = pd.to_datetime(frame['ds'])
    frame[value_key] = pd.to_numeric(frame[value_key], errors='coerce')
    return frame.sort_values('ds')


def validate_history_frame(frame: pd.DataFrame, *, label: str) -> pd.Series:
    if frame.empty:
        raise HistoryRefreshError(f'{label} history is empty.', code='INVALID_HISTORY')
    if frame['ds'].duplicated().any():
        raise HistoryRefreshError(f'{label} history contains duplicate dates.', code='INVALID_HISTORY')
    series = frame.set_index('ds')['y'].astype(float)
    expected = pd.date_range(series.index.min(), series.index.max(), freq='D')
    if not series.index.equals(expected):
        raise HistoryRefreshError(f'{label} history must be continuous daily observations.', code='INVALID_HISTORY')
    if not (series.map(lambda v: float(v) == float(v) and v >= 0).all()):
        raise HistoryRefreshError(f'{label} history must be finite and non-negative.', code='INVALID_HISTORY')
    if len(series) < MINIMUM_DAYS:
        raise HistoryRefreshError(
            f'{label} history requires at least {MINIMUM_DAYS} days.',
            code='INVALID_HISTORY',
            details={'days': int(len(series)), 'minimum_days': MINIMUM_DAYS},
        )
    return series


def validate_incoming_rows(rows: list[dict], *, frozen_bookings: list[dict], frozen_guests: list[dict]) -> dict:
    if not isinstance(rows, list) or not rows:
        raise HistoryRefreshError('History refresh requires a non-empty rows array.', code='INVALID_HISTORY')

    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise HistoryRefreshError(f'Row {index} must be an object.', code='INVALID_HISTORY')
        date_value = row.get('demand_date') or row.get('ds') or row.get('date')
        if not date_value:
            raise HistoryRefreshError(f'Row {index} is missing demand_date.', code='INVALID_HISTORY')
        try:
            ds = pd.Timestamp(date_value).strftime('%Y-%m-%d')
        except Exception as exc:
            raise HistoryRefreshError(f'Row {index} has an invalid date.', code='INVALID_HISTORY') from exc
        try:
            bookings = float(row.get('bookings', row.get('y_bookings')))
            guests = float(row.get('guests', row.get('y_guests', row.get('total_guests'))))
        except (TypeError, ValueError) as exc:
            raise HistoryRefreshError(f'Row {index} has non-numeric bookings/guests.', code='INVALID_HISTORY') from exc
        if bookings != bookings or guests != guests or bookings < 0 or guests < 0:
            raise HistoryRefreshError(f'Row {index} has invalid bookings/guests values.', code='INVALID_HISTORY')
        if guests < bookings:
            raise HistoryRefreshError(
                f'Row {index} guests must be greater than or equal to bookings.',
                code='INVALID_HISTORY',
            )
        normalized.append({'ds': ds, 'bookings': bookings, 'guests': guests})

    incoming = pd.DataFrame(normalized).drop_duplicates('ds', keep='last').sort_values('ds')
    incoming['ds'] = pd.to_datetime(incoming['ds'])
    booking_series = validate_history_frame(incoming.rename(columns={'bookings': 'y'})[['ds', 'y']], label='Bookings')
    guest_series = validate_history_frame(incoming.rename(columns={'guests': 'y'})[['ds', 'y']], label='Guests')

    frozen_b = validate_history_frame(_to_frame(frozen_bookings, 'y'), label='Frozen bookings')
    frozen_g = validate_history_frame(_to_frame(frozen_guests, 'y'), label='Frozen guests')
    if booking_series.index.min() != frozen_b.index.min():
        raise HistoryRefreshError(
            'Refreshed history must preserve the frozen history start date.',
            code='INVALID_HISTORY',
            details={'expected_start': str(frozen_b.index.min().date())},
        )
    # Prefix integrity: every frozen day must remain identical.
    for stamp, value in frozen_b.items():
        if stamp not in booking_series.index:
            raise HistoryRefreshError(
                'Refreshed history is missing frozen booking days.',
                code='INVALID_HISTORY',
                details={'missing_date': str(stamp.date())},
            )
        if float(booking_series.loc[stamp]) != float(value):
            raise HistoryRefreshError(
                'Refreshed history altered frozen booking values; refusing activation.',
                code='INVALID_HISTORY',
                details={'date': str(stamp.date())},
            )
    for stamp, value in frozen_g.items():
        if stamp not in guest_series.index:
            raise HistoryRefreshError(
                'Refreshed history is missing frozen guest days.',
                code='INVALID_HISTORY',
                details={'missing_date': str(stamp.date())},
            )
        if float(guest_series.loc[stamp]) != float(value):
            raise HistoryRefreshError(
                'Refreshed history altered frozen guest values; refusing activation.',
                code='INVALID_HISTORY',
                details={'date': str(stamp.date())},
            )

    if booking_series.index.max() < frozen_b.index.max():
        raise HistoryRefreshError('Refreshed history cannot shorten served history.', code='INVALID_HISTORY')

    booking_rows = [{'ds': stamp.strftime('%Y-%m-%d'), 'y': float(value)} for stamp, value in booking_series.items()]
    guest_rows = [{'ds': stamp.strftime('%Y-%m-%d'), 'y': float(value)} for stamp, value in guest_series.items()]
    return {
        'format_version': 1,
        'source': 'booking_demand_daily',
        'bookings_history': booking_rows,
        'guests_history': guest_rows,
        'date_from': booking_rows[0]['ds'],
        'date_to': booking_rows[-1]['ds'],
        'training_days': len(booking_rows),
        'evaluation_metrics_unchanged': True,
        'evaluation_scope_preserved': 'retrospective_rolling_28_origins',
        'model_fitted': False,
        'hyperparameters_changed': False,
        'automatic_v3_fallback': False,
    }


def apply_to_bundles(bookings: dict, guests: dict, overlay: dict | None = None) -> tuple[dict, dict]:
    payload = overlay if overlay is not None else load_active_overlay()
    if not payload:
        return bookings, guests
    out_b = copy.deepcopy(bookings)
    out_g = copy.deepcopy(guests)
    out_b['baseline_history'] = list(payload['bookings_history'])
    out_g['baseline_history'] = list(payload['guests_history'])
    # Keep frozen model types / hyperparameters untouched.
    return out_b, out_g


def apply_to_metadata(metadata: dict, overlay: dict | None = None) -> dict:
    payload = overlay if overlay is not None else load_active_overlay()
    result = copy.deepcopy(metadata)
    if not payload:
        result.setdefault('served_history_refreshed', False)
        return result
    result['date_from'] = payload['date_from']
    result['date_to'] = payload['date_to']
    result['training_days'] = payload['training_days']
    result['usable_days'] = payload['training_days']
    result['served_history_refreshed'] = True
    result['history_refresh'] = {
        'activated_at': payload.get('activated_at'),
        'source': payload.get('source'),
        'evaluation_metrics_unchanged': True,
        'evaluation_scope_preserved': payload.get('evaluation_scope_preserved'),
        'model_fitted': False,
        'hyperparameters_changed': False,
    }
    # Do not rewrite selected_metrics; callers keep frozen weekly evidence.
    return result


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def activate_overlay(payload: dict, *, validate_apply) -> dict:
    """Validate, snapshot previous active overlay, activate, or restore on failure."""
    with _lock:
        history_dir().mkdir(parents=True, exist_ok=True)
        previous = _read_json(_active_path())
        staging = copy.deepcopy(payload)
        staging['validated_at'] = datetime.now(timezone.utc).isoformat()
        _atomic_write(_staging_path(), staging)
        try:
            validate_apply(staging)
            staging['activated_at'] = datetime.now(timezone.utc).isoformat()
            if previous is not None:
                _atomic_write(_snapshot_path(), previous)
            elif _snapshot_path().exists():
                # Keep last known-good snapshot if present.
                pass
            else:
                # First activation: snapshot encodes "revert to frozen baseline".
                _atomic_write(_snapshot_path(), {
                    'format_version': 1,
                    'represents_frozen_baseline': True,
                    'saved_at': datetime.now(timezone.utc).isoformat(),
                })
            _atomic_write(_active_path(), staging)
            if _staging_path().exists():
                _staging_path().unlink()
            clear_overlay_cache()
            return staging
        except Exception:
            # Failed refresh must not leave a bad active overlay.
            if previous is not None:
                _atomic_write(_active_path(), previous)
            elif _active_path().exists():
                _active_path().unlink()
            if _staging_path().exists():
                _staging_path().unlink()
            clear_overlay_cache()
            raise


def restore_snapshot() -> dict:
    with _lock:
        snap = _read_json(_snapshot_path())
        if snap is None:
            raise HistoryRefreshError('No previous history snapshot is available.', code='HISTORY_SNAPSHOT_MISSING', status_code=409)
        if snap.get('represents_frozen_baseline'):
            if _active_path().exists():
                _active_path().unlink()
            clear_overlay_cache()
            return {'restored': 'frozen_baseline', 'active_overlay': False}
        _atomic_write(_active_path(), snap)
        clear_overlay_cache()
        return {'restored': 'snapshot_previous', 'active_overlay': True, 'date_to': snap.get('date_to')}


def status() -> dict:
    active = load_active_overlay(use_cache=False)
    snap = _read_json(_snapshot_path())
    return {
        'history_refresh_supported': True,
        'dataset_sync_refreshes_served_history': False,
        'active_overlay': bool(active),
        'history_as_of': (active or {}).get('date_to'),
        'training_days': (active or {}).get('training_days'),
        'snapshot_available': bool(snap),
        'evaluation_metrics_unchanged': True,
        'model_fitted': False,
    }
