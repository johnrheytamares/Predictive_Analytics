"""Production adapter around immutable v3.2 Sum-Ratio inference; never trains or falls back.

Authorized for production use. Rollback remains v3.1 (git 8892d29 /
RC reserVision_v3_1_two_stage_rc_20261124) until traffic is cut over.
"""
from pathlib import Path, PurePosixPath
import copy, hashlib, importlib.util, json, os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / 'experiments/reserVision_v3_2_sumratio_guests_rc_20261126'
MANIFEST = Path(__file__).with_name('release_manifest.json')
# Explicit authorization granted 2026-11-27.
PRODUCTION_APPROVED = True
DEPLOYMENT_SCOPE = 'production-authorized-release'
ROLLBACK_COMMIT = '8892d292d267cf870eee2657be639e74014c0dc1'
ROLLBACK_RC = 'reserVision_v3_1_two_stage_rc_20261124'

# Keep v3.2 history overlay separate from active v3.1 runtime_history.
os.environ.setdefault(
    'PREDICTION_HISTORY_DIR',
    str((Path(__file__).resolve().parent / 'runtime_history').resolve()),
)

from production_v31 import history_refresh  # noqa: E402

spec = importlib.util.spec_from_file_location('frozen_v32_engine', RC / 'shadow_predict.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
ForecastError = engine.serving.ForecastError


def weekly_performance() -> dict:
    """Serve frozen evaluation evidence only; never recompute or retune."""
    verification = json.loads((RC / 'verification_results.json').read_text(encoding='utf-8'))
    h7 = next(row for row in verification['horizon_pooled'] if int(row['horizon']) == 7)
    bookings = {
        'wape': float(h7['booking_wape_pct']),
        'mae': float(h7['booking_mae']),
        'rmse': float(h7['booking_rmse']),
        'bias': float(h7['booking_bias']),
        'n': int(h7['origins']),
    }
    guests = {
        'wape': float(h7['guest_wape_pct']),
        'mae': float(h7['guest_mae']),
        'rmse': float(h7['guest_rmse']),
        'bias': float(h7['guest_bias']),
        'n': int(h7['origins']),
        'interval_coverage_week_total': float(h7['interval_coverage']),
        'interval_coverage_daily': float(verification['interval']['daily_coverage']),
        'interval_method': verification['interval']['method'],
        'interval_observations': int(verification['interval']['observations']),
        'interval_disclaimer': (
            'Observed interval coverage only; not a validated confidence level.'
        ),
    }
    return {
        'bookings': {
            'raw': dict(bookings),
            'operational_floor_round': dict(bookings),
            'selected': dict(bookings),
        },
        'guests': {
            'raw': dict(guests),
            'operational_floor_round': dict(guests),
            'selected': dict(guests),
        },
        'evaluation_scope': 'retrospective_rolling_28_origins',
        'evaluation_horizon_days': 7,
        'source': 'experiments/reserVision_v3_2_sumratio_guests_rc_20261126/verification_results.json',
        'long_horizon_accuracy_evaluated': True,
        'horizon_pooled': verification['horizon_pooled'],
        'interval': verification['interval'],
        'sample_2027': verification['sample_2027'],
        'primary_metrics': ['mae', 'rmse'],
        'supplementary_metrics': ['wape', 'bias'],
        'interval_disclaimer': guests['interval_disclaimer'],
    }


def validate_release():
    try:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if manifest.get('production_approved') is not True:
            raise ValueError('Manifest must record production_approved=true for this authorized release')
        for name, expected in manifest['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid manifest path')
            path = ROOT.joinpath(*relative.parts)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(name)
        snapshot = json.loads((RC / 'serving_snapshot.json').read_text(encoding='utf-8'))
        verification = json.loads((RC / 'verification_results.json').read_text(encoding='utf-8'))
        sample = verification.get('sample_2027') or {}
        if int(sample.get('bookings') or -1) != 12 or int(sample.get('guests') or -1) != 108:
            raise ValueError('Sample gate 12 bookings / 108 guests failed')
        h7 = next(row for row in verification['horizon_pooled'] if int(row['horizon']) == 7)
        if round(float(h7['booking_mae']), 2) != 2.54 or round(float(h7['booking_rmse']), 2) != 3.87:
            raise ValueError('Booking MAE/RMSE gate failed')
        if round(float(h7['guest_mae']), 2) != 29.18 or round(float(h7['guest_rmse']), 2) != 40.94:
            raise ValueError('Guest MAE/RMSE gate failed')
        for name in manifest['artifact_names']:
            selected = engine.ARTIFACTS / name
            expected = manifest['files'][(RC / 'artifacts' / name).relative_to(ROOT).as_posix()]
            if not selected.is_file() or hashlib.sha256(selected.read_bytes()).hexdigest() != expected:
                raise ValueError('Selected artifact directory is not this release')
        if manifest.get('rollback_commit') != ROLLBACK_COMMIT:
            raise ValueError('Rollback commit reference mismatch')
        return snapshot
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise ForecastError(
            'MODEL_CONFIGURATION_MISMATCH',
            'Frozen v3.2 release integrity/configuration failed; no fallback is allowed.',
            status_code=422,
        ) from exc


engine.ARTIFACTS = Path(os.getenv('PREDICTION_ARTIFACT_DIR') or RC / 'artifacts').resolve()
engine.validate_integrity = validate_release

_original_get_bundles = engine.get_bundles
_original_load_metadata = engine.load_metadata


def get_bundles(*, use_cache: bool = True):
    bookings, guests = _original_get_bundles(use_cache=use_cache)
    return history_refresh.apply_to_bundles(bookings, guests)


def load_metadata(*, use_cache: bool = True):
    return history_refresh.apply_to_metadata(_original_load_metadata(use_cache=use_cache))


engine.get_bundles = get_bundles
engine.load_metadata = load_metadata


def metadata():
    original = engine.validate_readiness(_original_load_metadata())
    bookings, guests = get_bundles()
    engine.validate_v32_configuration(bookings, guests, original)
    result = history_refresh.apply_to_metadata(copy.deepcopy(original))
    performance = weekly_performance()
    for target in ('bookings', 'guests'):
        result['targets'][target]['raw_weekly_metrics'] = performance[target]['raw']
        result['targets'][target]['operational_weekly_metrics'] = performance[target]['operational_floor_round']
        result['targets'][target]['selected_metrics'] = performance[target]['operational_floor_round']
    guests_metrics = result['targets']['guests']
    guests_metrics['interval_disclaimer'] = performance['interval_disclaimer']
    guests_metrics['interval_coverage_evaluated'] = True
    overlay = history_refresh.load_active_overlay()
    result.update(
        serving_mode='production_prepared',
        shadow_serving_only=False,
        deployment_scope=DEPLOYMENT_SCOPE,
        production_approved=PRODUCTION_APPROVED,
        frozen_history=overlay is None,
        history_refresh_supported=True,
        original_evaluation_metadata_preserved=True,
        frozen_model_configuration=True,
        frozen_evaluation_metrics=True,
        prospective_accuracy_validated=False,
        rollback_commit=ROLLBACK_COMMIT,
        rollback_release_candidate_id=ROLLBACK_RC,
        primary_metrics=['mae', 'rmse'],
        supplementary_metrics=['wape', 'bias'],
    )
    result['lifecycle'] = lifecycle(result, overlay=overlay)
    return result, performance


def lifecycle(meta, *, overlay=None):
    active = overlay if overlay is not None else history_refresh.load_active_overlay()
    history_as_of = meta.get('date_to')
    return dict(
        history_as_of=history_as_of,
        dataset_sync_refreshes_served_history=False,
        history_refresh_supported=True,
        served_history_refreshed=bool(active),
        history_refresh_mode='validated_overlay',
        automatic_retraining=False,
        model_fitted=False,
        runtime_optimization=False,
        automatic_v3_fallback=False,
        prospective_accuracy_validated=False,
        evaluation_metrics_refreshed=False,
        evaluation_scope='retrospective_rolling_28_origins',
        refresh_policy=(
            'Validated booking_demand_daily rows may refresh served baseline_history via '
            'POST /history/refresh without retraining hyperparameters. Retrospective '
            'weekly metrics remain the frozen evaluation evidence and are not newly validated. '
            'Guest intervals disclose observed coverage only; not a validated confidence level.'
        ),
        evaluation_horizon_days=7,
        long_horizon_accuracy_evaluated=True,
        production_approved=PRODUCTION_APPROVED,
        activation_disabled=False,
        rollback_commit=ROLLBACK_COMMIT,
    )


def health():
    payload = dict(status='ok', service='predictive-analytics', model_ready=False, metadata=None)
    try:
        meta, performance = metadata()
        _, guests = get_bundles()
        cutoff = max(
            pd.Timestamp.now(tz='Asia/Manila').tz_localize(None).normalize(),
            pd.Timestamp(meta['date_to']) + pd.Timedelta(days=1),
        )
        caps = engine.shadow_capabilities(guests, cutoff)
        caps.update(
            serving_mode='production_prepared',
            runtime_optimization=False,
            history_as_of=meta['date_to'],
            history_through=meta['date_to'],
            maximum_days_per_cutoff=90,
            latest_supported_date=(caps.get('supported_dates') or [None])[-1],
        )
        payload.update(
            model_ready=bool(caps['supported_dates']),
            metadata=meta,
            forecast_capabilities=caps,
            weekly_performance=performance,
            evaluation_scope=meta.get('evaluation_scope') or performance.get('evaluation_scope'),
            prospective_validation=meta.get('prospective_validation'),
            lifecycle=meta['lifecycle'],
            production_approved=PRODUCTION_APPROVED,
            history_status=history_refresh.status(),
        )
    except Exception as exc:
        payload.update(
            code=getattr(exc, 'code', 'MODEL_LOAD_FAILED'),
            message='Version 3.2 is not ready; fallback is disabled.',
        )
    return payload


def forecast(start_date, end_date, *, allow_historical=False):
    payload = engine.run_forecast(start_date, end_date, allow_historical=allow_historical)
    meta, performance = metadata()
    payload['model_metadata'] = meta
    payload['model'].update(
        targets=meta['targets'],
        serving_mode='production_prepared',
        shadow_serving_only=False,
        production_approved=PRODUCTION_APPROVED,
        deployment_scope=meta['deployment_scope'],
        lifecycle=meta['lifecycle'],
        frozen_evaluation_metrics=True,
        prospective_accuracy_validated=False,
        served_history_refreshed=meta.get('served_history_refreshed', False),
        baseline_note=(
            'Prepared V3.2: Two-Stage bookings preserved from v3.1; '
            'Historical Guests-per-Booking Ratio (sum-ratio, past-84-day Σg/Σb); '
            'floor_round; period-preserving guest demographics. '
            'Guest intervals use past-only party-size ratio quantiles (q10–q90) with '
            'disclosed observed coverage only — not a validated confidence level. '
            'Production-approved. Rollback: v3.1 commit '
            f'{ROLLBACK_COMMIT[:7]} / {ROLLBACK_RC}.'
        ),
        primary_metrics=['mae', 'rmse'],
        supplementary_metrics=['wape', 'bias'],
    )
    start = pd.Timestamp(start_date)
    _, guests = get_bundles()
    caps = engine.shadow_capabilities(guests, start)
    caps.update(
        serving_mode='production_prepared',
        runtime_optimization=False,
        history_as_of=meta['date_to'],
        history_through=meta['date_to'],
        maximum_days_per_cutoff=90,
        cutoff=str(start.date()),
        latest_supported_date=(caps.get('supported_dates') or [None])[-1],
    )
    payload.update(forecast_capabilities=caps, weekly_performance=performance)
    return payload


def _validate_overlay_for_serving(overlay: dict) -> None:
    engine.clear_cache()
    history_refresh.clear_overlay_cache()
    original = engine.validate_readiness(_original_load_metadata(use_cache=False))
    bookings, guests = history_refresh.apply_to_bundles(
        *_original_get_bundles(use_cache=False), overlay
    )
    engine.validate_v32_configuration(bookings, guests, original)
    meta = history_refresh.apply_to_metadata(copy.deepcopy(original), overlay)
    cutoff = pd.Timestamp(meta['date_to']) + pd.Timedelta(days=1)
    caps = engine.shadow_capabilities(guests, cutoff)
    if not caps.get('supported_dates'):
        raise history_refresh.HistoryRefreshError(
            'Refreshed history did not produce supported forecast dates.',
            code='HISTORY_REFRESH_NOT_READY',
        )


def refresh_history(*, rows, source='booking_demand_daily'):
    try:
        validate_release()
        frozen_bookings, frozen_guests = history_refresh.frozen_baseline_rows(
            *_original_get_bundles(use_cache=False)
        )
        payload = history_refresh.validate_incoming_rows(
            rows,
            frozen_bookings=frozen_bookings,
            frozen_guests=frozen_guests,
        )
        payload['source'] = source or 'booking_demand_daily'
        activated = history_refresh.activate_overlay(payload, validate_apply=_validate_overlay_for_serving)
        engine.clear_cache()
        history_refresh.clear_overlay_cache()
        ready = health()
        return {
            'success': True,
            'history_refreshed': True,
            'model_retrained': False,
            'model_fitted': False,
            'database_synchronized': False,
            'hyperparameters_changed': False,
            'evaluation_metrics_unchanged': True,
            'evaluation_scope': 'retrospective_rolling_28_origins',
            'history_as_of': activated['date_to'],
            'training_days': activated['training_days'],
            'date_from': activated['date_from'],
            'production_approved': PRODUCTION_APPROVED,
            'automatic_v3_fallback': False,
            'model_ready': ready.get('model_ready') is True,
            'message': (
                'Served history refreshed from validated demand rows. '
                'Frozen Two-Stage / Sum-Ratio / floor_round configuration was not refit. '
                'Retrospective weekly metrics remain the original evaluation evidence.'
            ),
            'lifecycle': ready.get('lifecycle'),
            'weekly_performance': ready.get('weekly_performance'),
        }
    except history_refresh.HistoryRefreshError as exc:
        engine.clear_cache()
        history_refresh.clear_overlay_cache()
        raise ForecastError(
            exc.code, exc.message, status_code=exc.status_code, extra=exc.details or None
        ) from exc
    except ForecastError:
        engine.clear_cache()
        history_refresh.clear_overlay_cache()
        raise


def restore_history_snapshot():
    restored = history_refresh.restore_snapshot()
    engine.clear_cache()
    history_refresh.clear_overlay_cache()
    ready = health()
    return {
        'success': True,
        'history_refreshed': False,
        'history_restored': True,
        'model_retrained': False,
        'restore': restored,
        'model_ready': ready.get('model_ready') is True,
        'lifecycle': ready.get('lifecycle'),
        'message': 'Previous served-history snapshot restored. Frozen evaluation metrics unchanged.',
    }
