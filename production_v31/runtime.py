"""Production adapter around immutable v3.1 inference; never trains or falls back."""
from pathlib import Path, PurePosixPath
import copy, hashlib, importlib.util, json, os
import pandas as pd
from . import history_refresh

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / 'experiments/reserVision_v3_1_two_stage_rc_20261124'
MANIFEST = Path(__file__).with_name('release_manifest.json')
# Release authorization: retrospective weekly metrics remain the evaluation evidence.
# Prospective accuracy is still not newly validated by this flag.
PRODUCTION_APPROVED = True
DEPLOYMENT_SCOPE = 'production-authorized-release'

spec = importlib.util.spec_from_file_location('frozen_v31_engine', RC / 'shadow_predict.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
ForecastError = engine.serving.ForecastError

def validate_release():
    try:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        for name, expected in manifest['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('Invalid manifest path')
            path = ROOT.joinpath(*relative.parts)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(name)
        snapshot = json.loads((RC / 'serving_snapshot.json').read_text(encoding='utf-8'))
        recorded = json.loads((RC / 'verification_horizons_postprocess.json').read_text(encoding='utf-8'))
        if any(row['cfg'] != snapshot['forward_parameters'] for row in recorded['horizons'].values()):
            raise ValueError('Forward parameters changed')
        # PREDICTION_ARTIFACT_DIR keeps its existing meaning, but may only select
        # byte-identical v3.1 artifacts; a v3.0 mount cannot be silently accepted.
        for name in manifest['artifact_names']:
            selected = engine.ARTIFACTS / name
            expected = manifest['files'][(RC / 'artifacts' / name).relative_to(ROOT).as_posix()]
            if not selected.is_file() or hashlib.sha256(selected.read_bytes()).hexdigest() != expected:
                raise ValueError('Selected artifact directory is not this release')
        return snapshot
    except (OSError, ValueError, KeyError) as exc:
        raise ForecastError('MODEL_CONFIGURATION_MISMATCH',
            'Frozen v3.1 release integrity/configuration failed; no fallback is allowed.',status_code=422) from exc

engine.ARTIFACTS = Path(os.getenv('PREDICTION_ARTIFACT_DIR') or RC / 'artifacts').resolve()
# The frozen inference code is reused unchanged. Replace only its workspace-wide
# Windows integrity adapter with this portable, explicit release dependency set.
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
    engine.validate_v31_configuration(bookings, guests, original)
    result = history_refresh.apply_to_metadata(copy.deepcopy(original))
    performance = engine.weekly_performance()  # Arithmetic on saved evidence only.
    for target in ('bookings', 'guests'):
        result['targets'][target]['raw_weekly_metrics'] = performance[target]['raw']
        result['targets'][target]['operational_weekly_metrics'] = performance[target]['operational_floor_round']
        result['targets'][target]['selected_metrics'] = performance[target]['operational_floor_round']
    overlay = history_refresh.load_active_overlay()
    result.update(serving_mode='production_prepared', shadow_serving_only=False,
        deployment_scope=DEPLOYMENT_SCOPE, production_approved=PRODUCTION_APPROVED,
        frozen_history=overlay is None, history_refresh_supported=True,
        original_evaluation_metadata_preserved=True,
        frozen_model_configuration=True,
        frozen_evaluation_metrics=True,
        prospective_accuracy_validated=False)
    result['lifecycle'] = lifecycle(result, overlay=overlay)
    return result, performance

def lifecycle(meta, *, overlay=None):
    active = overlay if overlay is not None else history_refresh.load_active_overlay()
    history_as_of = meta.get('date_to')
    return dict(
        history_as_of=history_as_of,
        # DB sync alone still does not refresh; Node must call POST /history/refresh.
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
            'weekly metrics remain the frozen evaluation evidence and are not newly validated.'
        ),
        evaluation_horizon_days=7,
        long_horizon_accuracy_evaluated=False,
    )

def health():
    payload = dict(status='ok',service='predictive-analytics',model_ready=False,metadata=None)
    try:
        meta, performance = metadata()
        _, guests = get_bundles()
        # Capabilities are anchored to today's requested serving context, not a
        # stale 90-day list starting at the original November 22 cutoff.
        cutoff = max(pd.Timestamp.now(tz='Asia/Manila').tz_localize(None).normalize(),
                     pd.Timestamp(meta['date_to']) + pd.Timedelta(days=1))
        caps = engine.shadow_capabilities(guests,cutoff)
        caps.update(serving_mode='production_prepared',runtime_optimization=False,
                    history_as_of=meta['date_to'])
        payload.update(model_ready=bool(caps['supported_dates']),metadata=meta,
            forecast_capabilities=caps,weekly_performance=performance,
            evaluation_scope=meta.get('evaluation_scope') or performance.get('evaluation_scope'),
            prospective_validation=meta.get('prospective_validation'),
            lifecycle=meta['lifecycle'],production_approved=PRODUCTION_APPROVED,
            history_status=history_refresh.status())
    except Exception as exc:
        payload.update(code=getattr(exc,'code','MODEL_LOAD_FAILED'),
                       message='Version 3.1 is not ready; fallback is disabled.')
    return payload

def forecast(start_date,end_date,*,allow_historical=False,compare_variant_a=False):
    payload = engine.run_forecast(start_date,end_date,allow_historical=allow_historical,
                                  include_variant_a_comparison=compare_variant_a)
    meta, performance = metadata()
    payload['model_metadata'] = meta
    payload['model'].update(targets=meta['targets'],serving_mode='production_prepared',
        shadow_serving_only=False,production_approved=PRODUCTION_APPROVED,
        deployment_scope=meta['deployment_scope'],
        lifecycle=meta['lifecycle'], frozen_evaluation_metrics=True,
        prospective_accuracy_validated=False,
        served_history_refreshed=meta.get('served_history_refreshed', False))
    start = pd.Timestamp(start_date)
    _, guests = get_bundles()
    caps = engine.shadow_capabilities(guests,start)
    caps.update(serving_mode='production_prepared',runtime_optimization=False,history_as_of=meta['date_to'])
    payload.update(forecast_capabilities=caps,weekly_performance=performance)
    return payload


def _validate_overlay_for_serving(overlay: dict) -> None:
    """Ensure refreshed history still produces a ready v3.1 configuration."""
    engine.clear_cache()
    history_refresh.clear_overlay_cache()
    original = engine.validate_readiness(_original_load_metadata(use_cache=False))
    bookings, guests = history_refresh.apply_to_bundles(*_original_get_bundles(use_cache=False), overlay)
    engine.validate_v31_configuration(bookings, guests, original)
    # Continuous history + champion checks already done; require forward capability.
    meta = history_refresh.apply_to_metadata(copy.deepcopy(original), overlay)
    cutoff = pd.Timestamp(meta['date_to']) + pd.Timedelta(days=1)
    caps = engine.shadow_capabilities(guests, cutoff)
    if not caps.get('supported_dates'):
        raise history_refresh.HistoryRefreshError(
            'Refreshed history did not produce supported forecast dates.',
            code='HISTORY_REFRESH_NOT_READY',
        )


def refresh_history(*, rows, source='booking_demand_daily'):
    """Validate incoming demand history, activate overlay, or restore prior snapshot."""
    try:
        validate_release()
        frozen_bookings, frozen_guests = history_refresh.frozen_baseline_rows(*_original_get_bundles(use_cache=False))
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
                'Frozen Two-Stage / Robust Median / floor_round configuration was not refit. '
                'Retrospective weekly metrics remain the original evaluation evidence.'
            ),
            'lifecycle': ready.get('lifecycle'),
            'weekly_performance': ready.get('weekly_performance'),
        }
    except history_refresh.HistoryRefreshError as exc:
        engine.clear_cache()
        history_refresh.clear_overlay_cache()
        raise ForecastError(exc.code, exc.message, status_code=exc.status_code, extra=exc.details or None) from exc
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
