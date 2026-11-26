"""Isolated V3.2 release candidate: two_stage bookings + guests_per_booking_sumratio.

Does not replace production v3.1. Shadow / RC only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import predict_real as serving  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "v32_intermittent",
    ROOT / "experiments/intermittent_bookings_20261124/experiment.py",
)
intermittent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(intermittent)

_ops_spec = importlib.util.spec_from_file_location("v32_ops", HERE / "artifacts" / "ops.py")
ops = importlib.util.module_from_spec(_ops_spec)
_ops_spec.loader.exec_module(ops)

ARTIFACTS = HERE / "artifacts"
ROLLBACK_V31_GUESTS = HERE / "package" / "rollback" / "v31_guests_model.joblib"
GUESTS_PER_BOOKING_SUMRATIO = "guests_per_booking_sumratio"
LOOKBACK_DAYS = 84

_lock = threading.Lock()
_cached_metadata = None
_cached_bookings = None
_cached_guests = None
_cached_v31_guests = None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_paths() -> dict:
    return {
        "bookings": ARTIFACTS / "bookings_model.joblib",
        "guests": ARTIFACTS / "guests_model.joblib",
        "metadata": ARTIFACTS / "forecast_models_metadata.json",
    }


def validate_integrity() -> dict:
    snapshot = json.loads((HERE / "serving_snapshot.json").read_text(encoding="utf-8"))
    for name, expected in snapshot["protected_files"].items():
        path = ROOT / name
        if not path.is_file() or digest(path) != expected:
            raise serving.ForecastError(
                "MODEL_CONFIGURATION_MISMATCH",
                f"V3.2 RC integrity failed for {name}.",
                status_code=422,
            )
    return snapshot


def load_metadata(*, use_cache: bool = True) -> dict:
    global _cached_metadata
    validate_integrity()
    with _lock:
        if use_cache and _cached_metadata is not None:
            return _cached_metadata
        path = artifact_paths()["metadata"]
        if not path.exists():
            raise serving.ForecastError("METADATA_MISSING", "V3.2 RC metadata missing.", status_code=404)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if use_cache:
            _cached_metadata = metadata
        return metadata


def get_bundles(*, use_cache: bool = True):
    global _cached_bookings, _cached_guests
    validate_integrity()
    with _lock:
        if use_cache and _cached_bookings is not None and _cached_guests is not None:
            return _cached_bookings, _cached_guests
        paths = artifact_paths()
        bookings = joblib.load(paths["bookings"])
        guests = joblib.load(paths["guests"])
        if use_cache:
            _cached_bookings, _cached_guests = bookings, guests
        return bookings, guests


def get_v31_guests_aux(*, use_cache: bool = True):
    """v3.1 robust-median guests used only to keep floor_round booking integers identical."""
    global _cached_v31_guests
    with _lock:
        if use_cache and _cached_v31_guests is not None:
            return _cached_v31_guests
        if not ROLLBACK_V31_GUESTS.is_file():
            raise serving.ForecastError(
                "MODEL_MISSING",
                "v3.1 rollback guests artifact missing for booking-identity floor_round.",
                status_code=404,
            )
        bundle = joblib.load(ROLLBACK_V31_GUESTS)
        if bundle.get("model_type") != serving.ROBUST_MEDIAN_GUESTS:
            raise serving.ForecastError(
                "MODEL_CONFIGURATION_MISMATCH",
                "Rollback guests artifact must be robust_median_guests.",
                status_code=422,
            )
        if use_cache:
            _cached_v31_guests = bundle
        return bundle


def clear_cache() -> None:
    global _cached_metadata, _cached_bookings, _cached_guests, _cached_v31_guests
    with _lock:
        _cached_metadata = None
        _cached_bookings = None
        _cached_guests = None
        _cached_v31_guests = None


def validate_v32_configuration(bookings, guests, metadata: dict) -> None:
    if bookings.get("model_type") != "two_stage":
        raise serving.ForecastError("MODEL_CONFIGURATION_MISMATCH", "V3.2 bookings must be two_stage.", status_code=422)
    if guests.get("model_type") != GUESTS_PER_BOOKING_SUMRATIO:
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "V3.2 guests must be guests_per_booking_sumratio.",
            status_code=422,
        )
    targets = metadata.get("targets") or {}
    if (targets.get("bookings") or {}).get("selected_model") != "two_stage":
        raise serving.ForecastError("MODEL_CONFIGURATION_MISMATCH", "Metadata bookings must be two_stage.", status_code=422)
    if (targets.get("guests") or {}).get("selected_model") != GUESTS_PER_BOOKING_SUMRATIO:
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "Metadata guests must be guests_per_booking_sumratio.",
            status_code=422,
        )
    if (metadata.get("postprocess") or {}).get("selected_mode") != "floor_round":
        raise serving.ForecastError("MODEL_CONFIGURATION_MISMATCH", "Postprocess must be floor_round.", status_code=422)
    if metadata.get("shadow_serving_only") is not True or metadata.get("production_approved") is True:
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "V3.2 RC must remain shadow_serving_only with production_approved=false.",
            status_code=422,
        )
    if int(guests.get("lookback_days") or 0) != LOOKBACK_DAYS:
        raise serving.ForecastError("MODEL_CONFIGURATION_MISMATCH", "Sum-ratio lookback must be 84 days.", status_code=422)


def history_frame(bundle) -> pd.DataFrame:
    history = pd.DataFrame(bundle.get("baseline_history") or [])
    if history.empty or not {"ds", "y"}.issubset(history.columns):
        raise serving.ForecastError("MODEL_INVALID", "Bundle missing baseline_history.", status_code=422)
    history = history.copy()
    history["ds"] = pd.to_datetime(history["ds"])
    history = history.set_index("ds").sort_index()
    if history.index.has_duplicates:
        raise serving.ForecastError("MODEL_INVALID", "Duplicate history dates.", status_code=422)
    values = history["y"].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise serving.ForecastError("MODEL_INVALID", "History must be finite nonnegative.", status_code=422)
    expected = pd.date_range(history.index.min(), history.index.max(), freq="D")
    if not history.index.equals(expected):
        raise serving.ForecastError("MODEL_INVALID", "History must be continuous daily.", status_code=422)
    return history


def combined_history(bookings_bundle, guests_bundle, cutoff: pd.Timestamp) -> pd.DataFrame:
    b = history_frame(bookings_bundle)
    g = history_frame(guests_bundle)
    if not b.index.equals(g.index):
        raise serving.ForecastError("MODEL_INVALID", "Booking and guest histories must share dates.", status_code=422)
    frame = pd.DataFrame({"bookings": b["y"].astype(float), "guests": g["y"].astype(float)}, index=b.index)
    frame = frame.loc[frame.index < cutoff]
    if len(frame) < 120:
        raise serving.ForecastError("MODEL_NOT_READY", "Insufficient history before cutoff.", status_code=422)
    return frame


def sumratio_stats(train: pd.DataFrame, lookback: int = LOOKBACK_DAYS) -> dict:
    """Past-only party-size stats from positive-booking days in the lookback window."""
    window = train.tail(lookback)
    positive = window[window.bookings > 0]
    if positive.empty:
        return dict(ratio=0.0, q10=0.0, q50=0.0, q90=0.0, n_positive=0)
    ratios = (positive.guests / positive.bookings).to_numpy(float)
    return dict(
        ratio=float(positive.guests.sum() / positive.bookings.sum()),
        q10=float(np.quantile(ratios, 0.10)),
        q50=float(np.quantile(ratios, 0.50)),
        q90=float(np.quantile(ratios, 0.90)),
        n_positive=int(len(positive)),
    )


def predict_sumratio_guests(
    train: pd.DataFrame,
    operational_bookings: np.ndarray,
    stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point forecast and sumratio-specific ratio-quantile intervals.

    Intervals use past-only q10/q90 of guests/booking — not robust-median margins.
    Zero predicted bookings => zero guests and zero-width intervals.
    """
    stats = stats or sumratio_stats(train)
    ratio = max(0.0, float(stats["ratio"]))
    q10 = max(0.0, float(stats["q10"]))
    q90 = max(0.0, float(stats["q90"]))
    b = np.asarray(operational_bookings, float)
    raw = b * ratio
    lower_raw = b * q10
    upper_raw = b * q90
    # Operational consistency without party-size floor: guests >= bookings when b>0.
    point = np.where(b > 0, np.maximum(np.floor(np.maximum(raw, 0.0)), b), 0.0)
    lower = np.where(b > 0, np.maximum(np.floor(np.maximum(lower_raw, 0.0)), b), 0.0)
    upper = np.where(b > 0, np.maximum(np.ceil(np.maximum(upper_raw, 0.0)), point), 0.0)
    # Ensure lower <= point <= upper
    lower = np.minimum(lower, point)
    upper = np.maximum(upper, point)
    return point.astype(float), lower.astype(float), upper.astype(float)


def two_stage_raw_bookings(bookings_bundle, dates: pd.DatetimeIndex) -> tuple[np.ndarray, dict]:
    cutoff = dates.min()
    train_b = history_frame(bookings_bundle)["y"]
    train_b = train_b[train_b.index < cutoff]
    if len(train_b) < 120:
        raise serving.ForecastError("MODEL_NOT_READY", "Insufficient booking history.", status_code=422)
    snapshot = validate_integrity()
    if cutoff >= pd.Timestamp(snapshot["forward_cutoff"]):
        cfg = snapshot["forward_parameters"]
    else:
        freeze = json.loads((ARTIFACTS / "frozen_selection.json").read_text(encoding="utf-8"))
        saved = [
            r["parameters"]
            for r in freeze["per_origin_parameters"]
            if r["protocol"] == "rolling_7day" and r["origin"] == str(cutoff.date())
        ]
        if not saved:
            raise serving.ForecastError(
                "FROZEN_PARAMETERS_UNAVAILABLE",
                "No frozen origin parameters for this start date.",
                status_code=422,
            )
        cfg = saved[0]
    raw_b = intermittent.predict(train_b, dates, "two_stage", cfg)
    if len(raw_b) != len(dates) or not np.isfinite(raw_b).all():
        raise serving.ForecastError("MODEL_INVALID", "Invalid two_stage booking predictions.", status_code=422)
    return np.asarray(raw_b, float), cfg


def robust_median_guest_raw(guest_history: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Past-only robust median guest totals (same rule as v3.1 serving helper)."""
    g = guest_history.to_numpy(float)
    dow = guest_history.index.dayofweek.to_numpy()
    out = []
    for day in dates:
        same = g[dow == day.dayofweek][-4:]
        out.append(max(0.0, round(float(np.median(same))) if len(same) else 0.0))
    return np.asarray(out, float)


def operational_bookings_matching_v31(
    raw_b: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    guest_history_before: pd.Series | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Reproduce v3.1 floor_round booking integers.

    Robust-median guests are used only inside floor_round consistency so booking
    integers match v3.1. Served guest totals still come from sumratio.
    """
    if guest_history_before is None:
        aux = get_v31_guests_aux(use_cache=use_cache)
        guest_rows = serving.predict_series(aux, dates)
        raw_g = np.asarray([float(r["prediction"]) for r in guest_rows], float)
    else:
        raw_g = robust_median_guest_raw(guest_history_before, dates)
    op_b, _op_g = ops.apply_ops(raw_b, raw_g, mode="floor_round")
    return np.asarray(op_b, float)


def shadow_capabilities(guests_bundle, cutoff: pd.Timestamp) -> dict:
    history = history_frame(guests_bundle)
    usable = history.loc[history.index < cutoff]
    supported = []
    day = pd.Timestamp(cutoff)
    for _ in range(90):
        if len(usable) < 120:
            break
        supported.append(str(day.date()))
        day += pd.Timedelta(days=1)
    return {
        "bookings_method": "two_stage",
        "guests_method": GUESTS_PER_BOOKING_SUMRATIO,
        "postprocess": "floor_round",
        "party_size_floor": False,
        "supported_horizons_days": [7, 30, 90],
        "supported_dates": supported,
        "serving_mode": "isolated_shadow",
        "guest_interval_method": "past_only_party_size_ratio_quantiles_q10_q90",
    }


def validate_readiness(metadata: dict | None) -> dict:
    if not metadata:
        raise serving.ForecastError("METADATA_MISSING", "Required prediction model is missing.", status_code=404)
    if metadata.get("is_demo") is True:
        raise serving.ForecastError("DEMO_MODEL", "Demo model is not allowed.", status_code=422)
    if metadata.get("ready") is not True:
        raise serving.ForecastError("MODEL_NOT_READY", "RC metadata is not ready.", status_code=422)
    if metadata.get("safe_for_live_use") is not True:
        raise serving.ForecastError("MODEL_NOT_SAFE_FOR_LIVE_USE", "RC not marked safe_for_live_use.", status_code=422)
    if metadata.get("production_approved") is True:
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "V3.2 RC must remain production_approved=false.",
            status_code=422,
        )
    return metadata


def generate_shadow_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    metadata: dict,
    *,
    use_cache: bool = True,
) -> dict:
    bookings, guests = get_bundles(use_cache=use_cache)
    validate_v32_configuration(bookings, guests, metadata)

    dates = pd.date_range(start=start, end=end, freq="D")
    raw_b, cfg = two_stage_raw_bookings(bookings, dates)
    train = combined_history(bookings, guests, dates.min())
    # Keep booking integers identical to v3.1 floor_round (robust-median only for consistency).
    # Forward ranges after history end use the frozen aux bundle; historical cutoffs use past-only series.
    history_end = pd.Timestamp(metadata.get("date_to"))
    if dates.min() <= history_end:
        op_b = operational_bookings_matching_v31(
            raw_b, dates, guest_history_before=train["guests"], use_cache=use_cache
        )
    else:
        op_b = operational_bookings_matching_v31(raw_b, dates, use_cache=use_cache)
    stats = sumratio_stats(train)
    point, lower, upper = predict_sumratio_guests(train, op_b, stats)

    forecasts = []
    for i, ds in enumerate(dates):
        predicted_bookings = int(op_b[i])
        predicted_guests = int(point[i])
        # Zero-demand: both zero. Positive bookings never yield zero guests after consistency.
        if predicted_bookings == 0 and predicted_guests != 0:
            raise serving.ForecastError("MODEL_INVALID", "Zero bookings with positive guests.", status_code=500)
        if predicted_guests < predicted_bookings:
            raise serving.ForecastError("MODEL_INVALID", "Guests below bookings after sumratio ops.", status_code=500)
        if predicted_guests > 0 and predicted_bookings < 1:
            raise serving.ForecastError("MODEL_INVALID", "Positive guests with zero bookings.", status_code=500)

        bookings_lower, bookings_upper = serving.reconcile_bounds(
            predicted_bookings,
            serving.clamp_int(max(0.0, float(raw_b[i]) - 1.0)),
            serving.clamp_int(float(raw_b[i]) + 1.0),
        )
        guests_lower, guests_upper = serving.reconcile_bounds(
            predicted_guests, int(lower[i]), int(upper[i])
        )
        forecasts.append(
            {
                "date": ds.strftime("%Y-%m-%d"),
                "predicted_bookings": predicted_bookings,
                "bookings_lower": bookings_lower,
                "bookings_upper": bookings_upper,
                "predicted_total_guests": predicted_guests,
                "guests_lower": guests_lower,
                "guests_upper": guests_upper,
                "estimated_guest_mix": None,
                "guest_mix_ready": metadata.get("guest_mix_ready") is True,
            }
        )

    serving.apply_period_guest_mix(forecasts, metadata)

    # Demographic consistency: daily mix sums to daily guests; period mix matches aggregate allocator.
    period_total = sum(r["predicted_total_guests"] for r in forecasts)
    if metadata.get("guest_mix_ready") is True:
        desired = serving.estimate_guest_mix(period_total, metadata)
        agg = {k: 0 for k in serving.GUEST_MIX_CATEGORIES}
        for row in forecasts:
            mix = row["estimated_guest_mix"]
            assert mix is not None
            assert sum(mix.values()) == row["predicted_total_guests"]
            for k, v in mix.items():
                agg[k] += v
        if agg != desired:
            raise serving.ForecastError("MODEL_INVALID", "Period guest mix inconsistency.", status_code=500)

    bookings_metrics = ((metadata.get("targets") or {}).get("bookings") or {}).get("selected_metrics") or {}
    guests_metrics = ((metadata.get("targets") or {}).get("guests") or {}).get("selected_metrics") or {}

    return {
        "success": True,
        "model_version": str(metadata.get("model_version", "3.2")),
        "generated_at": serving.utc_now_iso(),
        "model": {
            "version": str(metadata.get("model_version", "3.2")),
            "trained_at": metadata.get("trained_at"),
            "training_days": metadata.get("training_days"),
            "safe_for_live_use": metadata.get("safe_for_live_use") is True,
            "data_source": metadata.get("data_source"),
            "date_from": metadata.get("date_from"),
            "date_to": metadata.get("date_to"),
            "minimum_training_days": metadata.get("minimum_training_days"),
            "targets": metadata.get("targets") or {},
            "postprocess": metadata.get("postprocess") or {},
            "metrics": {
                "mae": bookings_metrics.get("mae"),
                "rmse": bookings_metrics.get("rmse"),
                "wape": bookings_metrics.get("wape"),
                "bias": bookings_metrics.get("bias"),
            },
            "interval_status": guests_metrics.get("interval_status", "sumratio_ratio_quantiles"),
            "history_as_of": metadata.get("date_to"),
            "runtime_optimization": False,
            "baseline_note": (
                "Isolated V3.2 RC: two_stage bookings unchanged from v3.1; "
                "guests_per_booking_sumratio (past-84-day Σg/Σb); floor_round; "
                "period-preserving demographic allocation. Not production-approved."
            ),
            "two_stage_parameters": cfg,
            "sumratio_stats": stats,
            "deployment_scope": metadata.get("deployment_scope"),
            "production_approved": False,
            "shadow_serving_only": True,
            "rollback_release": metadata.get("rollback_release"),
        },
        "model_metadata": metadata,
        "forecast_range": {
            "start_date": dates[0].strftime("%Y-%m-%d"),
            "end_date": dates[-1].strftime("%Y-%m-%d"),
            "days": int(len(dates)),
        },
        "guest_mix": serving.build_guest_mix_block(metadata),
        "aggregate_interval_available": False,
        "forecast": forecasts,
    }


def run_forecast(
    start_date: str,
    end_date: str,
    *,
    allow_historical: bool = False,
    use_cache: bool = True,
) -> dict:
    metadata = validate_readiness(load_metadata(use_cache=use_cache))
    start = serving.parse_ymd(start_date, "start_date")
    end = serving.parse_ymd(end_date, "end_date")
    serving.validate_range(start, end, metadata, allow_historical)
    return generate_shadow_forecast(start, end, metadata, use_cache=use_cache)
