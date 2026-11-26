"""
Phase 4 – Forecast generation.

Loads the version 3.0 artifacts (bookings + guests Prophet/baseline models and
forecast_models_metadata.json), validates readiness, and emits JSON-only
daily forecasts for a requested future date range.

Usage (CLI):
    python predict_real.py --start-date 2026-07-27 --end-date 2026-08-26
    python predict_real.py metadata                 (legacy)
    python predict_real.py date 2026-08-01          (legacy)
    python predict_real.py range 2026-08-01 2026-08-07   (legacy)

Reusable entry for the FastAPI service:
    from predict_real import run_forecast
    payload = run_forecast("2026-08-27", "2026-09-02")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# Explicit opt-in for isolated local releases; canonical defaults are unchanged.
ARTIFACT_ROOT = Path(os.getenv("PREDICTION_ARTIFACT_DIR") or ROOT).resolve()
BOOKINGS_MODEL_PATH = ARTIFACT_ROOT / "bookings_model.joblib"
GUESTS_MODEL_PATH = ARTIFACT_ROOT / "guests_model.joblib"
METADATA_PATH = ARTIFACT_ROOT / "forecast_models_metadata.json"
LEGACY_METADATA_PATH = ARTIFACT_ROOT / "bookings_model_metadata.json"

MAX_FORECAST_DAYS = int(os.getenv("PREDICTION_MAX_FORECAST_DAYS", "90"))
GUEST_MIX_UNAVAILABLE_MESSAGE = "Insufficient reliable demographic history."
ROBUST_MEDIAN_GUESTS = "robust_median_guests"

_model_lock = threading.Lock()
_cached_metadata: dict | None = None
_cached_bookings_model = None
_cached_guests_model = None


class ForecastError(Exception):
    """Domain error raised by reusable forecast helpers (safe for API mapping)."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        extra: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.extra = extra or {}


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def emit(payload: dict, exit_code: int = 0) -> None:
    print(json.dumps(payload))
    sys.exit(exit_code)


def fail(code: str, message: str, extra: dict | None = None, exit_code: int = 2) -> None:
    """Raise ForecastError for reusable callers; CLI layers catch and emit JSON."""
    status = 404 if code in {"MODEL_MISSING", "METADATA_MISSING"} else 400
    if code in {"MODEL_NOT_READY", "MODEL_NOT_SAFE_FOR_LIVE_USE", "DEMO_MODEL", "MODEL_INVALID"}:
        status = 422
    raise ForecastError(code, message, status_code=status, extra=extra)


def _cli_fail(code: str, message: str, extra: dict | None = None, exit_code: int = 2) -> None:
    payload = {"success": False, "code": code, "message": message}
    if extra:
        payload.update(extra)
    emit(payload, exit_code)


def load_metadata(*, use_cache: bool = True) -> dict | None:
    global _cached_metadata
    if use_cache and _cached_metadata is not None:
        return _cached_metadata

    for path in (METADATA_PATH, LEGACY_METADATA_PATH):
        if path.exists():
            try:
                with path.open(encoding="utf-8") as handle:
                    metadata = json.load(handle)
                if use_cache:
                    _cached_metadata = metadata
                return metadata
            except (OSError, json.JSONDecodeError):
                continue
    return None


def load_model_bundle(path: Path | str, label: str):
    model_path = Path(path)
    if not model_path.exists():
        fail(
            "MODEL_MISSING",
            f"{label} model file not found. Run train_real_model.py first.",
            {"missing_model": model_path.name},
        )
    bundle = joblib.load(model_path)
    model = bundle.get("model") if isinstance(bundle, dict) else bundle
    if model is None:
        fail(
            "MODEL_INVALID",
            f"{label} model bundle does not contain a Prophet model.",
            {"model_file": model_path.name},
        )
    return bundle if isinstance(bundle, dict) else {"model": model, "model_type": "prophet"}


def get_model_bundles(*, use_cache: bool = True):
    """Load bookings + guests model bundles once (thread-safe cache)."""
    global _cached_bookings_model, _cached_guests_model
    if use_cache and _cached_bookings_model is not None and _cached_guests_model is not None:
        return _cached_bookings_model, _cached_guests_model

    with _model_lock:
        if use_cache and _cached_bookings_model is not None and _cached_guests_model is not None:
            return _cached_bookings_model, _cached_guests_model
        bookings = load_model_bundle(BOOKINGS_MODEL_PATH, "Bookings")
        guests = load_model_bundle(GUESTS_MODEL_PATH, "Guests")
        if use_cache:
            _cached_bookings_model = bookings
            _cached_guests_model = guests
        return bookings, guests


def clear_model_cache() -> None:
    global _cached_metadata, _cached_bookings_model, _cached_guests_model
    with _model_lock:
        _cached_metadata = None
        _cached_bookings_model = None
        _cached_guests_model = None


def validate_readiness(metadata: dict | None) -> None:
    if not metadata:
        fail(
            "METADATA_MISSING",
            "forecast_models_metadata.json not found. Train the models first.",
        )
    if metadata.get("is_demo"):
        fail("DEMO_MODEL", "Model was trained on demo data and cannot serve live forecasts.")
    if metadata.get("ready") is not True:
        fail(
            "MODEL_NOT_READY",
            "Trained models are not marked ready.",
            {"blocking_reasons": metadata.get("blocking_reasons", [])},
        )
    if metadata.get("safe_for_live_use") is not True:
        fail(
            "MODEL_NOT_SAFE_FOR_LIVE_USE",
            "Model metadata does not allow live forecasting.",
        )


def parse_ymd(value: str, label: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(datetime.strptime(str(value).strip(), "%Y-%m-%d"))
    except (ValueError, TypeError):
        fail("INVALID_DATE", f"{label} must use YYYY-MM-DD format.", {label: value})
    return ts.normalize()


def validate_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    metadata: dict,
    allow_historical: bool,
) -> None:
    if end < start:
        fail("INVALID_RANGE", "End date must not be earlier than start date")

    days = int((end - start).days) + 1
    if days > MAX_FORECAST_DAYS:
        fail(
            "RANGE_TOO_LARGE",
            f"Forecast range cannot exceed {MAX_FORECAST_DAYS} days (requested {days}).",
        )

    history_end = metadata.get("date_to")
    if history_end and not allow_historical:
        history_ts = pd.Timestamp(history_end).normalize()
        if start <= history_ts:
            fail(
                "RANGE_OVERLAPS_HISTORY",
                (
                    f"Start date overlaps training history (history ends {history_end}). "
                    "Use --allow-historical to override."
                ),
            )


def clamp_int(value: float) -> int:
    return int(max(0, round(float(value))))


def reconcile_bounds(prediction: int, lower: int, upper: int) -> tuple[int, int]:
    lower = min(lower, prediction)
    upper = max(upper, prediction)
    return lower, upper


def predict_series(bundle, dates: pd.DatetimeIndex) -> list[dict]:
    if bundle.get("model_type") == ROBUST_MEDIAN_GUESTS:
        return predict_robust_median_guests(bundle, dates)
    model = bundle.get("model")
    if bundle.get("model_type") == "weekday_baseline":
        history = pd.DataFrame(bundle.get("baseline_history") or [])
        if history.empty:
            fail("MODEL_INVALID", "Weekday baseline model has no historical observations.")
        history["ds"] = pd.to_datetime(history["ds"])
        history = history.set_index("ds").sort_index()
        values = []
        residuals = []
        for ds, row in history.iterrows():
            prior = [history.loc[ds - pd.Timedelta(days=7 * week), "y"]
                     for week in range(1, 5)
                     if ds - pd.Timedelta(days=7 * week) in history.index]
            if prior:
                residuals.append(abs(float(row["y"]) - float(np.mean(prior))))
        margin = float(np.quantile(residuals, 0.90)) if residuals else 0.0
        mean = float(history["y"].mean())
        for ds in dates:
            prior = [history.loc[ds - pd.Timedelta(days=7 * week), "y"]
                     for week in range(1, 5)
                     if ds - pd.Timedelta(days=7 * week) in history.index]
            # A requested range may begin more than four weeks after the last
            # historical arrival. Preserve weekday seasonality in that case
            # by using the eight most recent observations for the same
            # weekday instead of collapsing every future date to one global
            # average.
            if not prior:
                weekday_values = history.loc[
                    history.index.dayofweek == ds.dayofweek, "y"
                ].tail(8).to_numpy(dtype=float)
                prior = weekday_values.tolist()
            prediction = float(np.mean(prior)) if prior else mean
            values.append({
                "prediction": clamp_int(prediction),
                "lower": clamp_int(max(0.0, prediction - margin)),
                "upper": clamp_int(prediction + margin),
            })
        return values

    frame = pd.DataFrame({"ds": dates})
    forecast = model.predict(frame)
    rows = []
    for i in range(len(dates)):
        prediction = clamp_int(forecast["yhat"].iloc[i])
        lower = clamp_int(forecast["yhat_lower"].iloc[i])
        upper = clamp_int(forecast["yhat_upper"].iloc[i])
        lower, upper = reconcile_bounds(prediction, lower, upper)
        rows.append({"prediction": prediction, "lower": lower, "upper": upper})
    return rows


def robust_history(bundle, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Validate observations usable at a fixed cutoff, without filling missing days."""
    history = pd.DataFrame(bundle.get("baseline_history") or [])
    if history.empty or not {"ds", "y"}.issubset(history.columns):
        raise ForecastError("ROBUST_HISTORY_INVALID", "Robust median guests requires saved daily history.", status_code=422)
    try:
        history["ds"] = pd.to_datetime(history["ds"], errors="raise")
        if history["ds"].isna().any() or not (history["ds"] == history["ds"].dt.normalize()).all():
            raise ValueError("Dates must be daily observations")
        history = history.loc[history["ds"] < cutoff].set_index("ds").sort_index()
        values = history["y"].to_numpy(dtype=float)
        if history.index.has_duplicates or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Duplicate dates or invalid counts")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ForecastError("ROBUST_HISTORY_INVALID", "Robust guest history requires unique daily dates and finite nonnegative counts.", status_code=422) from exc
    if history.empty:
        raise ForecastError("ROBUST_HISTORY_INVALID", "No robust guest history exists before the forecast cutoff.", status_code=422)
    return history


def robust_weekday_values(history: pd.DataFrame, weekday: int, *, before: pd.Timestamp | None = None) -> list[float]:
    """Four most recent matching-weekday actuals before an optional bound (zeros kept)."""
    frame = history
    if before is not None:
        frame = frame.loc[frame.index < before]
    values = frame.loc[frame.index.dayofweek == int(weekday), "y"]
    return [float(v) for v in values.tail(4).tolist()]


def robust_capabilities(bundle, cutoff: pd.Timestamp) -> dict:
    """Cutoff-anchored profile: each weekday needs four pre-cutoff actuals; horizon up to MAX_FORECAST_DAYS."""
    history = robust_history(bundle, cutoff)
    weekday_counts = {
        weekday: int((history.index.dayofweek == weekday).sum()) for weekday in range(7)
    }
    supported = []
    for ds in pd.date_range(cutoff, periods=MAX_FORECAST_DAYS):
        if weekday_counts[int(ds.dayofweek)] >= 4:
            supported.append(ds.strftime("%Y-%m-%d"))
        else:
            break
    return {
        "method": ROBUST_MEDIAN_GUESTS,
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "required_matching_weekdays": 4,
        "includes_zero_days": True,
        "profile": "cutoff_anchored_four_week_weekday_median",
        "maximum_days_per_cutoff": MAX_FORECAST_DAYS,
        "supported_dates": supported,
        "latest_supported_date": max(supported) if supported else None,
        "history_through": history.index.max().strftime("%Y-%m-%d"),
        "weekday_observation_counts": weekday_counts,
        "fallback": None,
        "recursive_predictions": False,
    }


def validate_robust_configuration(bookings_bundle, guests_bundle, metadata):
    actual = guests_bundle.get("model_type")
    label = ((metadata.get("targets") or {}).get("guests") or {}).get("selected_model")
    if ROBUST_MEDIAN_GUESTS in (actual, label):
        if actual != label or bookings_bundle.get("model_type") != "weekday_baseline" or (
            ((metadata.get("targets") or {}).get("bookings") or {}).get("selected_model") != "weekday_baseline"
        ):
            raise ForecastError("MODEL_CONFIGURATION_MISMATCH", "Robust guest dispatch and metadata must agree, with weekday baseline bookings.", status_code=422)


def predict_robust_median_guests(bundle, dates: pd.DatetimeIndex) -> list[dict]:
    """Cutoff-anchored Robust Median: median of four pre-cutoff matching weekdays, zeros included.

    For each weekday required by the request, take the four most recent recorded
    actuals strictly before the forecast cutoff and reuse that median for every
    matching target date in the horizon. Never consume on/after-cutoff actuals
    and never substitute recursively predicted values. Bounds use historical
    residuals of this same rule; coverage is not validated.
    """
    if len(dates) == 0:
        return []
    cutoff = dates.min()
    history = robust_history(bundle, cutoff)

    profiles: dict[int, float] = {}
    missing = []
    seen_weekdays: set[int] = set()
    for ds in dates:
        weekday = int(ds.dayofweek)
        if weekday in seen_weekdays:
            continue
        seen_weekdays.add(weekday)
        prior = robust_weekday_values(history, weekday)
        if len(prior) < 4:
            missing.append({
                "target_date": ds.strftime("%Y-%m-%d"),
                "weekday": weekday,
                "missing_historical_dates": [],
                "available_matching_weekdays": len(prior),
                "required_matching_weekdays": 4,
            })
        else:
            profiles[weekday] = float(np.median(prior))
    if missing:
        raise ForecastError(
            "ROBUST_RANGE_UNSUPPORTED",
            "Every weekday in the robust guest range requires four matching weekdays strictly before the forecast cutoff. No recursive or mean fallback is used.",
            status_code=422,
            extra={"policy": robust_capabilities(bundle, cutoff), "unsupported_dates": missing},
        )

    residuals = []
    for ds, row in history.iterrows():
        prior = robust_weekday_values(history, int(ds.dayofweek), before=ds)
        if len(prior) == 4:
            residuals.append(abs(float(row["y"]) - float(np.median(prior))))
    margin = float(np.quantile(residuals, 0.90)) if residuals else 0.0

    rows = []
    for ds in dates:
        prediction = profiles[int(ds.dayofweek)]
        rows.append({
            "prediction": clamp_int(prediction),
            "lower": clamp_int(max(0.0, prediction - margin)),
            "upper": clamp_int(prediction + margin),
        })
    return rows


def build_guest_mix_block(metadata: dict) -> dict:
    ready = metadata.get("guest_mix_ready") is True
    coverage = float(metadata.get("guest_mix_coverage_pct") or 0)
    block = {"ready": ready, "coverage_pct": coverage}
    if not ready:
        block["message"] = GUEST_MIX_UNAVAILABLE_MESSAGE
    return block


GUEST_MIX_CATEGORIES = ("adults", "children", "seniors", "infants", "unknown_guests")


def estimate_guest_mix(total: int, metadata: dict) -> dict | None:
    """Largest-remainder allocation of a single guest total onto frozen proportions."""
    if metadata.get("guest_mix_ready") is not True:
        return None
    proportions = metadata.get("guest_mix_proportions") or {}
    categories = GUEST_MIX_CATEGORIES
    total = max(0, int(total))
    raw = {key: max(0.0, float(proportions.get(key, 0))) * total for key in categories}
    allocated = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = total - sum(allocated.values())
    # Tie-break: larger fractional remainder, then stable category order.
    order = sorted(
        categories,
        key=lambda item: (raw[item] - allocated[item], -categories.index(item)),
        reverse=True,
    )
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def allocate_guest_mix_period(
    daily_guests: list[int] | tuple[int, ...],
    metadata: dict,
) -> list[dict] | None:
    """Allocate guest categories across a forecast period.

    1. Allocate the period total with ``estimate_guest_mix`` (frozen proportions).
    2. Seed each day with the same single-total allocator.
    3. Transfer units *within* days so category column sums match the period
       aggregate while each day's guest total is unchanged.

    Does not alter booking or total-guest forecasts.
    """
    if metadata.get("guest_mix_ready") is not True:
        return None

    categories = GUEST_MIX_CATEGORIES
    guests = [max(0, int(value)) for value in daily_guests]
    period_total = sum(guests)
    if period_total == 0:
        return [{key: 0 for key in categories} for _ in guests]

    desired = estimate_guest_mix(period_total, metadata)
    assert desired is not None
    assert sum(desired.values()) == period_total

    rows: list[dict] = []
    for day_total in guests:
        if day_total == 0:
            rows.append({key: 0 for key in categories})
        else:
            seed = estimate_guest_mix(day_total, metadata)
            assert seed is not None
            rows.append(dict(seed))

    current = {key: sum(row[key] for row in rows) for key in categories}
    # At most one transfer per guest-unit of mismatch.
    for _ in range(period_total * len(categories) + 1):
        surplus = [key for key in categories if current[key] > desired[key]]
        deficit = [key for key in categories if current[key] < desired[key]]
        if not surplus:
            break
        moved = False
        for source in surplus:
            if current[source] <= desired[source]:
                continue
            for target in deficit:
                if current[target] >= desired[target]:
                    continue
                for row in rows:
                    if row[source] <= 0:
                        continue
                    row[source] -= 1
                    row[target] += 1
                    current[source] -= 1
                    current[target] += 1
                    moved = True
                    break
                if moved:
                    break
            if moved:
                break
        if not moved:
            raise RuntimeError("Unable to reconcile daily guest mix to period aggregate.")

    if current != desired:
        raise RuntimeError("Guest mix period reconciliation left residual mismatch.")
    for day_total, row in zip(guests, rows):
        if sum(row.values()) != day_total:
            raise RuntimeError("Guest mix redistribution changed a daily guest total.")
    return rows


def apply_period_guest_mix(forecasts: list[dict], metadata: dict) -> None:
    """Set each row's estimated_guest_mix from period-preserving allocation."""
    ready = metadata.get("guest_mix_ready") is True
    if not ready:
        for row in forecasts:
            row["estimated_guest_mix"] = None
            row["guest_mix_ready"] = False
        return
    mixes = allocate_guest_mix_period(
        [int(row["predicted_total_guests"]) for row in forecasts],
        metadata,
    )
    assert mixes is not None and len(mixes) == len(forecasts)
    for row, mix in zip(forecasts, mixes):
        row["estimated_guest_mix"] = mix
        row["guest_mix_ready"] = True


def generate_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    metadata: dict,
    *,
    use_cache: bool = True,
) -> dict:
    bookings_model, guests_model = get_model_bundles(use_cache=use_cache)
    validate_robust_configuration(bookings_model, guests_model, metadata)

    dates = pd.date_range(start=start, end=end, freq="D")
    bookings_rows = predict_series(bookings_model, dates)
    guests_rows = predict_series(guests_model, dates)

    guest_mix_ready = metadata.get("guest_mix_ready") is True

    forecasts = []
    for i, ds in enumerate(dates):
        b = bookings_rows[i]
        g = guests_rows[i]

        predicted_bookings = b["prediction"]
        predicted_guests = g["prediction"]

        # Guests without a booking is confusing for admins: bump bookings to 1.
        if predicted_guests > 0 and predicted_bookings < 1:
            predicted_bookings = 1

        # Guest demand is conditional on bookings. Use the median historical
        # party size as a robust floor; unlike the mean it is not dominated by
        # occasional large event bookings. Keep a larger guest-model estimate
        # when it has evidence for an event-heavy date.
        typical_party_size = max(
            1.0, float(metadata.get("typical_guests_per_booking") or 1.0)
        )
        conditional_guests = clamp_int(predicted_bookings * typical_party_size)
        predicted_guests = max(conditional_guests, predicted_guests)

        bookings_lower, bookings_upper = reconcile_bounds(
            predicted_bookings, b["lower"], b["upper"]
        )
        guests_lower, guests_upper = reconcile_bounds(
            predicted_guests, g["lower"], g["upper"]
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
                # Placeholder; replaced by period-preserving allocation below.
                "estimated_guest_mix": None,
                "guest_mix_ready": guest_mix_ready,
            }
        )

    apply_period_guest_mix(forecasts, metadata)

    bookings_target = (metadata.get("targets") or {}).get("bookings") or {}
    # Older bundles store Prophet metrics under `metrics`, even when the
    # selected predictor is the baseline. Never label those as baseline scores.
    bookings_metrics = bookings_target.get("selected_metrics") or (
        bookings_target.get("baseline_metrics")
        if bookings_target.get("selected_model") == "weekday_baseline"
        else bookings_target.get("metrics")
    ) or {}
    bookings_baseline = bookings_target.get("baseline_metrics") or {}

    return {
        "success": True,
        "model_version": str(metadata.get("model_version", "3.0")),
        "generated_at": utc_now_iso(),
        "model": {
            "version": str(metadata.get("model_version", "3.0")),
            "trained_at": metadata.get("trained_at"),
            "training_days": metadata.get("training_days") or metadata.get("usable_days"),
            "safe_for_live_use": metadata.get("safe_for_live_use") is True,
            "data_source": metadata.get("data_source"),
            "date_from": metadata.get("date_from"),
            "date_to": metadata.get("date_to"),
            "minimum_training_days": metadata.get("minimum_training_days"),
            "targets": metadata.get("targets") or {},
            # Authoritative holdout metrics from training metadata (not invented).
            "metrics": {
                "mae": bookings_metrics.get("mae", metadata.get("mae")),
                "rmse": bookings_metrics.get("rmse", metadata.get("rmse")),
                "wape": bookings_metrics.get("wape", metadata.get("wape")),
                "bias": bookings_metrics.get("bias", metadata.get("bias")),
                "prediction_interval_coverage": bookings_metrics.get(
                    "prediction_interval_coverage"
                ),
            },
            "baseline_metrics": bookings_baseline or None,
            "baseline_note": metadata.get("baseline_note")
            or bookings_target.get("baseline_note"),
        },
        "model_metadata": metadata,
        "forecast_range": {
            "start_date": dates[0].strftime("%Y-%m-%d"),
            "end_date": dates[-1].strftime("%Y-%m-%d"),
            "days": int(len(dates)),
        },
        "guest_mix": build_guest_mix_block(metadata),
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
    """
    Reusable forecast entrypoint for CLI and FastAPI.
    Raises ForecastError on validation / readiness / model failures.
    """
    metadata = load_metadata(use_cache=use_cache)
    validate_readiness(metadata)

    start = parse_ymd(start_date, "start_date")
    end = parse_ymd(end_date, "end_date")
    validate_range(start, end, metadata, allow_historical)

    return generate_forecast(start, end, metadata, use_cache=use_cache)


def run_phase4(start_str: str, end_str: str, allow_historical: bool) -> None:
    try:
        emit(run_forecast(start_str, end_str, allow_historical=allow_historical, use_cache=False))
    except ForecastError as exc:
        _cli_fail(exc.code, exc.message, extra=exc.extra or None)


def run_legacy(argv: list[str]) -> None:
    mode = argv[0]

    try:
        if mode == "metadata":
            metadata = load_metadata(use_cache=False)
            emit(
                {
                    "model_metadata": metadata,
                    "safe_for_live_use": bool(metadata and metadata.get("safe_for_live_use")),
                }
            )

        metadata = load_metadata(use_cache=False)
        validate_readiness(metadata)

        if mode == "date":
            if len(argv) < 2:
                fail("INVALID_ARGS", "Usage: predict_real.py date YYYY-MM-DD")
            start = parse_ymd(argv[1], "date")
            result = generate_forecast(start, start, metadata, use_cache=False)
            row = result["forecast"][0]
            emit(
                {
                    "date": row["date"],
                    "predicted_bookings": row["predicted_bookings"],
                    "predicted_bookings_lower": row["bookings_lower"],
                    "predicted_bookings_upper": row["bookings_upper"],
                    "predicted_total_guests": row["predicted_total_guests"],
                    "guests_lower": row["guests_lower"],
                    "guests_upper": row["guests_upper"],
                    "estimated_guest_mix": row["estimated_guest_mix"],
                    "model_metadata": metadata,
                }
            )

        if mode == "range":
            if len(argv) < 3:
                fail("INVALID_ARGS", "Usage: predict_real.py range START END")
            start = parse_ymd(argv[1], "start_date")
            end = parse_ymd(argv[2], "end_date")
            validate_range(start, end, metadata, allow_historical=True)
            result = generate_forecast(start, end, metadata, use_cache=False)
            rows = [
                {
                    "date": row["date"],
                    "predicted_bookings": row["predicted_bookings"],
                    "predicted_bookings_lower": row["bookings_lower"],
                    "predicted_bookings_upper": row["bookings_upper"],
                    "predicted_total_guests": row["predicted_total_guests"],
                    "guests_lower": row["guests_lower"],
                    "guests_upper": row["guests_upper"],
                }
                for row in result["forecast"]
            ]
            emit(
                {
                    "rows": rows,
                    "model_metadata": metadata,
                    "guest_mix": result["guest_mix"],
                    "generated_at": result["generated_at"],
                }
            )

        fail("INVALID_ARGS", "Use --start-date/--end-date, or legacy modes: metadata | date | range")
    except ForecastError as exc:
        _cli_fail(exc.code, exc.message, extra=exc.extra or None)


def main() -> None:
    argv = sys.argv[1:]

    # Legacy positional interface used by older controller code.
    if argv and argv[0] in {"metadata", "date", "range"}:
        run_legacy(argv)
        return

    parser = argparse.ArgumentParser(description="Phase 4 forecast generation")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--allow-historical",
        action="store_true",
        help="Allow forecasting dates that overlap training history",
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit:
        _cli_fail(
            "INVALID_ARGS",
            "start_date and end_date are required (YYYY-MM-DD).",
            exit_code=2,
        )

    run_phase4(args.start_date, args.end_date, args.allow_historical)


if __name__ == "__main__":
    main()
