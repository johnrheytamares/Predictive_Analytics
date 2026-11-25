"""Isolated V3.1 forecast engine: two_stage + robust_median_guests + floor_round.

Does not replace production predict_real.generate_forecast / Variant A path.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
sys.path.insert(0, str(ROOT))

import predict_real as serving  # noqa: E402

# Frozen intermittent recipe (no re-optimization).
_spec = importlib.util.spec_from_file_location(
    "v31_intermittent",
    HERE / "package" / "evaluation_evidence" / "intermittent_experiment.py",
)
intermittent = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(intermittent)

_ops_path = ARTIFACTS / "ops.py"
if not _ops_path.exists():
    _ops_path = HERE / "package" / "postprocess_evidence" / "ops.py"
_ops_spec = importlib.util.spec_from_file_location("v31_ops", _ops_path)
ops = importlib.util.module_from_spec(_ops_spec)
assert _ops_spec.loader is not None
_ops_spec.loader.exec_module(ops)

_lock = threading.Lock()
_cached_metadata: dict | None = None
_cached_bookings = None
_cached_guests = None


def validate_integrity() -> dict:
    """Fail closed on changed frozen evidence or serving dependencies; no writes."""
    try:
        snapshot = json.loads((HERE / "serving_snapshot.json").read_text(encoding="utf-8"))
        for name, expected in snapshot["protected_files"].items():
            path = ROOT / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(path.name)
        recorded = json.loads((HERE / "verification_horizons_postprocess.json").read_text(encoding="utf-8"))
        if any(r["cfg"] != snapshot["forward_parameters"] for r in recorded["horizons"].values()):
            raise ValueError("Forward parameters differ from frozen horizon evidence")
        return snapshot
    except (OSError, ValueError, KeyError) as exc:
        raise serving.ForecastError("MODEL_CONFIGURATION_MISMATCH",
            "Frozen V3.1 serving evidence or dependencies failed integrity verification.", status_code=422) from exc


def weekly_performance() -> dict:
    """Score existing frozen predictions only; never rerun forecasts or tuning."""
    frame = pd.read_csv(HERE / "package/postprocess_evidence/daily_predictions.csv")
    result = {}
    for target, actual, predicted in (("bookings", "actual_bookings", "pred_bookings"),
                                      ("guests", "actual_guests", "pred_guests")):
        result[target] = {}
        for mode, label in (("raw_float", "raw"), ("floor_round", "operational_floor_round"),
                            ("variant_a", "variant_a")):
            part = frame[(frame.model == "two_stage") & (frame.ops == mode)]
            if len(part) != 196 or part.origin.nunique() != 28 or part.duplicated(["origin", "date"]).any():
                raise serving.ForecastError("MODEL_CONFIGURATION_MISMATCH", "Frozen evaluation rows are incomplete.", status_code=422)
            grouped = part.groupby(["origin", "week"])[[actual, predicted]].sum()
            result[target][label] = intermittent.metrics(grouped[actual], grouped[predicted])
    result["evaluation_scope"] = "retrospective_rolling_28_origins"
    result["evaluation_horizon_days"] = 7
    result["source"] = "package/postprocess_evidence/daily_predictions.csv"
    result["long_horizon_accuracy_evaluated"] = False
    return result


def artifact_paths() -> dict[str, Path]:
    return {
        "metadata": ARTIFACTS / "forecast_models_metadata.json",
        "bookings": ARTIFACTS / "bookings_model.joblib",
        "guests": ARTIFACTS / "guests_model.joblib",
        "ops": ARTIFACTS / "ops.py",
        "freeze": ARTIFACTS / "frozen_selection.json",
    }


def load_metadata(*, use_cache: bool = True) -> dict:
    global _cached_metadata
    validate_integrity()
    with _lock:
        if use_cache and _cached_metadata is not None:
            return _cached_metadata
        path = artifact_paths()["metadata"]
        if not path.exists():
            raise serving.ForecastError(
                "METADATA_MISSING",
                "V3.1 shadow metadata missing; run prepare_shadow_serving.py",
                status_code=404,
            )
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
        for key in ("bookings", "guests"):
            if not paths[key].exists():
                raise serving.ForecastError(
                    "MODEL_MISSING",
                    f"V3.1 shadow {key} artifact missing; run prepare_shadow_serving.py",
                    status_code=404,
                )
        bookings = joblib.load(paths["bookings"])
        guests = joblib.load(paths["guests"])
        if use_cache:
            _cached_bookings, _cached_guests = bookings, guests
        return bookings, guests


def clear_cache() -> None:
    global _cached_metadata, _cached_bookings, _cached_guests
    with _lock:
        _cached_metadata = None
        _cached_bookings = None
        _cached_guests = None


def validate_v31_configuration(bookings, guests, metadata: dict) -> None:
    if bookings.get("model_type") != "two_stage":
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "V3.1 shadow bookings must be two_stage.",
            status_code=422,
        )
    if guests.get("model_type") != serving.ROBUST_MEDIAN_GUESTS:
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "V3.1 shadow guests must be robust_median_guests.",
            status_code=422,
        )
    targets = metadata.get("targets") or {}
    if (targets.get("bookings") or {}).get("selected_model") != "two_stage":
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "Metadata bookings selection must be two_stage.",
            status_code=422,
        )
    if (targets.get("guests") or {}).get("selected_model") != serving.ROBUST_MEDIAN_GUESTS:
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "Metadata guests selection must be robust_median_guests.",
            status_code=422,
        )
    if (metadata.get("postprocess") or {}).get("selected_mode") != "floor_round":
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "Metadata postprocess must be floor_round.",
            status_code=422,
        )
    if metadata.get("shadow_serving_only") is not True or metadata.get("production_approved") is not False:
        raise serving.ForecastError("MODEL_CONFIGURATION_MISMATCH", "Frozen candidate must remain isolated shadow only.", status_code=422)


def history_series(bundle) -> pd.Series:
    history = pd.DataFrame(bundle.get("baseline_history") or [])
    if history.empty:
        raise serving.ForecastError(
            "MODEL_INVALID",
            "Shadow model bundle has no baseline_history.",
            status_code=422,
        )
    history["ds"] = pd.to_datetime(history["ds"])
    series = history.set_index("ds").sort_index()["y"].astype(float)
    if series.index.has_duplicates:
        raise serving.ForecastError(
            "MODEL_INVALID",
            "Shadow history contains duplicate dates.",
            status_code=422,
        )
    if not np.isfinite(series.values).all() or (series.values < 0).any() or not series.index.equals(pd.date_range(series.index.min(), series.index.max())):
        raise serving.ForecastError("MODEL_INVALID", "Shadow history must contain continuous finite nonnegative daily observations.", status_code=422)
    return series


def raw_predictions(
    bookings_bundle,
    guests_bundle,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Return raw float bookings, raw guest predictions, guest bound rows, and two_stage cfg."""
    cutoff = dates.min()
    train_b = history_series(bookings_bundle)
    train_b = train_b[train_b.index < cutoff]
    if len(train_b) < 120:
        raise serving.ForecastError(
            "MODEL_NOT_READY",
            "Insufficient booking history before cutoff for two_stage nested tune.",
            status_code=422,
        )
    snapshot = validate_integrity()
    if cutoff >= pd.Timestamp(snapshot["forward_cutoff"]):
        cfg = snapshot["forward_parameters"]
    else:
        freeze = json.loads((HERE / "package/frozen_selection.json").read_text(encoding="utf-8"))
        saved = [r["parameters"] for r in freeze["per_origin_parameters"]
                 if r["protocol"] == "rolling_7day" and r["origin"] == str(cutoff.date())]
        if not saved:
            raise serving.ForecastError("FROZEN_PARAMETERS_UNAVAILABLE",
                "Historical start has no frozen seven-day origin parameters; no retuning is permitted.", status_code=422)
        cfg = saved[0]
    raw_b = intermittent.predict(train_b, dates, "two_stage", cfg)

    # Robust median path (reuse production helper; no Variant A applied here).
    guest_rows = serving.predict_series(guests_bundle, dates)
    raw_g = np.asarray([float(r["prediction"]) for r in guest_rows], dtype=float)
    if len(raw_b) != len(dates) or len(raw_g) != len(dates) or not np.isfinite(raw_b).all() or not np.isfinite(raw_g).all():
        raise serving.ForecastError("MODEL_INVALID", "Frozen predictor returned invalid values or row counts.", status_code=422)
    return np.asarray(raw_b, float), raw_g, guest_rows, cfg


def apply_floor_round(
    raw_b: np.ndarray, raw_g: np.ndarray
) -> tuple[list[int], list[int]]:
    return ops.apply_ops(raw_b, raw_g, mode="floor_round")


def apply_variant_a(
    raw_b: np.ndarray, raw_g: np.ndarray
) -> tuple[list[int], list[int]]:
    """Comparison-only path; not used for serving."""
    return ops.apply_ops(raw_b, raw_g, mode="variant_a", party_size=7.0)


def shadow_capabilities(guests_bundle, cutoff: pd.Timestamp) -> dict:
    caps = serving.robust_capabilities(guests_bundle, cutoff)
    caps = dict(caps)
    caps.update(
        {
            "bookings_method": "two_stage",
            "postprocess": "floor_round",
            "party_size_floor": False,
            "supported_horizons_days": [7, 30, 90],
            "serving_mode": "isolated_shadow",
        }
    )
    return caps


def validate_readiness(metadata: dict | None) -> dict:
    if not metadata:
        raise serving.ForecastError(
            "METADATA_MISSING",
            "Required prediction model is missing.",
            status_code=404,
        )
    if metadata.get("is_demo") is True:
        raise serving.ForecastError("DEMO_MODEL", "Demo model is not allowed for shadow serving.", status_code=422)
    if metadata.get("ready") is not True:
        raise serving.ForecastError("MODEL_NOT_READY", "Shadow model metadata is not ready.", status_code=422)
    if metadata.get("safe_for_live_use") is not True:
        raise serving.ForecastError(
            "MODEL_NOT_SAFE_FOR_LIVE_USE",
            "Shadow model is not marked safe_for_live_use.",
            status_code=422,
        )
    if metadata.get("production_approved") is True:
        raise serving.ForecastError(
            "MODEL_CONFIGURATION_MISMATCH",
            "V3.1 shadow artifacts must remain production_approved=false.",
            status_code=422,
        )
    return metadata


def generate_shadow_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    metadata: dict,
    *,
    use_cache: bool = True,
    include_variant_a_comparison: bool = False,
) -> dict:
    bookings, guests = get_bundles(use_cache=use_cache)
    validate_v31_configuration(bookings, guests, metadata)

    dates = pd.date_range(start=start, end=end, freq="D")
    raw_b, raw_g, guest_rows, cfg = raw_predictions(bookings, guests, dates)
    op_b, op_g = apply_floor_round(raw_b, raw_g)

    guest_mix_ready = metadata.get("guest_mix_ready") is True
    forecasts = []
    for i, ds in enumerate(dates):
        predicted_bookings = int(op_b[i])
        predicted_guests = int(op_g[i])
        g_row = guest_rows[i]
        bookings_lower, bookings_upper = serving.reconcile_bounds(
            predicted_bookings,
            serving.clamp_int(max(0.0, float(raw_b[i]) - 1.0)),
            serving.clamp_int(float(raw_b[i]) + 1.0),
        )
        guests_lower, guests_upper = serving.reconcile_bounds(
            predicted_guests, int(g_row["lower"]), int(g_row["upper"])
        )
        if predicted_guests < predicted_bookings:
            raise serving.ForecastError(
                "MODEL_INVALID",
                "floor_round consistency failed: guests < bookings.",
                status_code=500,
            )
        if predicted_guests > 0 and predicted_bookings < 1:
            raise serving.ForecastError(
                "MODEL_INVALID",
                "floor_round consistency failed: positive guests with zero bookings.",
                status_code=500,
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
                "estimated_guest_mix": serving.estimate_guest_mix(predicted_guests, metadata),
                "guest_mix_ready": guest_mix_ready,
            }
        )

    bookings_target = (metadata.get("targets") or {}).get("bookings") or {}
    guests_target = (metadata.get("targets") or {}).get("guests") or {}
    bookings_metrics = bookings_target.get("selected_metrics") or {}
    guests_metrics = guests_target.get("selected_metrics") or {}
    performance = weekly_performance()

    payload = {
        "success": True,
        "model_version": str(metadata.get("model_version", "3.1")),
        "generated_at": serving.utc_now_iso(),
        "model": {
            "version": str(metadata.get("model_version", "3.1")),
            "trained_at": metadata.get("trained_at"),
            "training_days": metadata.get("training_days") or metadata.get("usable_days"),
            "safe_for_live_use": metadata.get("safe_for_live_use") is True,
            "data_source": metadata.get("data_source"),
            "date_from": metadata.get("date_from"),
            "date_to": metadata.get("date_to"),
            "minimum_training_days": metadata.get("minimum_training_days"),
            "targets": metadata.get("targets") or {},
            "postprocess": metadata.get("postprocess") or {},
            "metrics": {
                "mae": bookings_metrics.get("mae", metadata.get("mae")),
                "rmse": bookings_metrics.get("rmse", metadata.get("rmse")),
                "wape": bookings_metrics.get("wape", metadata.get("wape")),
                "bias": bookings_metrics.get("bias", metadata.get("bias")),
                "prediction_interval_coverage": bookings_metrics.get(
                    "prediction_interval_coverage"
                ),
            },
            "bookings_weekly_metrics": performance["bookings"],
            "guests_weekly_metrics": performance["guests"],
            "weekly_metric_scope": {k: v for k, v in performance.items() if k not in {"bookings", "guests"}},
            "interval_status": "Not evaluated",
            "history_as_of": metadata.get("date_to"),
            "runtime_optimization": False,
            "baseline_metrics": None,
            "baseline_note": (
                "Experimental Version 3.1 shadow: original two_stage bookings, "
                "robust_median_guests, floor_round postprocess. Retrospective metrics only."
            ),
            "two_stage_parameters": cfg,
            "deployment_scope": metadata.get("deployment_scope"),
            "production_approved": False,
            "shadow_serving_only": True,
            "evaluation_scope": metadata.get("evaluation_scope"),
            "prospective_validation": metadata.get("prospective_validation"),
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

    if include_variant_a_comparison:
        var_b, var_g = apply_variant_a(raw_b, raw_g)
        payload["postprocess_comparison"] = {
            "floor_round": {
                "bookings_sum": int(sum(op_b)),
                "guests_sum": int(sum(op_g)),
            },
            "variant_a": {
                "bookings_sum": int(sum(var_b)),
                "guests_sum": int(sum(var_g)),
            },
            "guest_reduction_vs_variant_a": int(sum(var_g) - sum(op_g)),
            "explanation": (
                "Variant A rounds then applies a ~7 guests-per-booking party floor and "
                "booking-minimum consistency, inflating guest totals whenever bookings>0. "
                "floor_round floors both series and only enforces guests>=bookings / "
                "booking-min when guests remain >0 — it does not multiply by party size. "
                "The substantial guest reduction is therefore an intentional ops-policy "
                "difference, not a change to the Robust Median guest model."
            ),
        }
    return payload


def run_forecast(
    start_date: str,
    end_date: str,
    *,
    allow_historical: bool = False,
    use_cache: bool = True,
    include_variant_a_comparison: bool = False,
) -> dict:
    metadata = validate_readiness(load_metadata(use_cache=use_cache))
    start = serving.parse_ymd(start_date, "start_date")
    end = serving.parse_ymd(end_date, "end_date")
    serving.validate_range(start, end, metadata, allow_historical)
    return generate_shadow_forecast(
        start,
        end,
        metadata,
        use_cache=use_cache,
        include_variant_a_comparison=include_variant_a_comparison,
    )
