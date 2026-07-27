"""
Phase 4 – Forecast generation.

Loads the version 3.0 artifacts (bookings + guests Prophet models and
forecast_models_metadata.json), validates readiness, and emits JSON-only
daily forecasts for a requested future date range.

Usage:
    python predict_real.py --start-date 2026-07-27 --end-date 2026-08-26
    python predict_real.py metadata                 (legacy)
    python predict_real.py date 2026-08-01          (legacy)
    python predict_real.py range 2026-08-01 2026-08-07   (legacy)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOKINGS_MODEL_PATH = os.path.join(ROOT, "bookings_model.joblib")
GUESTS_MODEL_PATH = os.path.join(ROOT, "guests_model.joblib")
METADATA_PATH = os.path.join(ROOT, "forecast_models_metadata.json")
LEGACY_METADATA_PATH = os.path.join(ROOT, "bookings_model_metadata.json")

MAX_FORECAST_DAYS = int(os.getenv("PREDICTION_MAX_FORECAST_DAYS", "90"))
GUEST_MIX_UNAVAILABLE_MESSAGE = "Insufficient reliable demographic history."


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
    payload = {"success": False, "code": code, "message": message}
    if extra:
        payload.update(extra)
    emit(payload, exit_code)


def load_metadata() -> dict | None:
    for path in (METADATA_PATH, LEGACY_METADATA_PATH):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
    return None


def load_model_bundle(path: str, label: str):
    if not os.path.exists(path):
        fail(
            "MODEL_MISSING",
            f"{label} model file not found. Run train_real_model.py first.",
            {"missing_model": os.path.basename(path)},
        )
    bundle = joblib.load(path)
    model = bundle.get("model") if isinstance(bundle, dict) else bundle
    if model is None:
        fail(
            "MODEL_INVALID",
            f"{label} model bundle does not contain a Prophet model.",
            {"model_file": os.path.basename(path)},
        )
    return model


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


def predict_series(model, dates: pd.DatetimeIndex) -> list[dict]:
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


def build_guest_mix_block(metadata: dict) -> dict:
    ready = metadata.get("guest_mix_ready") is True
    coverage = float(metadata.get("guest_mix_coverage_pct") or 0)
    block = {"ready": ready, "coverage_pct": coverage}
    if not ready:
        block["message"] = GUEST_MIX_UNAVAILABLE_MESSAGE
    return block


def generate_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    metadata: dict,
) -> dict:
    bookings_model = load_model_bundle(BOOKINGS_MODEL_PATH, "Bookings")
    guests_model = load_model_bundle(GUESTS_MODEL_PATH, "Guests")

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

        predicted_guests = max(predicted_bookings, predicted_guests)

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
                "estimated_guest_mix": None,
                "guest_mix_ready": guest_mix_ready,
            }
        )

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
        },
        "forecast_range": {
            "start_date": dates[0].strftime("%Y-%m-%d"),
            "end_date": dates[-1].strftime("%Y-%m-%d"),
            "days": int(len(dates)),
        },
        "guest_mix": build_guest_mix_block(metadata),
        "aggregate_interval_available": False,
        "forecast": forecasts,
    }


def run_phase4(start_str: str, end_str: str, allow_historical: bool) -> None:
    metadata = load_metadata()
    validate_readiness(metadata)

    start = parse_ymd(start_str, "start_date")
    end = parse_ymd(end_str, "end_date")
    validate_range(start, end, metadata, allow_historical)

    emit(generate_forecast(start, end, metadata))


def run_legacy(argv: list[str]) -> None:
    mode = argv[0]

    if mode == "metadata":
        metadata = load_metadata()
        emit(
            {
                "model_metadata": metadata,
                "safe_for_live_use": bool(metadata and metadata.get("safe_for_live_use")),
            }
        )

    metadata = load_metadata()
    validate_readiness(metadata)

    if mode == "date":
        if len(argv) < 2:
            fail("INVALID_ARGS", "Usage: predict_real.py date YYYY-MM-DD")
        start = parse_ymd(argv[1], "date")
        result = generate_forecast(start, start, metadata)
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
        result = generate_forecast(start, end, metadata)
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
        fail(
            "INVALID_ARGS",
            "start_date and end_date are required (YYYY-MM-DD).",
            exit_code=2,
        )

    run_phase4(args.start_date, args.end_date, args.allow_historical)


if __name__ == "__main__":
    main()
