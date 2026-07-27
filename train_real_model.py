"""
Phase 3 – Train and validate forecasting models.

Trains two Prophet models (daily bookings, daily guests) from the validated
`booking_demand_daily` dataset. Refuses to train when readiness checks fail.
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
import pymysql
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_VERSION = "3.0"
MINIMUM_TRAINING_DAYS = int(os.getenv("PREDICTION_MINIMUM_DEMAND_DAYS", "120"))
MINIMUM_GUEST_MIX_COVERAGE_PCT = float(
    os.getenv("PREDICTION_MINIMUM_GUEST_MIX_COVERAGE_PCT", "70")
)
DEFAULT_TEST_DAYS = int(os.getenv("PREDICTION_BACKTEST_DAYS", "28"))
BASELINE_LOOKBACK_WEEKS = 4

ROOT = os.path.dirname(os.path.abspath(__file__))
BOOKINGS_MODEL_PATH = os.path.join(ROOT, "bookings_model.joblib")
GUESTS_MODEL_PATH = os.path.join(ROOT, "guests_model.joblib")
METADATA_PATH = os.path.join(ROOT, "forecast_models_metadata.json")
LEGACY_METADATA_PATH = os.path.join(ROOT, "bookings_model_metadata.json")

DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "eduardos"),
    "port": int(os.getenv("DB_PORT", "3306")),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def print_abort(reasons: list[str], extra: dict | None = None) -> None:
    payload = {
        "ready": False,
        "blocking_reasons": reasons,
        "model_version": MODEL_VERSION,
        "trained": False,
        "message": "Training Aborted",
    }
    if extra:
        payload.update(extra)
    print("Training Aborted\n")
    print("Reason:")
    for reason in reasons:
        print(f"- {reason}")
    print("\n" + json.dumps(payload, indent=2))


def load_demand_dataset() -> pd.DataFrame:
    conn = pymysql.connect(**DB)
    try:
        df = pd.read_sql(
            """
            SELECT
              demand_date,
              bookings,
              guests,
              adults,
              children,
              seniors,
              infants,
              unknown_guests,
              known_guests,
              guest_mix_coverage_pct
            FROM booking_demand_daily
            ORDER BY demand_date ASC
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["demand_date"] = pd.to_datetime(df["demand_date"]).dt.normalize()
    for column in (
        "bookings",
        "guests",
        "adults",
        "children",
        "seniors",
        "infants",
        "unknown_guests",
        "known_guests",
        "guest_mix_coverage_pct",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df.sort_values("demand_date").reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> list[str]:
    reasons: list[str] = []

    if df is None or df.empty:
        reasons.append("Dataset validation failed: booking_demand_daily is empty")
        return reasons

    required = {"demand_date", "bookings", "guests"}
    missing = sorted(required - set(df.columns))
    if missing:
        reasons.append(
            f"Dataset validation failed: missing columns {', '.join(missing)}"
        )
        return reasons

    if df["demand_date"].duplicated().any():
        reasons.append("Duplicate dates found in booking_demand_daily")

    expected = pd.date_range(df["demand_date"].min(), df["demand_date"].max(), freq="D")
    actual_dates = pd.to_datetime(df["demand_date"]).dt.normalize()
    if (
        len(df) != len(expected)
        or not actual_dates.is_monotonic_increasing
        or not actual_dates.reset_index(drop=True).equals(
            pd.Series(expected).dt.normalize().reset_index(drop=True)
        )
    ):
        reasons.append("Continuous daily calendar check failed")

    guest_total = (
        df.get("adults", 0)
        + df.get("children", 0)
        + df.get("seniors", 0)
        + df.get("infants", 0)
        + df.get("unknown_guests", 0)
    )
    if not np.allclose(guest_total.to_numpy(dtype=float), df["guests"].to_numpy(dtype=float)):
        reasons.append("Dataset validation failed: guest category totals do not match guests")

    bad_rows = df[(df["bookings"] > 0) & (df["guests"] < df["bookings"])]
    if not bad_rows.empty:
        reasons.append("Dataset validation failed: guests below bookings on one or more dates")

    return reasons


def assess_readiness(df: pd.DataFrame, provenance_verified: bool = True) -> dict:
    blocking_reasons = validate_dataset(df)

    training_days = int(len(df)) if df is not None and not df.empty else 0
    if training_days < MINIMUM_TRAINING_DAYS:
        blocking_reasons.append(
            f"History below {MINIMUM_TRAINING_DAYS} days ({training_days}/{MINIMUM_TRAINING_DAYS})"
        )

    if not provenance_verified:
        blocking_reasons.append("Provenance not verified")

    total_guests = float(df["guests"].sum()) if training_days else 0.0
    known_guests = float(df["known_guests"].sum()) if training_days and "known_guests" in df else 0.0
    guest_mix_coverage_pct = (
        round((known_guests / total_guests) * 100, 2) if total_guests > 0 else 100.0
    )

    # Guest-mix coverage gates demographic proportion readiness, not the two total models.
    # Still report it so Phase 4+ can reuse the same metadata.
    coverage_ready = guest_mix_coverage_pct >= MINIMUM_GUEST_MIX_COVERAGE_PCT

    # Deduplicate while preserving order
    seen = set()
    unique_reasons = []
    for reason in blocking_reasons:
        if reason not in seen:
            seen.add(reason)
            unique_reasons.append(reason)

    return {
        "ready": len(unique_reasons) == 0,
        "blocking_reasons": unique_reasons,
        "model_version": MODEL_VERSION,
        "training_days": training_days,
        "minimum_training_days": MINIMUM_TRAINING_DAYS,
        "guest_mix_coverage_pct": guest_mix_coverage_pct,
        "guest_mix_ready": coverage_ready,
        "minimum_guest_mix_coverage_pct": MINIMUM_GUEST_MIX_COVERAGE_PCT,
        "date_from": df["demand_date"].min().strftime("%Y-%m-%d") if training_days else None,
        "date_to": df["demand_date"].max().strftime("%Y-%m-%d") if training_days else None,
        "provenance_verified": provenance_verified,
    }


def to_prophet_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(df["demand_date"]),
            "y": pd.to_numeric(df[target], errors="coerce").fillna(0.0),
        }
    )
    return out.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)


def build_prophet() -> Prophet:
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode="additive",
        interval_width=0.8,
    )


def compute_metrics(actual: np.ndarray, predicted: np.ndarray, lower=None, upper=None) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    denom = float(np.sum(np.abs(actual)))
    wape = float((np.sum(np.abs(actual - predicted)) / denom) * 100) if denom > 0 else 0.0
    bias = float(np.mean(predicted - actual)) if len(actual) else 0.0

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
        "bias": round(bias, 4),
        "n": int(len(actual)),
    }

    if lower is not None and upper is not None and len(actual):
        covered = np.logical_and(actual >= lower, actual <= upper)
        metrics["prediction_interval_coverage"] = round(float(np.mean(covered) * 100), 4)

    return metrics


def weekday_baseline_predictions(history_df: pd.DataFrame, test_dates: pd.Series) -> np.ndarray:
    """Prediction = average of the same weekday over the previous 4 weeks."""
    history = history_df.copy()
    history["ds"] = pd.to_datetime(history["ds"])
    history = history.set_index("ds").sort_index()

    preds = []
    for ds in pd.to_datetime(test_dates):
        values = []
        for week in range(1, BASELINE_LOOKBACK_WEEKS + 1):
            prior = ds - pd.Timedelta(days=7 * week)
            if prior in history.index:
                values.append(float(history.loc[prior, "y"]))
        preds.append(float(np.mean(values)) if values else float(history["y"].mean()))
    return np.asarray(preds, dtype=float)


def chronological_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= test_days + 30:
        raise ValueError(
            f"Not enough rows for chronological split. Need > {test_days + 30}, got {len(df)}"
        )
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()
    return train_df, test_df


def evaluate_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict, dict, float, str | None]:
    model = build_prophet()
    model.fit(train_df)

    future = pd.DataFrame({"ds": test_df["ds"]})
    forecast = model.predict(future)
    yhat = np.maximum(0.0, forecast["yhat"].to_numpy(dtype=float))
    lower = np.maximum(0.0, forecast["yhat_lower"].to_numpy(dtype=float))
    upper = np.maximum(0.0, forecast["yhat_upper"].to_numpy(dtype=float))
    actual = test_df["y"].to_numpy(dtype=float)

    prophet_metrics = compute_metrics(actual, yhat, lower=lower, upper=upper)

    # Baseline uses only history available before each test day (train + earlier test days).
    rolling_history = train_df.copy()
    baseline_preds = []
    for _, row in test_df.iterrows():
        baseline_preds.append(
            weekday_baseline_predictions(rolling_history, pd.Series([row["ds"]]))[0]
        )
        rolling_history = pd.concat(
            [rolling_history, pd.DataFrame({"ds": [row["ds"]], "y": [row["y"]]})],
            ignore_index=True,
        )
    baseline_preds = np.asarray(baseline_preds, dtype=float)
    baseline_metrics = compute_metrics(actual, baseline_preds)

    if baseline_metrics["wape"] > 0:
        improvement_pct = (
            (baseline_metrics["wape"] - prophet_metrics["wape"]) / baseline_metrics["wape"]
        ) * 100
    else:
        improvement_pct = 0.0 if prophet_metrics["wape"] == 0 else -100.0

    baseline_note = None
    if float(np.sum(np.abs(actual))) == 0.0:
        baseline_note = "Holdout period contained only zero-demand days."
        baseline_metrics["note"] = baseline_note

    return prophet_metrics, baseline_metrics, float(round(improvement_pct, 4)), baseline_note


def train_full_model(df: pd.DataFrame) -> Prophet:
    model = build_prophet()
    model.fit(df)
    return model


def save_artifacts(
    bookings_model: Prophet,
    guests_model: Prophet,
    bookings_eval: dict,
    guests_eval: dict,
    readiness: dict,
    overall_ready: bool,
) -> dict:
    trained_at = _utc_now_iso()
    baseline_notes = [
        note
        for note in (
            bookings_eval.get("baseline_note"),
            guests_eval.get("baseline_note"),
        )
        if note
    ]
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "training_days": readiness["training_days"],
        "minimum_training_days": readiness["minimum_training_days"],
        "date_from": readiness["date_from"],
        "date_to": readiness["date_to"],
        "data_source": "booking_demand_daily",
        "is_demo": False,
        "provenance_verified": readiness["provenance_verified"],
        "guest_mix_coverage_pct": readiness["guest_mix_coverage_pct"],
        "guest_mix_ready": readiness["guest_mix_ready"],
        "ready": overall_ready,
        "blocking_reasons": [] if overall_ready else readiness.get("blocking_reasons", []),
        "targets": {
            "bookings": {
                **bookings_eval,
                "model_path": os.path.basename(BOOKINGS_MODEL_PATH),
            },
            "guests": {
                **guests_eval,
                "model_path": os.path.basename(GUESTS_MODEL_PATH),
            },
        },
        # Backward-compatible top-level metrics (bookings target)
        "mae": bookings_eval["metrics"]["mae"],
        "rmse": bookings_eval["metrics"]["rmse"],
        "wape": bookings_eval["metrics"]["wape"],
        "bias": bookings_eval["metrics"]["bias"],
        "baseline_improvement_pct": bookings_eval["baseline_improvement_pct"],
        "baseline_note": baseline_notes[0] if baseline_notes else None,
        "usable_days": readiness["training_days"],
        "safe_for_live_use": overall_ready,
    }

    joblib.dump(
        {
            "model": bookings_model,
            "target": "bookings",
            "metrics": bookings_eval["metrics"],
            "metadata": metadata,
        },
        BOOKINGS_MODEL_PATH,
    )
    joblib.dump(
        {
            "model": guests_model,
            "target": "guests",
            "metrics": guests_eval["metrics"],
            "metadata": metadata,
        },
        GUESTS_MODEL_PATH,
    )

    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    # Keep legacy path for existing readiness / predict_real.py readers.
    legacy = {
        "data_source": "live_bookings",
        "trained_at": trained_at,
        "training_start": readiness["date_from"],
        "training_end": readiness["date_to"],
        "usable_days": readiness["training_days"],
        "is_demo": False,
        "safe_for_live_use": overall_ready,
        "model_version": MODEL_VERSION,
        "mae": bookings_eval["metrics"]["mae"],
        "rmse": bookings_eval["metrics"]["rmse"],
        "wape": bookings_eval["metrics"]["wape"],
        "bias": bookings_eval["metrics"]["bias"],
        "baseline_improvement_pct": bookings_eval["baseline_improvement_pct"],
        "baseline_note": baseline_notes[0] if baseline_notes else None,
        "ready": overall_ready,
        "dataset": "booking_demand_daily",
    }
    with open(LEGACY_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(legacy, handle, indent=2)

    return metadata


def train_target(df: pd.DataFrame, target: str, test_days: int) -> tuple[Prophet, dict]:
    series = to_prophet_frame(df, target)
    train_df, test_df = chronological_split(series, test_days=test_days)
    metrics, baseline_metrics, improvement, baseline_note = evaluate_prophet(
        train_df, test_df
    )
    final_model = train_full_model(series)
    beats_baseline = metrics["wape"] <= baseline_metrics["wape"]
    return final_model, {
        "target": target,
        "test_days": int(test_days),
        "train_days": int(len(train_df)),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_improvement_pct": improvement,
        "baseline_note": baseline_note,
        "beats_baseline": beats_baseline,
    }


def run_training(
    provenance_verified: bool = True,
    test_days: int = DEFAULT_TEST_DAYS,
    check_only: bool = False,
) -> dict:
    try:
        df = load_demand_dataset()
    except Exception as exc:  # noqa: BLE001 - surface as readiness failure
        reasons = [f"Dataset not verified: failed to load booking_demand_daily ({exc})"]
        print_abort(reasons)
        return {
            "ready": False,
            "blocking_reasons": reasons,
            "model_version": MODEL_VERSION,
            "trained": False,
        }

    readiness = assess_readiness(df, provenance_verified=provenance_verified)
    if not readiness["ready"]:
        print_abort(readiness["blocking_reasons"], extra={"readiness": readiness})
        return {
            "ready": False,
            "blocking_reasons": readiness["blocking_reasons"],
            "model_version": MODEL_VERSION,
            "trained": False,
            "readiness": readiness,
        }

    if check_only:
        payload = {
            "ready": True,
            "blocking_reasons": [],
            "model_version": MODEL_VERSION,
            "trained": False,
            "check_only": True,
            "readiness": readiness,
        }
        print(json.dumps(payload, indent=2))
        return payload

    bookings_model, bookings_eval = train_target(df, "bookings", test_days=test_days)
    guests_model, guests_eval = train_target(df, "guests", test_days=test_days)

    performance_blockers = []
    if not bookings_eval["beats_baseline"]:
        performance_blockers.append(
            "Bookings Prophet model did not outperform the weekday baseline"
        )
    if not guests_eval["beats_baseline"]:
        performance_blockers.append(
            "Guests Prophet model did not outperform the weekday baseline"
        )

    overall_ready = len(performance_blockers) == 0
    metadata = save_artifacts(
        bookings_model,
        guests_model,
        bookings_eval,
        guests_eval,
        readiness,
        overall_ready=overall_ready,
    )

    result = {
        "ready": overall_ready,
        "blocking_reasons": performance_blockers,
        "model_version": MODEL_VERSION,
        "trained": True,
        "training_days": readiness["training_days"],
        "metadata": metadata,
        "models": {
            "bookings": BOOKINGS_MODEL_PATH,
            "guests": GUESTS_MODEL_PATH,
        },
        "evaluation": {
            "bookings": bookings_eval,
            "guests": guests_eval,
        },
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Prophet training pipeline")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run readiness checks only; do not train or write model files",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_TEST_DAYS,
        help="Chronological holdout length for backtesting",
    )
    parser.add_argument(
        "--provenance-verified",
        type=str,
        default="true",
        help="Whether Phase 2 reported provenance_verified=true",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    provenance_verified = str(args.provenance_verified).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    result = run_training(
        provenance_verified=provenance_verified,
        test_days=max(7, int(args.test_days)),
        check_only=bool(args.check_only),
    )
    # Exit 0 only when readiness passes (check-only) or training is marked ready.
    return 0 if result.get("ready") else 2


if __name__ == "__main__":
    sys.exit(main())
