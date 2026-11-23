"""Reconstruct the verified 356-day demand series from trusted serving artifacts.

Does not connect to MySQL, does not modify production artifacts, and does not
claim byte-for-byte identity with the original training export file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from periods import (
    RECORDED_INPUT_SHA256,
    SNAPSHOT_BOOKINGS,
    SNAPSHOT_DAYS,
    SNAPSHOT_END,
    SNAPSHOT_GUESTS,
    SNAPSHOT_START,
)

ROOT = Path(__file__).resolve().parents[2]
BOOKINGS_MODEL = ROOT / "bookings_model.joblib"
GUESTS_MODEL = ROOT / "guests_model.joblib"
METADATA = ROOT / "forecast_models_metadata.json"


def serving_hashes() -> dict[str, str]:
    paths = list(ROOT.glob("*.joblib")) + list(ROOT.glob("*metadata.json")) + [
        ROOT / name for name in (
            "api.py",
            "predict_real.py",
            "train_real_model.py",
            "evaluate_forecasts.py",
            "evaluate_weekly_forecasts.py",
        )
        if (ROOT / name).exists()
    ]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def load_baseline_histories() -> tuple[list[dict], list[dict], dict]:
    bookings = joblib.load(BOOKINGS_MODEL)
    guests = joblib.load(GUESTS_MODEL)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    return bookings["baseline_history"], guests["baseline_history"], metadata


def reconstruct_daily_frame() -> tuple[pd.DataFrame, dict]:
    bookings_history, guests_history, metadata = load_baseline_histories()
    if len(bookings_history) != len(guests_history):
        raise ValueError("Bookings and guests baseline histories differ in length")
    if [row["ds"] for row in bookings_history] != [row["ds"] for row in guests_history]:
        raise ValueError("Bookings and guests baseline histories differ in dates")

    frame = pd.DataFrame(
        {
            "demand_date": [row["ds"] for row in bookings_history],
            "bookings": [float(row["y"]) for row in bookings_history],
            "guests": [float(row["y"]) for row in guests_history],
        }
    )
    frame["demand_date"] = pd.to_datetime(frame["demand_date"]).dt.normalize()
    frame["bookings"] = frame["bookings"].round().astype(int)
    frame["guests"] = frame["guests"].round().astype(int)
    frame = frame.sort_values("demand_date").reset_index(drop=True)

    expected_dates = pd.date_range(SNAPSHOT_START, SNAPSHOT_END, freq="D")
    if len(frame) != SNAPSHOT_DAYS:
        raise ValueError(f"Expected {SNAPSHOT_DAYS} days, found {len(frame)}")
    # Compare calendar dates only; pandas may store us vs s datetime units.
    observed_days = frame["demand_date"].dt.strftime("%Y-%m-%d").tolist()
    expected_days = [ts.strftime("%Y-%m-%d") for ts in expected_dates]
    if observed_days != expected_days:
        raise ValueError("Reconstructed series is not a continuous daily calendar")
    if int(frame["bookings"].sum()) != SNAPSHOT_BOOKINGS:
        raise ValueError("Bookings total does not match verified snapshot")
    if int(frame["guests"].sum()) != SNAPSHOT_GUESTS:
        raise ValueError("Guests total does not match verified snapshot")
    if (frame["bookings"] < 0).any() or (frame["guests"] < 0).any():
        raise ValueError("Negative demand values are invalid")
    if (frame["guests"] < frame["bookings"]).any():
        raise ValueError("Guests below bookings on one or more dates")

    snapshot = metadata.get("training_snapshot") or {}
    recorded_sha = snapshot.get("input_sha256")
    if recorded_sha != RECORDED_INPUT_SHA256:
        raise ValueError("Metadata input_sha256 does not match the recorded experiment SHA")

    provenance = {
        "reconstruction_source": "bookings_model.joblib + guests_model.joblib baseline_history",
        "metadata_path": METADATA.name,
        "recorded_input_sha256": recorded_sha,
        "byte_identical_to_original_export": False,
        "content_match": {
            "days": SNAPSHOT_DAYS,
            "bookings": SNAPSHOT_BOOKINGS,
            "guests": SNAPSHOT_GUESTS,
            "date_from": SNAPSHOT_START.isoformat(),
            "date_to": SNAPSHOT_END.isoformat(),
            "matches_metadata_training_snapshot_totals": True,
        },
        "verification_basis": snapshot.get("verification_basis"),
        "zero_demand_days": {
            "bookings": int((frame["bookings"] == 0).sum()),
            "guests": int((frame["guests"] == 0).sum()),
        },
        "nonzero_demand_days": {
            "bookings": int((frame["bookings"] > 0).sum()),
            "guests": int((frame["guests"] > 0).sum()),
        },
    }
    return frame, provenance


def write_dataset(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output directory already exists; refusing to overwrite")
    before = serving_hashes()
    frame, provenance = reconstruct_daily_frame()
    output_dir.mkdir(parents=True)
    data_dir = output_dir / "data"
    data_dir.mkdir()
    csv_path = data_dir / "daily_demand.csv"
    frame.to_csv(csv_path, index=False)
    reconstructed_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    summary = {
        "provenance": provenance,
        "reconstructed_csv_sha256": reconstructed_sha,
        "note": (
            "The reconstructed CSV matches verified totals and the continuous date "
            "range embedded in the trusted artifacts. It is not proven byte-for-byte "
            "identical to the original training export whose SHA-256 is recorded in "
            "forecast_models_metadata.json."
        ),
        "serving_hashes_before": before,
        "serving_hashes_after": serving_hashes(),
    }
    if summary["serving_hashes_before"] != summary["serving_hashes_after"]:
        raise ValueError("Serving artifacts changed during reconstruction")
    summary["serving_hashes_unchanged"] = True
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
