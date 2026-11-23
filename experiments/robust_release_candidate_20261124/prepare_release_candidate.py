"""Build Robust Median release-candidate artifacts. Never touches canonical files."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import predict_real as p

CANONICAL = [
    "bookings_model.joblib",
    "guests_model.joblib",
    "forecast_models_metadata.json",
    "bookings_model_metadata.json",
]
OUT = HERE / "artifacts"
RELEASE_ID = "robust_median_rc_20261124"
SCOPE = "release-candidate-not-deployed"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> dict:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing release candidate at {OUT}")

    before = {name: sha256(ROOT / name) for name in CANONICAL}
    bookings_src = ROOT / "bookings_model.joblib"
    guests_src = ROOT / "guests_model.joblib"
    meta_src = ROOT / "forecast_models_metadata.json"
    legacy_src = ROOT / "bookings_model_metadata.json"

    bookings = joblib.load(bookings_src)
    guests = copy.copy(joblib.load(guests_src))
    if bookings.get("model_type") != "weekday_baseline" or guests.get("model_type") != "weekday_baseline":
        raise SystemExit("Canonical bundles must both be weekday_baseline before RC preparation")
    if (bookings.get("baseline_history") or []) != (guests.get("baseline_history") or []):
        # Histories are per-target series; equality of length/dates is enough for shared calendar.
        b_dates = [str(r["ds"])[:10] for r in bookings["baseline_history"]]
        g_dates = [str(r["ds"])[:10] for r in guests["baseline_history"]]
        if b_dates != g_dates:
            raise SystemExit("Bookings and guests histories must share the same dates")

    guests["model_type"] = p.ROBUST_MEDIAN_GUESTS

    meta = json.loads(meta_src.read_text(encoding="utf-8"))
    meta["deployment_scope"] = SCOPE
    meta["release_candidate_id"] = RELEASE_ID
    meta["release_configuration"] = "robust_median_guests_variant_a"
    meta.pop("local_configuration", None)
    meta["source_trained_at"] = meta.get("trained_at")
    meta["production_approved"] = False
    meta["staging_approved"] = False

    guest_target = meta["targets"]["guests"]
    guest_target.update(
        selected_model=p.ROBUST_MEDIAN_GUESTS,
        selected_metrics={},
        metrics={},
        baseline_metrics={},
        evaluation_protocol="offline_phase5_fixed_cutoffs",
        selection_metric="wape",
        recipe_evaluation={
            "development_weekly_wape": 32.087330466424085,
            "previously_examined_reserved_weekly_wape": 80.0,
            "independently_validated": False,
            "source": "experiments/final_ml_evaluation_freeze_20261123",
        },
        interval_coverage_evaluated=False,
        interval_status="Not evaluated",
        interval_note="Daily residual bands are uncalibrated; not validated prediction intervals.",
    )
    if meta["targets"]["bookings"].get("selected_model") != "weekday_baseline":
        raise SystemExit("Bookings selected_model must remain weekday_baseline")

    cutoff = pd.Timestamp(meta["date_to"]) + pd.Timedelta(days=1)
    policy = p.robust_capabilities(guests, cutoff)
    if policy.get("maximum_days_per_cutoff") != p.MAX_FORECAST_DAYS:
        raise SystemExit("forecast_policy maximum_days_per_cutoff must equal MAX_FORECAST_DAYS")
    if policy.get("profile") != "cutoff_anchored_four_week_weekday_median":
        raise SystemExit("forecast_policy profile mismatch")
    if policy.get("recursive_predictions") is not False:
        raise SystemExit("forecast_policy must disable recursive predictions")
    policy["interval_coverage_evaluated"] = False
    policy["interval_status"] = "Not evaluated"
    policy["interval_note"] = "Daily residual bands are uncalibrated; not validated prediction intervals."
    meta["forecast_policy"] = policy
    meta["postprocess"] = {
        "variant": "A",
        "description": "Production Variant A: booking minimum when guests>0; guest floor from typical_guests_per_booking",
        "typical_guests_per_booking": meta.get("typical_guests_per_booking"),
    }

    p.validate_robust_configuration(bookings, guests, meta)

    OUT.mkdir(parents=True)
    # Prefer byte-identical bookings + legacy metadata; only guests/metadata are rewritten.
    shutil.copy2(bookings_src, OUT / "bookings_model.joblib")
    shutil.copy2(legacy_src, OUT / "bookings_model_metadata.json")
    joblib.dump(guests, OUT / "guests_model.joblib")
    (OUT / "forecast_models_metadata.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    after_canonical = {name: sha256(ROOT / name) for name in CANONICAL}
    if after_canonical != before:
        raise SystemExit("Canonical artifacts changed during RC preparation")

    rc_hashes = {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    if rc_hashes["bookings_model.joblib"] != before["bookings_model.joblib"]:
        raise SystemExit("Bookings artifact must remain byte-identical to canonical")
    if rc_hashes["bookings_model_metadata.json"] != before["bookings_model_metadata.json"]:
        raise SystemExit("Legacy bookings metadata must remain byte-identical to canonical")

    serving = {
        "predict_real.py": sha256(ROOT / "predict_real.py"),
        "api.py": sha256(ROOT / "api.py"),
        "Dockerfile": sha256(ROOT / "Dockerfile"),
        "requirements.txt": sha256(ROOT / "requirements.txt"),
    }

    integrity = {
        "release_candidate_id": RELEASE_ID,
        "deployment_scope": SCOPE,
        "production_approved": False,
        "staging_approved": False,
        "canonical_sha256_before": before,
        "canonical_unchanged": True,
        "release_candidate_sha256": rc_hashes,
        "serving_code_sha256": serving,
        "bookings_byte_identical_to_canonical": True,
        "configuration": {
            "bookings": "weekday_baseline",
            "guests": p.ROBUST_MEDIAN_GUESTS,
            "postprocess": "Variant A",
            "profile": policy["profile"],
            "maximum_days_per_cutoff": policy["maximum_days_per_cutoff"],
            "interval_status": "Not evaluated",
        },
    }
    (HERE / "source_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    return {
        "artifacts": str(OUT),
        "deployment_scope": SCOPE,
        "forecast_policy_maximum_days": policy["maximum_days_per_cutoff"],
        "supported_date_count": len(policy["supported_dates"]),
        "canonical_unchanged": True,
        "bookings_byte_identical": True,
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
