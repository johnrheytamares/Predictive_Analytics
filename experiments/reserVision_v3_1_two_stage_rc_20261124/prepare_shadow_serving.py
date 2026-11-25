"""Prepare isolated shadow-serving artifacts for ReserVision V3.1 (no production writes)."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PKG = HERE / "package"
ARTIFACTS = HERE / "artifacts"
ROBUST = PKG / "robust_median_artifacts"
FREEZE = PKG / "frozen_selection.json"
OPS_SRC = PKG / "postprocess_evidence" / "ops.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if ARTIFACTS.exists() and any(ARTIFACTS.iterdir()):
        raise SystemExit("Frozen serving artifacts already exist; refusing to rebuild or overwrite them.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    guests_src = ROBUST / "guests_model.joblib"
    bookings_src = ROBUST / "bookings_model.joblib"
    robust_meta = json.loads((ROBUST / "forecast_models_metadata.json").read_text(encoding="utf-8"))
    bookings_bundle = joblib.load(bookings_src)
    guests_bundle = joblib.load(guests_src)

    if guests_bundle.get("model_type") != "robust_median_guests":
        raise SystemExit("Expected robust_median_guests artifact")
    if bookings_bundle.get("model_type") != "weekday_baseline":
        raise SystemExit("Expected bookings history carrier from weekday_baseline bundle")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    two_stage_bundle = {
        "model": None,
        "model_type": "two_stage",
        "baseline_history": bookings_bundle["baseline_history"],
        "target": "bookings",
        "metrics": freeze["postprocess"]["reported_floor_round_bookings_weekly"],
        "metadata": {
            "recipe": freeze["recipe"],
            "source_experiment": freeze["source_experiment"],
            "postprocess": "floor_round",
        },
    }
    joblib.dump(two_stage_bundle, ARTIFACTS / "bookings_model.joblib")
    shutil.copy2(guests_src, ARTIFACTS / "guests_model.joblib")
    shutil.copy2(OPS_SRC, ARTIFACTS / "ops.py")
    shutil.copy2(FREEZE, ARTIFACTS / "frozen_selection.json")

    raw_b = freeze["postprocess"]["reported_raw_bookings_weekly"]
    ops_b = freeze["postprocess"]["reported_floor_round_bookings_weekly"]
    guests_ops = freeze["postprocess"]["guests_floor_round_vs_raw"]

    metadata = {
        "model_version": "3.1",
        "reservision_version": "3.1",
        "trained_at": robust_meta.get("trained_at"),
        "training_days": robust_meta.get("training_days"),
        "minimum_training_days": robust_meta.get("minimum_training_days", 120),
        "date_from": robust_meta.get("date_from"),
        "date_to": robust_meta.get("date_to"),
        "data_source": robust_meta.get("data_source"),
        "is_demo": False,
        "provenance_verified": True,
        "guest_mix_coverage_pct": robust_meta.get("guest_mix_coverage_pct"),
        "guest_mix_ready": robust_meta.get("guest_mix_ready"),
        "guest_mix_proportions": robust_meta.get("guest_mix_proportions"),
        "typical_guests_per_booking": robust_meta.get("typical_guests_per_booking", 7.0),
        "ready": True,
        "blocking_reasons": [],
        "safe_for_live_use": True,
        "deployment_scope": "release-candidate-not-deployed",
        "release_candidate_id": freeze["release_candidate_id"],
        "release_configuration": "two_stage_robust_median_floor_round",
        "production_approved": False,
        "staging_approved": False,
        "shadow_serving_only": True,
        "serving_mode": "isolated_shadow",
        "postprocess": {
            "selected_mode": "floor_round",
            "party_size_floor": False,
            "description": freeze["postprocess"]["description"],
            "source_experiment": freeze["postprocess"]["source_experiment"],
            "variant_a_not_selected": True,
        },
        "evaluation_scope": "retrospective_rolling_28_origins",
        "prospective_validation": "not_run_required_before_production",
        "targets": {
            "bookings": {
                "target": "bookings",
                "selected_model": "two_stage",
                "evaluation_protocol": "rolling_28_monday_origins_7day",
                "selection_metric": "weekly_wape",
                "model_path": "bookings_model.joblib",
                "raw_weekly_metrics": raw_b,
                "operational_weekly_metrics": ops_b,
                "selected_metrics": ops_b,
                "nested_past_only_selected_weekly_wape": freeze["postprocess"][
                    "nested_past_only_selected_weekly_wape"
                ],
                "variant_a_weekly_wape": freeze["postprocess"][
                    "reported_variant_a_bookings_weekly_wape"
                ],
                "metrics_note": (
                    "selected_metrics are floor_round operational weekly scores; "
                    "raw_weekly_metrics preserve the 32.72% selection-protocol headline."
                ),
            },
            "guests": {
                "target": "guests",
                "selected_model": "robust_median_guests",
                "evaluation_protocol": "rolling_28_monday_origins_7day",
                "selection_metric": "weekly_wape",
                "model_path": "guests_model.joblib",
                "raw_weekly_metrics": {"wape": guests_ops["raw_weekly_wape"]},
                "operational_weekly_metrics": {"wape": guests_ops["floor_round_weekly_wape"]},
                "selected_metrics": {"wape": guests_ops["floor_round_weekly_wape"]},
                "variant_a_weekly_wape": guests_ops["variant_a_weekly_wape"],
                "interval_coverage_evaluated": False,
                "interval_status": "Not evaluated",
                "metrics_note": guests_ops["note"],
            },
        },
        # Top-level bookings operational metrics for response compatibility.
        "mae": ops_b["mae"],
        "rmse": ops_b["rmse"],
        "wape": ops_b["wape"],
        "bias": ops_b["bias"],
        "usable_days": robust_meta.get("usable_days") or robust_meta.get("training_days"),
        "training_snapshot": robust_meta.get("training_snapshot"),
        "forecast_policy": {
            "bookings_method": "two_stage",
            "guests_method": "robust_median_guests",
            "postprocess": "floor_round",
            "profile": "cutoff_anchored_four_week_weekday_median",
            "maximum_days_per_cutoff": 90,
            "supported_horizons_days": [7, 30, 90],
            "party_size_floor": False,
            "recursive_predictions": False,
            "fallback": None,
            "interval_status": "Not evaluated",
        },
        "frozen_selection_sha256": sha(FREEZE),
        "prepared_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (ARTIFACTS / "forecast_models_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    # Lightweight companion for tools that also check bookings_model_metadata.json.
    (ARTIFACTS / "bookings_model_metadata.json").write_text(
        json.dumps(
            {
                "model_version": "3.1",
                "selected_model": "two_stage",
                "postprocess": "floor_round",
                "deployment_scope": "release-candidate-not-deployed",
                "production_approved": False,
                "shadow_serving_only": True,
                "date_from": metadata["date_from"],
                "date_to": metadata["date_to"],
                "ready": True,
                "safe_for_live_use": True,
                "is_demo": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    integrity = {
        "shadow_artifacts": {p.name: sha(p) for p in sorted(ARTIFACTS.iterdir()) if p.is_file()},
        "canonical_production_sha256": freeze["canonical_production_sha256"],
        "package_guests_joblib_sha256": sha(guests_src),
        "prepared_at_utc": metadata["prepared_at_utc"],
    }
    (HERE / "shadow_artifacts_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifacts": str(ARTIFACTS),
                "bookings": "two_stage",
                "guests": "robust_median_guests",
                "postprocess": "floor_round",
                "files": sorted(p.name for p in ARTIFACTS.iterdir()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
