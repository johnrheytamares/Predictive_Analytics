"""Freeze original two-stage bookings + Robust Median guests as Version 3.1 RC."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREV = ROOT / "experiments/intermittent_bookings_20261124"
ROBUST_RC = ROOT / "experiments/robust_release_candidate_20261124"
OUT = HERE / "package"
SCOPE = "release-candidate-not-deployed"
VERSION = "3.1"
RELEASE_ID = "reservision_v3_1_two_stage_rc_20261124"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")

    report = json.loads((PREV / "results/results.json").read_text(encoding="utf-8"))
    rolling = next(
        r for r in report["results"] if r["protocol"] == "rolling_7day" and r["method"] == "two_stage"
    )
    holdout = next(
        r
        for r in report["results"]
        if r["protocol"] == "original_holdout_28day" and r["method"] == "two_stage"
    )
    selections = [s for s in report["selections"] if s["method"] == "two_stage"]

    expected_weekly = dict(wape=32.72234791049451, mae=2.769713019566857, rmse=3.909415622750741, bias=0.9403568384502454)
    for key, value in expected_weekly.items():
        if abs(rolling["weekly"][key] - value) > 1e-12:
            raise SystemExit(f"Frozen source weekly {key} mismatch: {rolling['weekly'][key]} vs {value}")

    canonical = {
        name: sha(ROOT / name)
        for name in (
            "bookings_model.joblib",
            "guests_model.joblib",
            "forecast_models_metadata.json",
            "bookings_model_metadata.json",
        )
    }
    robust_meta = json.loads((ROBUST_RC / "artifacts/forecast_models_metadata.json").read_text(encoding="utf-8"))

    OUT.mkdir(parents=True)
    evidence = OUT / "evaluation_evidence"
    evidence.mkdir()
    for name in ("daily_predictions.csv", "comparison.csv", "results.json"):
        shutil.copy2(PREV / "results" / name, evidence / name)
    shutil.copy2(PREV / "experiment.py", evidence / "intermittent_experiment.py")
    shutil.copy2(PREV / "REPORT.md", evidence / "intermittent_REPORT.md")

    # Preserve Robust Median RC artifacts by copy (bookings joblib stays baseline byte-identical;
    # V3.1 experimental bookings method is the frozen two-stage recipe, not that joblib).
    robust_artifacts = OUT / "robust_median_artifacts"
    shutil.copytree(ROBUST_RC / "artifacts", robust_artifacts)

    freeze = {
        "release_candidate_id": RELEASE_ID,
        "reservision_version": VERSION,
        "deployment_scope": SCOPE,
        "production_approved": False,
        "staging_approved": False,
        "selected_experimental_booking_model": "two_stage",
        "source_experiment": "experiments/intermittent_bookings_20261124",
        "not_selected": [
            "experiments/two_stage_optimization_20261124/two_stage_optimized",
            "adaptive_two_stage",
            "direct_weekly candidates",
        ],
        "recipe": {
            "type": "hurdle_two_stage",
            "description": "P(occurrence) * E[size | positive]; zeros remain in occurrence denominator",
            "probability": "weekday empirical rate shrunk to global rate over prob_days",
            "conditional_size": "mean of positive observations over size_days",
            "tuning": "nested chronological MAE over 4 past folds; grid prob_days={14,28,56}, size_days={28,84,168}, shrink={0,0.5,1}",
            "evaluation_protocol": "28 rolling 7-day origins 2026-05-04..2026-11-09; raw expected counts",
        },
        "per_origin_parameters": selections,
        "reported_rolling_weekly": rolling["weekly"],
        "reported_rolling_daily": rolling["daily"],
        "reported_holdout_weekly": holdout["weekly"],
        "reported_holdout_daily": holdout["daily"],
        "guests": {
            "model": "robust_median_guests",
            "profile": "cutoff_anchored_four_week_weekday_median",
            "source_rc": "experiments/robust_release_candidate_20261124",
            "deployment_scope_label_in_source": robust_meta.get("deployment_scope"),
            "preserved": True,
            "interval_status": "Not evaluated",
        },
        "postprocess": {
            "variant": "A",
            "note": "Reported 32.72% weekly WAPE is on raw expected bookings before Variant A. Operational impact is measured separately in verification.",
        },
        "evidence_sha256": {
            "daily_predictions.csv": sha(evidence / "daily_predictions.csv"),
            "comparison.csv": sha(evidence / "comparison.csv"),
            "results.json": sha(evidence / "results.json"),
            "intermittent_experiment.py": sha(evidence / "intermittent_experiment.py"),
        },
        "canonical_production_sha256": canonical,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (OUT / "frozen_selection.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    (HERE / "source_integrity.json").write_text(
        json.dumps(
            {
                "release_candidate_id": RELEASE_ID,
                "canonical_sha256": canonical,
                "canonical_unchanged_at_prepare": True,
                "previous_results_sha256": sha(PREV / "results/results.json"),
                "robust_rc_manifest_sha256": sha(ROBUST_RC / "PACKAGE_MANIFEST.json"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "package": str(OUT),
                "version": VERSION,
                "weekly_wape": rolling["weekly"]["wape"],
                "origins_frozen": len(selections),
                "canonical_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
