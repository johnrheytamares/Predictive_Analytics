"""Amend V3.1 RC freeze to include selected floor_round ops. No production writes."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OPS = ROOT / "experiments/v3_1_postprocessing_optimization_20261124"
PKG = HERE / "package"
FREEZE = PKG / "frozen_selection.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    import pandas as pd

    ops_best = json.loads((OPS / "best_config.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(OPS / "results/comparison.csv")
    ts = comparison[comparison.model == "two_stage"]

    def row(ops: str) -> pd.Series:
        return ts[ts.ops == ops].iloc[0]

    nested_wape = float(row("selected_past_only").bookings_weekly_wape)
    guests_raw = float(row("raw_float").guests_weekly_wape)
    guests_floor = float(row("floor_round").guests_weekly_wape)
    guests_a = float(row("variant_a").guests_weekly_wape)
    # Keep prior freeze bytes for provenance.
    prior_sha = sha(FREEZE)
    shutil.copy2(FREEZE, PKG / "frozen_selection_before_floor_round.json")

    evidence = PKG / "postprocess_evidence"
    evidence.mkdir(exist_ok=True)
    for name in (
        "bookings_ops_ranking.csv",
        "comparison.csv",
        "daily_predictions.csv",
        "results.json",
    ):
        shutil.copy2(OPS / "results" / name, evidence / name)
    shutil.copy2(OPS / "best_config.json", evidence / "best_config.json")
    shutil.copy2(OPS / "REPORT.md", evidence / "postprocess_REPORT.md")
    shutil.copy2(OPS / "ops.py", evidence / "ops.py")

    freeze["postprocess"] = {
        "selected_mode": "floor_round",
        "source_experiment": "experiments/v3_1_postprocessing_optimization_20261124",
        "description": ops_best["best_operational_configuration"]["description"],
        "party_size_floor": False,
        "variant_a_not_selected": True,
        "nested_past_only_selected_weekly_wape": nested_wape,
        "nested_note": "Preserved for evidence; not the frozen serving ops mode. Fixed floor_round outperformed nested ops selection.",
        "reported_raw_bookings_weekly": {
            "wape": 32.72234791049451,
            "mae": 2.769713019566857,
            "rmse": 3.909415622750741,
            "bias": 0.9403568384502454,
        },
        "reported_floor_round_bookings_weekly": {
            "wape": float(row("floor_round").bookings_weekly_wape),
            "mae": float(row("floor_round").bookings_weekly_mae),
            "rmse": float(row("floor_round").bookings_weekly_rmse),
            "bias": float(row("floor_round").bookings_weekly_bias),
        },
        "reported_variant_a_bookings_weekly_wape": 38.39662447257384,
        "guests_floor_round_vs_raw": {
            "note": "Floor_round barely changes guest weekly WAPE vs raw Robust Median; Variant A worsens guests.",
            "raw_weekly_wape": guests_raw,
            "floor_round_weekly_wape": guests_floor,
            "variant_a_weekly_wape": guests_a,
        },
        "evaluation_scope": "retrospective_rolling_28_origins",
        "prospective_validation": "required_before_production",
    }
    freeze["configuration_complete"] = {
        "bookings": "two_stage",
        "guests": "robust_median_guests",
        "postprocess": "floor_round",
        "reservision_version": freeze["reservision_version"],
        "deployment_scope": "release-candidate-not-deployed",
        "production_approved": False,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prior_freeze_sha256": prior_sha,
    }
    freeze["production_approved"] = False
    freeze["deployment_scope"] = "release-candidate-not-deployed"

    FREEZE.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    (HERE / "source_integrity.json").write_text(
        json.dumps(
            {
                **json.loads((HERE / "source_integrity.json").read_text(encoding="utf-8")),
                "floor_round_update_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "postprocess_experiment_results_sha256": sha(OPS / "results/results.json"),
                "ops_module_sha256": sha(OPS / "ops.py"),
                "canonical_sha256": {
                    name: sha(ROOT / name)
                    for name in (
                        "bookings_model.joblib",
                        "guests_model.joblib",
                        "forecast_models_metadata.json",
                        "bookings_model_metadata.json",
                    )
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "updated": str(FREEZE),
                "postprocess": "floor_round",
                "raw_wape": 32.72234791049451,
                "floor_round_wape": ops_best["best_operational_configuration"]["rolling_bookings_weekly"][
                    "wape"
                ],
                "nested_preserved_wape": 35.0210970464135,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
