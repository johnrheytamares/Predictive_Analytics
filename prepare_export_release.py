"""Build an isolated serving release from a confirmed aggregate SQL/CSV export.

Never connects to MySQL or overwrites the repository's serving artifacts.
The output directory must not exist. No deployment or Git operations occur.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from evaluate_forecasts import load_history
import train_real_model as training
import predict_real as inference


def prepare(source, output, *, expected_days, expected_bookings, expected_guests):
    source, output = Path(source), Path(output)
    data = load_history(source)
    observed = (len(data), int(data.bookings.sum()), int(data.guests.sum()))
    if observed != (expected_days, expected_bookings, expected_guests):
        raise ValueError(f"Export totals do not match confirmed counts: {observed}")
    if output.exists():
        raise ValueError("Output must be a new directory; existing releases are never overwritten")
    demographic_columns = ("adults", "children", "seniors", "infants", "known_guests", "unknown_guests")
    available = [name in data.columns for name in demographic_columns]
    if any(available) and not all(available):
        raise ValueError("Incomplete demographic export: require all six guest-count columns")
    has_demographics = all(available)
    if has_demographics:
        import numpy as np
        for column in demographic_columns:
            data[column] = pd.to_numeric(data[column], errors="raise")
            values = data[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
                raise ValueError(f"Invalid demographic counts in {column}")
        if not (data[["adults", "children", "seniors", "infants"]].sum(axis=1) == data.known_guests).all():
            raise ValueError("Known guests must match category sums on every date")
        if not (data.known_guests + data.unknown_guests == data.guests).all():
            raise ValueError("Known + unknown guests must match totals on every date")
    else:
        # Preserve uncertainty for totals-only exports; never reuse old ratios.
        for column in demographic_columns[:-1]:
            data[column] = 0
        data["unknown_guests"] = data.guests
    data["guest_mix_coverage_pct"] = (
        data.known_guests.div(data.guests.where(data.guests > 0)).fillna(1) * 100
    )
    readiness = training.assess_readiness(data, provenance_verified=True)
    if not readiness["ready"]:
        raise ValueError(readiness["blocking_reasons"])
    models = {}
    evaluations = {}
    for target in ("bookings", "guests"):
        models[target], evaluations[target] = training.train_target(data, target, 28)
    output.mkdir(parents=True)
    original_paths = {name: getattr(training, name) for name in (
        "BOOKINGS_MODEL_PATH", "GUESTS_MODEL_PATH", "METADATA_PATH", "LEGACY_METADATA_PATH")}
    try:
        for name, filename in (
            ("BOOKINGS_MODEL_PATH", "bookings_model.joblib"),
            ("GUESTS_MODEL_PATH", "guests_model.joblib"),
            ("METADATA_PATH", "forecast_models_metadata.json"),
            ("LEGACY_METADATA_PATH", "bookings_model_metadata.json"),
        ):
            setattr(training, name, str(output / filename))
        metadata = training.save_artifacts(models["bookings"], models["guests"],
            evaluations["bookings"], evaluations["guests"], readiness, overall_ready=True)
    finally:
        for name, value in original_paths.items():
            setattr(training, name, value)
    # Store an auditable snapshot identity inside JSON and both bundles.
    metadata["training_snapshot"] = {
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "days": observed[0], "bookings": observed[1], "guests": observed[2],
        "verification_basis": "User-confirmed server aggregate export; source register not independently audited",
        "guest_mix_basis": ("Category counts from user-confirmed server export; validated against daily totals"
                            if has_demographics else "Unavailable: input contains only daily bookings and total guests"),
        "test_from": str(data.demand_date.iloc[-28].date()),
        "test_to": str(data.demand_date.iloc[-1].date()),
        "test_bookings": int(data.bookings.tail(28).sum()),
        "test_guests": int(data.guests.tail(28).sum()),
    }
    metadata["training_environment"] = {name: importlib.metadata.version(name) for name in (
        "prophet", "pandas", "numpy", "joblib", "scikit-learn", "cmdstanpy")}
    # Compatibility fields describe the selected booking predictor, too.
    selected = evaluations["bookings"]["selected_metrics"]
    for field in ("mae", "rmse", "wape", "bias"):
        metadata[field] = selected[field]
    (output / "forecast_models_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    legacy_path = output / "bookings_model_metadata.json"
    legacy = json.loads(legacy_path.read_text())
    for field in ("mae", "rmse", "wape", "bias"):
        legacy[field] = selected[field]
    legacy_path.write_text(json.dumps(legacy, indent=2, allow_nan=False))
    bundles = []
    for target in ("bookings", "guests"):
        path = output / f"{target}_model.joblib"
        bundle = joblib.load(path)
        bundle["metadata"] = metadata
        bundle["metrics"] = evaluations[target]["selected_metrics"]
        joblib.dump(bundle, path)
        reloaded = joblib.load(path)
        history = pd.DataFrame(reloaded["baseline_history"])
        if len(history) != expected_days or history.y.sum() != observed[1 if target == "bookings" else 2]:
            raise ValueError("Serialized history failed verification")
        bundles.append(reloaded)
    root = Path(__file__).resolve().parent
    for filename in ("api.py", "predict_real.py", "requirements.txt", "Dockerfile"):
        shutil.copy2(root / filename, output / filename)
    # Load the new artifacts in a fresh process using the copied serving code.
    import subprocess
    check = subprocess.run([
        "python", "-B", "predict_real.py", "--start-date",
        str((data.demand_date.max() + pd.Timedelta(days=1)).date()),
        "--end-date", str((data.demand_date.max() + pd.Timedelta(days=30)).date()),
    ], cwd=output, capture_output=True, text=True, check=True)
    served = json.loads(check.stdout)
    if served["model"]["training_days"] != expected_days or len(served["forecast"]) != 30:
        raise ValueError("Serving smoke check failed")
    summary = {
        "output": str(output), "snapshot": metadata["training_snapshot"],
        "selected": {target: {"method": evaluations[target]["selected_model"],
                              "metrics": evaluations[target]["selected_metrics"]}
                     for target in ("bookings", "guests")},
        "guest_mix_ready": metadata["guest_mix_ready"],
        "smoke_forecast_bookings": sum(row["predicted_bookings"] for row in served["forecast"]),
        "smoke_forecast_guests": sum(row["predicted_total_guests"] for row in served["forecast"]),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confirmed-server-export", action="store_true", required=True)
    parser.add_argument("--expected-days", type=int, required=True)
    parser.add_argument("--expected-bookings", type=int, required=True)
    parser.add_argument("--expected-guests", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, expected_days=args.expected_days,
                            expected_bookings=args.expected_bookings,
                            expected_guests=args.expected_guests), indent=2, allow_nan=False))
