"""Prepare isolated Phase 2 dataset and feature tables. No model training."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from features import (
    TARGETS,
    assert_horizon_uses_only_precutoff_actuals,
    build_horizon_feature_rows,
    build_supervised_frame,
    cutoffs_with_enough_history,
    feature_column_names,
    series_for_arima,
)
from periods import (
    development_cutoffs,
    period_bounds,
    reserved_cutoffs,
    week_starts,
)
from reconstruct import reconstruct_daily_frame, serving_hashes, write_dataset

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "prepared"


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUTPUT}")
    before = serving_hashes()
    summary = write_dataset(OUTPUT)
    frame, _ = reconstruct_daily_frame()

    supervised = build_supervised_frame(frame, min_history_days=28)
    supervised_path = OUTPUT / "data" / "supervised_features.csv"
    supervised.to_csv(supervised_path, index=False)

    for target in TARGETS:
        series = series_for_arima(frame, target)
        series.to_csv(OUTPUT / "data" / f"arima_{target}_series.csv", header=True)

    all_cutoffs = cutoffs_with_enough_history(frame, week_starts())
    if all_cutoffs != week_starts():
        raise SystemExit("Evaluation cutoffs lack the required 120-day history")

    horizon_rows = []
    for cutoff in all_cutoffs:
        rows = build_horizon_feature_rows(frame, cutoff, horizon_days=7)
        assert_horizon_uses_only_precutoff_actuals(frame, cutoff, rows)
        horizon_rows.append(rows)
    horizon = pd.concat(horizon_rows, ignore_index=True)
    horizon_path = OUTPUT / "data" / "horizon_features_static.csv"
    horizon.to_csv(horizon_path, index=False)

    # Static horizon rows intentionally leave short lags null when the lag
    # source falls inside the forecast week. Count them for documentation.
    short_lag_nulls = {
        f"{target}_lag_1": int(horizon[f"{target}_lag_1"].isna().sum())
        for target in TARGETS
    }

    feature_manifest = {
        "supervised_feature_columns": feature_column_names(include_targets=True),
        "predictor_columns": feature_column_names(include_targets=False),
        "supervised_rows": int(len(supervised)),
        "horizon_rows": int(len(horizon)),
        "evaluation_cutoffs": {
            "development": [d.isoformat() for d in development_cutoffs()],
            "reserved": [d.isoformat() for d in reserved_cutoffs()],
            "periods": period_bounds(),
        },
        "static_horizon_short_lag_nulls": short_lag_nulls,
        "leakage_policy": {
            "supervised": "Features for day t use only dates strictly before t",
            "static_horizon": (
                "At cutoff C, rolling/trend use history before C; lag_k for day "
                "C+offset uses actual y[C+offset-k] only when that date is before C; "
                "otherwise the lag is null until Phase 3 supplies recursive predictions"
            ),
            "arima": "Univariate pre-cutoff series only; no exogenous future targets",
        },
        "models_ready_for": {
            "weekday_baseline": "daily_demand.csv",
            "ridge": "supervised_features.csv + horizon_features_static.csv",
            "random_forest": "supervised_features.csv + horizon_features_static.csv",
            "arima": "arima_bookings_series.csv / arima_guests_series.csv",
        },
    }
    (OUTPUT / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2), encoding="utf-8"
    )

    after = serving_hashes()
    if before != after or summary["serving_hashes_before"] != summary["serving_hashes_after"]:
        raise SystemExit("Serving artifacts changed during preparation")
    print(json.dumps({
        "output": str(OUTPUT),
        "days": int(len(frame)),
        "bookings": int(frame.bookings.sum()),
        "guests": int(frame.guests.sum()),
        "supervised_rows": feature_manifest["supervised_rows"],
        "horizon_rows": feature_manifest["horizon_rows"],
        "serving_hashes_unchanged": True,
        "reconstructed_csv_sha256": summary["reconstructed_csv_sha256"],
        "recorded_input_sha256": summary["provenance"]["recorded_input_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
