"""Horizon compatibility (7/30/90) and Variant A impact on reported two-stage metrics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREV = ROOT / "experiments/intermittent_bookings_20261124"
SOURCE = ROOT / "experiments/multimodel_dataset_20261123/prepared/data/daily_demand.csv"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from evaluate_forecasts import load_history, metrics
import predict_real as serving

spec = importlib.util.spec_from_file_location("intermittent", PREV / "experiment.py")
intermittent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intermittent)

pp_spec = importlib.util.spec_from_file_location(
    "phase5_postprocess",
    ROOT / "experiments/postprocess_phase5_20261123/postprocess.py",
)
postprocess = importlib.util.module_from_spec(pp_spec)
pp_spec.loader.exec_module(postprocess)
apply_variant_a = postprocess.apply_variant_a


def two_stage_forecast(history: pd.Series, dates, cfg: dict) -> np.ndarray:
    return intermittent.predict(history, dates, "two_stage", cfg)


def robust_guests(history_guests: pd.Series, cutoff: pd.Timestamp, dates) -> np.ndarray:
    bundle = {
        "model_type": serving.ROBUST_MEDIAN_GUESTS,
        "baseline_history": [
            {"ds": str(ts.date()), "y": float(v)} for ts, v in history_guests.items()
        ],
    }
    rows = serving.predict_series(bundle, pd.DatetimeIndex(dates))
    return np.asarray([r["prediction"] for r in rows], float)


def default_cfg_for_forward(history: pd.Series, horizon: int = 7) -> dict:
    """Past-only nested tune using the original intermittent grid (no new optimization)."""
    return intermittent.tune(history, "two_stage", horizon)


def main() -> None:
    freeze = json.loads((HERE / "package/frozen_selection.json").read_text(encoding="utf-8"))
    data = load_history(SOURCE)
    bookings = data.set_index("demand_date").bookings.astype(float)
    guests = data.set_index("demand_date").guests.astype(float)
    params = {
        (s["protocol"], s["origin"]): s["parameters"]
        for s in freeze["per_origin_parameters"]
        if s["method"] == "two_stage" and s["protocol"] == "rolling_7day"
    }

    # --- Postprocess impact on the frozen 28-origin protocol ---
    raw_rows = []
    ops_rows = []
    for origin, cfg in sorted((o, c) for (p, o), c in params.items()):
        origin_ts = pd.Timestamp(origin)
        train_b = bookings[bookings.index < origin_ts]
        train_g = guests[guests.index < origin_ts]
        test_dates = pd.date_range(origin_ts, periods=7)
        raw_b = two_stage_forecast(train_b, test_dates, cfg)
        raw_g = robust_guests(train_g, origin_ts, test_dates)
        op_b, op_g = apply_variant_a(raw_b, raw_g)
        actual_b = bookings.loc[test_dates].to_numpy(float)
        for i, day in enumerate(test_dates):
            raw_rows.append(dict(origin=origin, week=0, date=str(day.date()), actual=actual_b[i], predicted=float(raw_b[i])))
            ops_rows.append(dict(origin=origin, week=0, date=str(day.date()), actual=actual_b[i], predicted=float(op_b[i])))

    raw_df = pd.DataFrame(raw_rows)
    ops_df = pd.DataFrame(ops_rows)
    raw_weekly = raw_df.groupby(["origin", "week"])[["actual", "predicted"]].sum()
    ops_weekly = ops_df.groupby(["origin", "week"])[["actual", "predicted"]].sum()
    raw_m = metrics(raw_weekly.actual, raw_weekly.predicted)
    ops_m = metrics(ops_weekly.actual, ops_weekly.predicted)

    # Sanity: raw weekly WAPE must still match frozen headline.
    assert abs(raw_m["wape"] - 32.72234791049451) < 1e-8, raw_m

    # --- Forward 7/30/90 structural compatibility after history end ---
    cutoff = bookings.index.max() + pd.Timedelta(days=1)  # 2026-11-22
    train_b = bookings[bookings.index < cutoff]
    train_g = guests[guests.index < cutoff]
    cfg = default_cfg_for_forward(train_b, horizon=7)
    horizon_results = {}
    for days in (7, 30, 90):
        dates = pd.date_range(cutoff, periods=days)
        raw_b = two_stage_forecast(train_b, dates, cfg)
        raw_g = robust_guests(train_g, cutoff, dates)
        op_b, op_g = apply_variant_a(raw_b, raw_g)
        assert len(raw_b) == days and np.isfinite(raw_b).all() and (raw_b >= 0).all()
        assert len(raw_g) == days and np.isfinite(raw_g).all() and (raw_g >= 0).all()
        for b, g in zip(op_b, op_g):
            assert isinstance(b, int) and isinstance(g, int)
            assert g >= 7 * b
            if g > 0:
                assert b >= 1
        horizon_results[str(days)] = dict(
            days=days,
            start=str(cutoff.date()),
            end=str(dates[-1].date()),
            raw_bookings_sum=float(raw_b.sum()),
            raw_guests_sum=float(raw_g.sum()),
            operational_bookings_sum=int(sum(op_b)),
            operational_guests_sum=int(sum(op_g)),
            cfg=cfg,
        )

    integrity = json.loads((HERE / "source_integrity.json").read_text(encoding="utf-8"))
    now = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in integrity["canonical_sha256"]}
    assert now == integrity["canonical_sha256"]

    payload = {
        "raw_rolling_weekly": raw_m,
        "operational_variant_a_rolling_weekly": ops_m,
        "postprocess_changes_reported_wape": abs(ops_m["wape"] - raw_m["wape"]) > 1e-9,
        "postprocess_delta_weekly_wape_pp": ops_m["wape"] - raw_m["wape"],
        "interpretation": (
            "Reported Version 3.1 headline metrics are raw expected booking counts from the intermittent "
            "experiment. Variant A (booking-min + guest floor) changes integer operational bookings when "
            "paired with Robust Median guests; operational weekly WAPE therefore differs and must be "
            "tracked separately. Raw headline 32.72% remains valid for the experimental selection protocol."
        ),
        "horizons": horizon_results,
        "canonical_unchanged": True,
        "guests_model": "robust_median_guests",
        "bookings_model": "two_stage",
    }
    (HERE / "verification_horizons_postprocess.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "raw_weekly_wape": raw_m["wape"],
        "ops_weekly_wape": ops_m["wape"],
        "delta_pp": ops_m["wape"] - raw_m["wape"],
        "horizons_ok": list(horizon_results),
        "canonical_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
