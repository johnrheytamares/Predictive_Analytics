"""Phase 5 operational post-processing evaluation. No retraining or production edits."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from statistics import median

import pandas as pd

from postprocess import (
    PRODUCTION_PARTY_SIZE,
    apply_variant_a,
    apply_variant_b,
    guest_floor_changes,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASE2_DAILY = ROOT / "experiments" / "multimodel_dataset_20261123" / "prepared" / "data" / "daily_demand.csv"
PHASE3_DAILY = ROOT / "experiments" / "multimodel_compare_20261123" / "results" / "daily_predictions.csv"
PHASE3_ARIMA = ROOT / "experiments" / "multimodel_compare_20261123" / "results" / "arima" / "daily_predictions.csv"
EXPECTED_DAILY_SHA = "7017de3fb24a1105e3a09872526367aa40ce1b05384e28065de99a1464ca1cf7"
OUTPUT = HERE / "results"

FIRST = date(2026, 4, 6)
DEV_WEEKS = 24
RES_WEEKS = 8
HIGH_GUEST_DAY = 30  # absolute high-demand day threshold for underprediction tally


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serving_hashes() -> dict[str, str]:
    names = [
        "bookings_model.joblib",
        "guests_model.joblib",
        "forecast_models_metadata.json",
        "bookings_model_metadata.json",
        "api.py",
        "predict_real.py",
        "train_real_model.py",
    ]
    return {n: sha(ROOT / n) for n in names if (ROOT / n).exists()}


def scores(actual, predicted) -> dict:
    actual = [float(v) for v in actual]
    predicted = [float(v) for v in predicted]
    errors = [p - a for a, p in zip(actual, predicted)]
    absolute = sum(abs(e) for e in errors)
    total = sum(actual)
    return {
        "n": len(actual),
        "actual_total": total,
        "predicted_total": sum(predicted),
        "absolute_error": absolute,
        "mae": absolute / len(actual),
        "rmse": (sum(e * e for e in errors) / len(actual)) ** 0.5,
        "wape": (100 * absolute / total) if total > 0 else None,
        "bias": sum(errors) / len(actual),
    }


def summarize(week_parts: list[tuple[list[float], list[float]]]) -> dict:
    weekly_a = [sum(a) for a, _ in week_parts]
    weekly_p = [sum(p) for _, p in week_parts]
    daily_a = [v for a, _ in week_parts for v in a]
    daily_p = [v for _, p in week_parts for v in p]
    zero_days = [(a, p) for a, p in zip(daily_a, daily_p) if a == 0]
    high_days = [(a, p) for a, p in zip(daily_a, daily_p) if a >= HIGH_GUEST_DAY]
    return {
        "weeks": len(week_parts),
        "weekly": scores(weekly_a, weekly_p),
        "daily": scores(daily_a, daily_p),
        "zero_demand_days": len(zero_days),
        "zero_day_false_positives": sum(1 for _, p in zero_days if p > 0),
        "zero_day_false_positive_total": sum(p for _, p in zero_days if p > 0),
        "high_demand_days": len(high_days),
        "high_demand_underpredictions": sum(1 for a, p in high_days if p < a),
        "high_demand_underprediction_total": sum(max(0.0, a - p) for a, p in high_days),
    }


def load_demand() -> pd.DataFrame:
    if sha(PHASE2_DAILY) != EXPECTED_DAILY_SHA:
        raise ValueError("Phase 2 daily_demand.csv SHA mismatch")
    frame = pd.read_csv(PHASE2_DAILY)
    frame["demand_date"] = pd.to_datetime(frame["demand_date"]).dt.normalize()
    if len(frame) != 356 or int(frame.bookings.sum()) != 519 or int(frame.guests.sum()) != 6669:
        raise ValueError("Phase 2 totals failed verification")
    return frame.sort_values("demand_date").reset_index(drop=True)


def load_predictions() -> pd.DataFrame:
    base = pd.read_csv(PHASE3_DAILY)
    arima = pd.read_csv(PHASE3_ARIMA)
    frame = pd.concat([base, arima], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["cutoff"] = pd.to_datetime(frame["cutoff"]).dt.normalize()
    return frame


def verify_against_demand(preds: pd.DataFrame, demand: pd.DataFrame) -> None:
    merged = preds.merge(
        demand,
        left_on="date",
        right_on="demand_date",
        how="left",
        validate="many_to_one",
    )
    if merged["demand_date"].isna().any():
        raise ValueError("Prediction dates missing from demand series")
    bookings = merged[merged.target == "bookings"]
    guests = merged[merged.target == "guests"]
    if not (bookings["actual"] == bookings["bookings"]).all():
        raise ValueError("Bookings actuals do not match demand series")
    if not (guests["actual"] == guests["guests"]).all():
        raise ValueError("Guests actuals do not match demand series")


def pivot_week(preds: pd.DataFrame, bookings_method: str, guests_method: str, period: str) -> list[dict]:
    """Return one dict per cutoff with aligned 7-day raw bookings/guests/actuals."""
    b = preds[(preds.method == bookings_method) & (preds.target == "bookings") & (preds.period == period)]
    g = preds[(preds.method == guests_method) & (preds.target == "guests") & (preds.period == period)]
    if b.empty or g.empty:
        raise ValueError(f"Missing predictions for {bookings_method}/{guests_method}/{period}")
    weeks = []
    for cutoff, b_week in b.groupby("cutoff"):
        g_week = g[g.cutoff == cutoff].sort_values("horizon_offset")
        b_week = b_week.sort_values("horizon_offset")
        if len(b_week) != 7 or len(g_week) != 7:
            raise ValueError(f"Incomplete week at {cutoff}")
        if list(b_week.date) != list(g_week.date):
            raise ValueError(f"Date mismatch at {cutoff}")
        weeks.append(
            {
                "cutoff": pd.Timestamp(cutoff).date().isoformat(),
                "dates": [d.date().isoformat() for d in b_week.date],
                "actual_bookings": b_week.actual.astype(float).tolist(),
                "actual_guests": g_week.actual.astype(float).tolist(),
                "raw_bookings": b_week.predicted.astype(float).tolist(),
                "raw_guests": g_week.predicted.astype(float).tolist(),
            }
        )
    weeks.sort(key=lambda row: row["cutoff"])
    return weeks


def robust_guest_weeks(demand: pd.DataFrame, period: str) -> list[dict]:
    """Regenerate Config C from the verified series using the prior robust_guests recipe."""
    data = {
        row.demand_date.date(): (float(row.bookings), float(row.guests))
        for row in demand.itertuples(index=False)
    }
    start = 0 if period == "development" else DEV_WEEKS
    count = DEV_WEEKS if period == "development" else RES_WEEKS
    weeks = []
    for i in range(count):
        cutoff = FIRST + timedelta(weeks=start + i)
        history = {d: v for d, v in data.items() if d < cutoff}
        raw_b, raw_g, actual_b, actual_g, dates = [], [], [], [], []
        for offset in range(7):
            day = cutoff + timedelta(days=offset)
            dates.append(day.isoformat())
            actual_b.append(data[day][0])
            actual_g.append(data[day][1])
            # Bookings: equal-weight 4-week weekday mean (current_4week).
            booking_vals = [history[day - timedelta(weeks=lag)][0] for lag in range(1, 5)]
            raw_b.append(sum(booking_vals) / 4)
            # Guests: median of four matching weekdays including zeros.
            guest_vals = [history[day - timedelta(weeks=lag)][1] for lag in range(1, 5)]
            raw_g.append(float(median(guest_vals)))
        weeks.append(
            {
                "cutoff": cutoff.isoformat(),
                "dates": dates,
                "actual_bookings": actual_b,
                "actual_guests": actual_g,
                "raw_bookings": raw_b,
                "raw_guests": raw_g,
            }
        )
    return weeks


def evaluate_pair(weeks: list[dict], variant: str, pair_name: str, period: str) -> dict:
    booking_parts, guest_parts = [], []
    floor_changed_days = 0
    floor_guest_increase = 0
    daily_rows = []
    for week in weeks:
        raw_b, raw_g = week["raw_bookings"], week["raw_guests"]
        if variant == "A":
            pred_b, pred_g = apply_variant_a(raw_b, raw_g, PRODUCTION_PARTY_SIZE)
        elif variant == "B":
            pred_b, pred_g = apply_variant_b(raw_b, raw_g)
        else:
            raise ValueError(variant)
        impact = guest_floor_changes(raw_b, raw_g, PRODUCTION_PARTY_SIZE)
        if variant == "A":
            floor_changed_days += impact["changed_days"]
            floor_guest_increase += impact["guest_total_increase"]
        booking_parts.append((week["actual_bookings"], pred_b))
        guest_parts.append((week["actual_guests"], pred_g))
        for offset, day in enumerate(week["dates"]):
            daily_rows.append(
                {
                    "pair": pair_name,
                    "variant": variant,
                    "period": period,
                    "cutoff": week["cutoff"],
                    "date": day,
                    "horizon_offset": offset,
                    "actual_bookings": week["actual_bookings"][offset],
                    "actual_guests": week["actual_guests"][offset],
                    "raw_bookings": raw_b[offset],
                    "raw_guests": raw_g[offset],
                    "pred_bookings": pred_b[offset],
                    "pred_guests": pred_g[offset],
                }
            )
    return {
        "pair": pair_name,
        "variant": variant,
        "period": period,
        "bookings": summarize(booking_parts),
        "guests": summarize(guest_parts),
        "guest_floor_changed_days": floor_changed_days if variant == "A" else None,
        "guest_floor_guest_total_increase": floor_guest_increase if variant == "A" else None,
        "daily_rows": daily_rows,
    }


def flatten_summary(result: dict) -> dict:
    row = {
        "pair": result["pair"],
        "variant": result["variant"],
        "period": result["period"],
        "guest_floor_changed_days": result["guest_floor_changed_days"],
        "guest_floor_guest_total_increase": result["guest_floor_guest_total_increase"],
    }
    for target in ("bookings", "guests"):
        summary = result[target]
        for scale in ("weekly", "daily"):
            metrics = summary[scale]
            for key in ("wape", "mae", "rmse", "bias"):
                row[f"{target}_{scale}_{key}"] = metrics[key]
        row[f"{target}_zero_day_false_positives"] = summary["zero_day_false_positives"]
        row[f"{target}_zero_day_false_positive_total"] = summary["zero_day_false_positive_total"]
        row[f"{target}_high_demand_underpredictions"] = summary["high_demand_underpredictions"]
        row[f"{target}_high_demand_underprediction_total"] = summary["high_demand_underprediction_total"]
    return row


def selection_key(row: dict) -> tuple:
    # Complete-pair selection: average of bookings+guests weekly WAPE, then MAE.
    bw = row["bookings_weekly_wape"] if row["bookings_weekly_wape"] is not None else float("inf")
    gw = row["guests_weekly_wape"] if row["guests_weekly_wape"] is not None else float("inf")
    return (
        (bw + gw) / 2,
        row["bookings_weekly_mae"] + row["guests_weekly_mae"],
        row["bookings_weekly_rmse"] + row["guests_weekly_rmse"],
    )


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing results: {OUTPUT}")
    before = serving_hashes()
    demand = load_demand()
    preds = load_predictions()
    verify_against_demand(preds, demand)

    pred_hashes = {
        "phase3_daily_predictions_sha256": sha(PHASE3_DAILY),
        "phase3_arima_daily_predictions_sha256": sha(PHASE3_ARIMA),
        "phase2_daily_demand_sha256": EXPECTED_DAILY_SHA,
    }

    pairs = {
        "baseline_baseline": ("weekday_baseline", "weekday_baseline"),
        "arima211_baseline": ("arima_211", "weekday_baseline"),
        "rf_baseline": ("random_forest", "weekday_baseline"),
    }

    results = []
    daily_rows = []
    for period in ("development", "reserved"):
        week_cache = {
            name: pivot_week(preds, b_method, g_method, period)
            for name, (b_method, g_method) in pairs.items()
        }
        week_cache["robust_median_guests"] = robust_guest_weeks(demand, period)
        for pair_name, weeks in week_cache.items():
            for variant in ("A", "B"):
                evaluated = evaluate_pair(weeks, variant, pair_name, period)
                daily_rows.extend(evaluated["daily_rows"])
                results.append(flatten_summary(evaluated))

    comparison = pd.DataFrame(results)
    development = comparison[comparison.period == "development"]
    best_idx = min(development.index, key=lambda i: selection_key(development.loc[i].to_dict()))
    selected = development.loc[best_idx].to_dict()
    reserved_selected = comparison[
        (comparison.period == "reserved")
        & (comparison.pair == selected["pair"])
        & (comparison.variant == selected["variant"])
    ].iloc[0].to_dict()

    # Production reference = baseline_baseline + variant A
    production_ref = development[
        (development.pair == "baseline_baseline") & (development.variant == "A")
    ].iloc[0].to_dict()

    after = serving_hashes()
    if before != after:
        raise SystemExit("Serving artifacts changed during Phase 5")

    payload = {
        "protocol": {
            "phase": 5,
            "variants": {
                "A": "production: round/clamp, booking min when guests>0, party-size floor 7.0, guests>=bookings",
                "B": "no guest floor: round/clamp, booking min when guests>0, guests>=bookings",
                "C": "forecast pair robust_median_guests regenerated from verified series (prior recipe)",
            },
            "pairs": {
                "baseline_baseline": "weekday baseline bookings + guests",
                "arima211_baseline": "ARIMA(2,1,1) bookings + baseline guests",
                "rf_baseline": "random forest bookings + baseline guests",
                "robust_median_guests": "4-week bookings + median-of-four weekday guests",
            },
            "selection_rule": (
                "Lowest average of bookings and guests development weekly WAPE; "
                "ties broken by summed weekly MAE then RMSE"
            ),
            "high_demand_day_threshold_guests": HIGH_GUEST_DAY,
            "party_size": PRODUCTION_PARTY_SIZE,
            "prediction_hashes": pred_hashes,
            "serving_hashes_unchanged": True,
            "limitations": [
                "No Phase 4 standalone directory; used Phase 3 saved raw predictions",
                "Ridge reserved predictions were not invented and were not used",
                "Config C is regenerated deterministically from the verified series",
            ],
        },
        "selection": {
            "development_choice": {
                "pair": selected["pair"],
                "variant": selected["variant"],
                "bookings_weekly_wape": selected["bookings_weekly_wape"],
                "guests_weekly_wape": selected["guests_weekly_wape"],
                "average_weekly_wape": (selected["bookings_weekly_wape"] + selected["guests_weekly_wape"]) / 2,
            },
            "reserved_of_selected": {
                "bookings_weekly_wape": reserved_selected["bookings_weekly_wape"],
                "guests_weekly_wape": reserved_selected["guests_weekly_wape"],
            },
            "production_reference_development": {
                "pair": "baseline_baseline",
                "variant": "A",
                "bookings_weekly_wape": production_ref["bookings_weekly_wape"],
                "guests_weekly_wape": production_ref["guests_weekly_wape"],
            },
        },
        "comparison": results,
        "serving_hashes": before,
    }

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    comparison.to_csv(OUTPUT / "comparison.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(OUTPUT / "daily_operational_predictions.csv", index=False)
    (OUTPUT / "protocol.json").write_text(json.dumps(payload["protocol"], indent=2), encoding="utf-8")
    (OUTPUT / "selection.json").write_text(json.dumps(payload["selection"], indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "selection": payload["selection"],
                "rows": len(comparison),
                "serving_hashes_unchanged": True,
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
