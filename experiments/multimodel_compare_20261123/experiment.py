"""Phase 3 multimodel comparison. Raw forecasts only. No production changes."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from metrics import summarize_weeks
from models import (
    ARIMA_ORDERS,
    RF_CONFIG,
    RIDGE_ALPHAS,
    arima_forecast,
    baseline_pair,
    random_forest_forecast,
    ridge_forecast,
    statsmodels_available,
)

HERE = Path(__file__).resolve().parent
PHASE2_PREPARED = HERE.parent / "multimodel_dataset_20261123" / "prepared"
EXPECTED_DAILY_SHA = "7017de3fb24a1105e3a09872526367aa40ce1b05384e28065de99a1464ca1cf7"
ROOT = HERE.parents[1]
OUTPUT = HERE / "results"

DEVELOPMENT = 24
RESERVED = 8
FIRST = date(2026, 4, 6)


def serving_hashes() -> dict[str, str]:
    names = [
        "bookings_model.joblib",
        "guests_model.joblib",
        "bookings_model_metadata.json",
        "forecast_models_metadata.json",
        "api.py",
        "predict_real.py",
        "train_real_model.py",
        "evaluate_forecasts.py",
        "evaluate_weekly_forecasts.py",
    ]
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in names
        if (ROOT / name).exists()
    }


def load_frame() -> pd.DataFrame:
    path = PHASE2_PREPARED / "data" / "daily_demand.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_DAILY_SHA:
        raise ValueError(f"Phase 2 daily_demand.csv SHA mismatch: {digest}")
    frame = pd.read_csv(path)
    frame["demand_date"] = pd.to_datetime(frame["demand_date"]).dt.normalize()
    if len(frame) != 356 or int(frame.bookings.sum()) != 519 or int(frame.guests.sum()) != 6669:
        raise ValueError("Phase 2 totals failed verification")
    return frame.sort_values("demand_date").reset_index(drop=True)


def cutoffs(count: int, start_index: int = 0) -> list[date]:
    return [FIRST + timedelta(weeks=start_index + i) for i in range(count)]


def actuals(frame: pd.DataFrame, cutoff: date, target: str) -> list[float]:
    start = pd.Timestamp(cutoff)
    end = start + pd.Timedelta(days=6)
    window = frame[(frame.demand_date >= start) & (frame.demand_date <= end)]
    if len(window) != 7:
        raise ValueError(f"Incomplete week at {cutoff}")
    return window[target].astype(float).tolist()


def evaluate_method(
    frame: pd.DataFrame,
    method_name: str,
    cutoff_list: list[date],
    predict_fn,
) -> dict:
    weekly_rows = []
    parts = {"bookings": [], "guests": []}
    daily_rows = []
    for cutoff in cutoff_list:
        predicted = predict_fn(frame, cutoff)
        for target in ("bookings", "guests"):
            actual = actuals(frame, cutoff, target)
            pred = predicted[target]
            parts[target].append((actual, pred))
            weekly_rows.append(
                {
                    "method": method_name,
                    "target": target,
                    "cutoff": cutoff.isoformat(),
                    "end": (cutoff + timedelta(days=6)).isoformat(),
                    "actual": sum(actual),
                    "predicted": sum(pred),
                    "error": sum(pred) - sum(actual),
                    "absolute_error": abs(sum(pred) - sum(actual)),
                }
            )
            for offset, (a, p) in enumerate(zip(actual, pred)):
                daily_rows.append(
                    {
                        "method": method_name,
                        "target": target,
                        "cutoff": cutoff.isoformat(),
                        "date": (cutoff + timedelta(days=offset)).isoformat(),
                        "horizon_offset": offset,
                        "actual": a,
                        "predicted": p,
                        "error": p - a,
                    }
                )
    summary = {
        target: summarize_weeks(parts[target]) for target in ("bookings", "guests")
    }
    return {"summary": summary, "weekly": weekly_rows, "daily": daily_rows}


def _rank_key(summary: dict) -> tuple:
    weekly = summary["weekly"]
    return (
        weekly["wape"] if weekly["wape"] is not None else float("inf"),
        weekly["mae"],
        weekly["rmse"],
    )


def select_on_development(dev_summaries: dict) -> dict:
    """Pick one configuration per target using development weekly WAPE, then MAE."""
    selected = {}
    for target in ("bookings", "guests"):
        best_name = None
        best_summary = None
        for name, payload in dev_summaries.items():
            candidate = payload["summary"][target]
            if best_summary is None or _rank_key(candidate) < _rank_key(best_summary):
                best_name, best_summary = name, candidate
        selected[target] = {
            "method": best_name,
            "development_weekly_wape": best_summary["weekly"]["wape"],
            "development_daily_wape": best_summary["daily"]["wape"],
            "development_weekly_mae": best_summary["weekly"]["mae"],
        }
    return selected


def build_selected_ridge_predictor(selection: dict):
    """Combine per-target Ridge alphas when both targets select Ridge configs."""
    alphas = {}
    for target in ("bookings", "guests"):
        method = selection[target]["method"]
        if method.startswith("ridge_alpha_"):
            alphas[target] = float(method.removeprefix("ridge_alpha_"))
    if len(alphas) != 2:
        return None

    def predict(frame, cutoff, alphas=alphas):
        return ridge_forecast(
            frame,
            cutoff,
            alpha_bookings=alphas["bookings"],
            alpha_guests=alphas["guests"],
        )

    return predict, alphas


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing results: {OUTPUT}")
    before = serving_hashes()
    frame = load_frame()
    OUTPUT.mkdir(parents=True)

    methods: dict[str, callable] = {
        "weekday_baseline": lambda f, c: baseline_pair(f, c),
    }
    for alpha in RIDGE_ALPHAS:
        name = f"ridge_alpha_{alpha:g}"
        methods[name] = (
            lambda f, c, a=alpha: ridge_forecast(
                f, c, alpha_bookings=a, alpha_guests=a
            )
        )
    methods["random_forest"] = lambda f, c: random_forest_forecast(f, c)

    arima_status = {
        "available": statsmodels_available(),
        "dependency": "statsmodels",
        "orders_planned": [list(order) for order in ARIMA_ORDERS],
        "ran": False,
        "reason": None,
    }
    if statsmodels_available():
        for order in ARIMA_ORDERS:
            label = f"arima_{order[0]}{order[1]}{order[2]}"

            def make_predict(order=order):
                def predict(f, c):
                    return {
                        "bookings": arima_forecast(f, c, "bookings", order),
                        "guests": arima_forecast(f, c, "guests", order),
                    }

                return predict

            methods[label] = make_predict()
        arima_status["ran"] = True
    else:
        arima_status["reason"] = (
            "statsmodels is not installed in this environment. "
            "No ARIMA fits were run. Approve installation before Phase 3 ARIMA results."
        )

    development_cutoffs = cutoffs(DEVELOPMENT)
    reserved_cutoffs = cutoffs(RESERVED, start_index=DEVELOPMENT)

    development = {}
    all_weekly = []
    all_daily = []
    for name, predict_fn in methods.items():
        result = evaluate_method(frame, name, development_cutoffs, predict_fn)
        development[name] = result
        all_weekly.extend(result["weekly"])
        all_daily.extend(result["daily"])

    selection = select_on_development(development)

    # Reserved evaluation for baseline, selected configs, and RF (always report RF).
    reserved_methods = {"weekday_baseline": methods["weekday_baseline"]}
    for target, choice in selection.items():
        reserved_methods[choice["method"]] = methods[choice["method"]]
    if "random_forest" not in reserved_methods:
        reserved_methods["random_forest"] = methods["random_forest"]
    selected_ridge = build_selected_ridge_predictor(selection)
    if selected_ridge is not None:
        predict_ridge, ridge_alphas = selected_ridge
        reserved_methods["ridge_selected_pair"] = predict_ridge
        selection["ridge_selected_pair_alphas"] = ridge_alphas

    reserved = {}
    for name, predict_fn in reserved_methods.items():
        result = evaluate_method(frame, name, reserved_cutoffs, predict_fn)
        reserved[name] = result
        # Tag period on appended rows
        for row in result["weekly"]:
            row["period"] = "reserved"
            all_weekly.append(row)
        for row in result["daily"]:
            row["period"] = "reserved"
            all_daily.append(row)
    for row in all_weekly:
        row.setdefault("period", "development")
    for row in all_daily:
        row.setdefault("period", "development")

    comparison = []
    for name, payload in development.items():
        for target in ("bookings", "guests"):
            summary = payload["summary"][target]
            comparison.append(
                {
                    "period": "development",
                    "method": name,
                    "target": target,
                    "weekly_wape": summary["weekly"]["wape"],
                    "daily_wape": summary["daily"]["wape"],
                    "weekly_mae": summary["weekly"]["mae"],
                    "daily_mae": summary["daily"]["mae"],
                    "weekly_rmse": summary["weekly"]["rmse"],
                    "daily_rmse": summary["daily"]["rmse"],
                    "weekly_bias": summary["weekly"]["bias"],
                    "daily_bias": summary["daily"]["bias"],
                }
            )
    for name, payload in reserved.items():
        for target in ("bookings", "guests"):
            summary = payload["summary"][target]
            comparison.append(
                {
                    "period": "reserved",
                    "method": name,
                    "target": target,
                    "weekly_wape": summary["weekly"]["wape"],
                    "daily_wape": summary["daily"]["wape"],
                    "weekly_mae": summary["weekly"]["mae"],
                    "daily_mae": summary["daily"]["mae"],
                    "weekly_rmse": summary["weekly"]["rmse"],
                    "daily_rmse": summary["daily"]["rmse"],
                    "weekly_bias": summary["weekly"]["bias"],
                    "daily_bias": summary["daily"]["bias"],
                }
            )

    after = serving_hashes()
    if before != after:
        raise SystemExit("Serving artifacts changed during the experiment")

    protocol = {
        "phase": 3,
        "forecast_type": "raw",
        "dataset_sha256": EXPECTED_DAILY_SHA,
        "dataset_totals": {"days": 356, "bookings": 519, "guests": 6669},
        "development": "24 weeks: 2026-04-06 through 2026-09-20",
        "reserved": "8 weeks: 2026-09-21 through 2026-11-15; previously inspected",
        "wape_definition": {
            "formula": "100 * sum(|predicted - actual|) / sum(actual)",
            "weekly": "seven-day totals aggregated first, then WAPE across weeks",
            "daily": "WAPE across all daily points in the period",
            "note": "WAPE is an error measure, not an accuracy percentage",
        },
        "ridge_alphas": list(RIDGE_ALPHAS),
        "random_forest": RF_CONFIG,
        "arima": arima_status,
        "selection_rule": (
            "Per target, lowest development weekly WAPE; ties broken by weekly MAE then RMSE"
        ),
        "serving_hashes_unchanged": True,
    }

    payload = {
        "protocol": protocol,
        "selection": selection,
        "comparison": comparison,
        "development_summaries": {
            name: payload["summary"] for name, payload in development.items()
        },
        "reserved_summaries": {
            name: payload["summary"] for name, payload in reserved.items()
        },
        "serving_hashes": before,
    }
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(comparison).to_csv(OUTPUT / "comparison.csv", index=False)
    pd.DataFrame(all_weekly).to_csv(OUTPUT / "weekly_predictions.csv", index=False)
    pd.DataFrame(all_daily).to_csv(OUTPUT / "daily_predictions.csv", index=False)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(json.dumps({
        "selection": selection,
        "arima": arima_status,
        "comparison_rows": len(comparison),
        "serving_hashes_unchanged": True,
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
