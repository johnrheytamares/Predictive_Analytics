"""Isolated ARIMA evaluation using the Phase 3 experiment venv. Raw forecasts only."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from experiment import (
    EXPECTED_DAILY_SHA,
    cutoffs,
    evaluate_method,
    load_frame,
    serving_hashes,
)
from models import ARIMA_ORDERS, arima_forecast, statsmodels_available

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "arima"
DEVELOPMENT = 24
RESERVED = 8


def select_arima(dev_summaries: dict) -> dict:
    def key(summary):
        weekly = summary["weekly"]
        return (
            weekly["wape"] if weekly["wape"] is not None else float("inf"),
            weekly["mae"],
            weekly["rmse"],
        )

    selected = {}
    for target in ("bookings", "guests"):
        best_name = None
        best = None
        for name, payload in dev_summaries.items():
            candidate = payload["summary"][target]
            if best is None or key(candidate) < key(best):
                best_name, best = name, candidate
        selected[target] = {
            "method": best_name,
            "development_weekly_wape": best["weekly"]["wape"],
            "development_daily_wape": best["daily"]["wape"],
            "development_weekly_mae": best["weekly"]["mae"],
        }
    return selected


def safe_arima_pair(order):
    def predict(frame, cutoff, order=order):
        out = {}
        for target in ("bookings", "guests"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[target] = arima_forecast(frame, cutoff, target, order)
        return out

    return predict


def main() -> None:
    if not statsmodels_available():
        raise SystemExit("statsmodels is unavailable in this interpreter")
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing ARIMA results: {OUTPUT}")

    before = serving_hashes()
    frame = load_frame()
    OUTPUT.mkdir(parents=True)

    methods = {}
    for order in ARIMA_ORDERS:
        label = f"arima_{order[0]}{order[1]}{order[2]}"
        methods[label] = safe_arima_pair(order)

    development_cutoffs = cutoffs(DEVELOPMENT)
    reserved_cutoffs = cutoffs(RESERVED, start_index=DEVELOPMENT)

    development = {}
    weekly_rows = []
    daily_rows = []
    failures = []
    for name, predict_fn in methods.items():
        try:
            result = evaluate_method(frame, name, development_cutoffs, predict_fn)
            development[name] = result
            for row in result["weekly"]:
                row["period"] = "development"
                weekly_rows.append(row)
            for row in result["daily"]:
                row["period"] = "development"
                daily_rows.append(row)
        except Exception as exc:  # noqa: BLE001 - record and continue other orders
            failures.append({"method": name, "period": "development", "error": str(exc)})

    if not development:
        raise SystemExit(f"All ARIMA development evaluations failed: {failures}")

    selection = select_arima(development)

    reserved_methods = {
        selection["bookings"]["method"]: methods[selection["bookings"]["method"]],
        selection["guests"]["method"]: methods[selection["guests"]["method"]],
    }
    # Always also score both selected configs as full pairs for reserved reporting.
    reserved = {}
    for name, predict_fn in reserved_methods.items():
        try:
            result = evaluate_method(frame, name, reserved_cutoffs, predict_fn)
            reserved[name] = result
            for row in result["weekly"]:
                row["period"] = "reserved"
                weekly_rows.append(row)
            for row in result["daily"]:
                row["period"] = "reserved"
                daily_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failures.append({"method": name, "period": "reserved", "error": str(exc)})

    # Mixed selected pair if bookings/guests chose different orders.
    if selection["bookings"]["method"] != selection["guests"]["method"]:
        def parse_order(label: str) -> tuple[int, int, int]:
            digits = label.removeprefix("arima_")
            if len(digits) != 3:
                raise ValueError(label)
            return int(digits[0]), int(digits[1]), int(digits[2])

        b_order = parse_order(selection["bookings"]["method"])
        g_order = parse_order(selection["guests"]["method"])

        def mixed(frame, cutoff, b_order=b_order, g_order=g_order):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return {
                    "bookings": arima_forecast(frame, cutoff, "bookings", b_order),
                    "guests": arima_forecast(frame, cutoff, "guests", g_order),
                }

        mixed_result = evaluate_method(frame, "arima_selected_pair", reserved_cutoffs, mixed)
        reserved["arima_selected_pair"] = mixed_result
        for row in mixed_result["weekly"]:
            row["period"] = "reserved"
            weekly_rows.append(row)
        for row in mixed_result["daily"]:
            row["period"] = "reserved"
            daily_rows.append(row)

    comparison = []
    for name, payload in development.items():
        for target in ("bookings", "guests"):
            s = payload["summary"][target]
            comparison.append(
                {
                    "period": "development",
                    "method": name,
                    "target": target,
                    "weekly_wape": s["weekly"]["wape"],
                    "daily_wape": s["daily"]["wape"],
                    "weekly_mae": s["weekly"]["mae"],
                    "daily_mae": s["daily"]["mae"],
                    "weekly_rmse": s["weekly"]["rmse"],
                    "daily_rmse": s["daily"]["rmse"],
                    "weekly_bias": s["weekly"]["bias"],
                    "daily_bias": s["daily"]["bias"],
                }
            )
    for name, payload in reserved.items():
        for target in ("bookings", "guests"):
            s = payload["summary"][target]
            comparison.append(
                {
                    "period": "reserved",
                    "method": name,
                    "target": target,
                    "weekly_wape": s["weekly"]["wape"],
                    "daily_wape": s["daily"]["wape"],
                    "weekly_mae": s["weekly"]["mae"],
                    "daily_mae": s["daily"]["mae"],
                    "weekly_rmse": s["weekly"]["rmse"],
                    "daily_rmse": s["daily"]["rmse"],
                    "weekly_bias": s["weekly"]["bias"],
                    "daily_bias": s["daily"]["bias"],
                }
            )

    # Merge prior Phase 3 comparison for side-by-side reporting.
    prior = pd.read_csv(HERE / "results" / "comparison.csv")
    prior["family"] = prior["method"].map(
        lambda m: "baseline"
        if m == "weekday_baseline"
        else "ridge"
        if m.startswith("ridge_")
        else "random_forest"
        if m == "random_forest"
        else "other"
    )
    arima_df = pd.DataFrame(comparison)
    arima_df["family"] = "arima"
    combined = pd.concat([prior, arima_df], ignore_index=True)

    after = serving_hashes()
    if before != after:
        raise SystemExit("Serving artifacts changed during ARIMA evaluation")

    payload = {
        "protocol": {
            "phase": "3-arima",
            "interpreter": str(Path(__file__).resolve()),
            "dataset_sha256": EXPECTED_DAILY_SHA,
            "orders": [list(o) for o in ARIMA_ORDERS],
            "development": "24 weeks: 2026-04-06 through 2026-09-20",
            "reserved": "8 weeks: 2026-09-21 through 2026-11-15; previously inspected",
            "wape": "100 * sum(|pred-actual|) / sum(actual); weekly uses week totals",
            "serving_hashes_unchanged": True,
            "failures": failures,
        },
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
    arima_df.to_csv(OUTPUT / "comparison.csv", index=False)
    combined.to_csv(OUTPUT / "combined_comparison.csv", index=False)
    pd.DataFrame(weekly_rows).to_csv(OUTPUT / "weekly_predictions.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(OUTPUT / "daily_predictions.csv", index=False)

    print(
        json.dumps(
            {
                "selection": selection,
                "failures": failures,
                "development_methods": list(development),
                "serving_hashes_unchanged": True,
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
