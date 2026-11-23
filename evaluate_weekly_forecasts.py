"""Seven-day, fixed-origin development backtest; stdout only, no serving writes.

Uses complete Monday-Sunday windows after >=120 days of history. All methods
receive the same observations before each cutoff. Existing windows have already
been inspected, so none of these results constitutes untouched final validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_forecasts import load_history, metrics, weekday_forecast, prophet_forecast, poisson_forecast


def weekly_windows(data, minimum_days=120):
    start = data.demand_date.min() + pd.Timedelta(days=minimum_days)
    start += pd.Timedelta(days=(-start.dayofweek) % 7)
    while start + pd.Timedelta(days=6) <= data.demand_date.max():
        yield start, start + pd.Timedelta(days=6)
        start += pd.Timedelta(days=7)


def summarize(parts):
    actual = np.concatenate([a for a, _ in parts])
    prediction = np.concatenate([p for _, p in parts])
    weekly_actual = np.array([a.sum() for a, _ in parts])
    weekly_prediction = np.array([p.sum() for _, p in parts])
    return {
        "weeks": len(parts), "zero_demand_weeks": int((weekly_actual == 0).sum()),
        "daily": metrics(actual, prediction),
        "weekly_totals": metrics(weekly_actual, weekly_prediction),
    }


def evaluate(data):
    windows, pooled = [], {}
    for start, end in weekly_windows(data):
        train = data[data.demand_date < start]
        test = data[data.demand_date.between(start, end)]
        if len(test) != 7:
            raise ValueError("Incomplete week; do not silently fill missing dates")
        dates = pd.DatetimeIndex(test.demand_date)
        for target in ("bookings", "guests"):
            history = train.set_index("demand_date")[target]
            predictions = {
                "weekday_baseline": weekday_forecast(history, dates),
                "prophet_weekly": prophet_forecast(history, dates, yearly=False),
                "poisson_boosting": poisson_forecast(history, dates),
                "zero_reference": np.zeros(7),
            }
            actual = test[target].to_numpy(dtype=float)
            for method, predicted in predictions.items():
                # Raw weekly sums and sums of daily rounded points are distinct.
                for output, values in (("raw", predicted), ("daily_rounded", np.rint(predicted))):
                    windows.append({"start": str(start.date()), "end": str(end.date()),
                                    "target": target, "method": method, "output": output,
                                    "actual_total": float(actual.sum()),
                                    "predicted_total": float(values.sum()),
                                    "daily": metrics(actual, values)})
                    pooled.setdefault((target, method, output), []).append((actual, values))
        print(f"Completed {start.date()} to {end.date()}", file=__import__('sys').stderr, flush=True)
    if not pooled:
        raise ValueError("No complete weeks after the minimum training history")
    results = []
    for (target, method, output), parts in pooled.items():
        results.append({"target": target, "method": method, "output": output,
                        **summarize(parts),
                        "latest_8_weeks": summarize(parts[-8:])})
    return {"protocol": "7-day fixed-origin; Monday-Sunday; expanding training; raw and daily-rounded",
            "limitations": ["Retrospective development evaluation; not untouched final validation",
                            "Weekly-total error can cancel opposing daily errors",
                            "No dashboard guest-floor or cross-target reconciliation applied",
                            "Latest-eight-week results overlap the full-period results",
                            "No hyperparameter search; no serving artifacts replaced"],
            "summary": results, "windows": windows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    data = load_history(args.input)
    report = evaluate(data)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2, allow_nan=False))
