"""Offline fixed-origin evaluation. Prints JSON; never saves serving artifacts.

Accepts a CSV or a pasted MariaDB table containing demand_date, bookings, guests.
Example: python -B evaluate_forecasts.py export.csv --window 2026-05-01:2026-05-28
All windows are development comparisons, not untouched final validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import HistGradientBoostingRegressor


def load_history(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    rows = re.findall(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", text, re.M)
    data = pd.DataFrame(rows, columns=["demand_date", "bookings", "guests"]) if rows else pd.read_csv(path)
    data = data[["demand_date", "bookings", "guests"]].copy()
    data["demand_date"] = pd.to_datetime(data["demand_date"], errors="raise")
    for target in ("bookings", "guests"):
        data[target] = pd.to_numeric(data[target], errors="raise")
        values = data[target].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{target} must contain finite nonnegative integer counts")
    data = data.sort_values("demand_date").reset_index(drop=True)
    dates = data.demand_date
    if data.empty or dates.duplicated().any() or not dates.equals(pd.Series(pd.date_range(dates.min(), dates.max()))):
        raise ValueError("Require unique, continuous daily observations; do not fill unverified gaps")
    if (data.guests < data.bookings).any():
        raise ValueError("Guests cannot be below bookings")
    return data


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    error = predicted - actual
    total = float(actual.sum())
    return {
        "n": len(actual), "actual_total": total,
        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(error.mean()),
        "wape": float(100 * np.abs(error).sum() / total) if total else None,
    }


def weekday_forecast(history, dates):
    # Use only actual values known at the cutoff, matching the existing baseline.
    result = []
    for date in dates:
        prior = [float(history.loc[date - pd.Timedelta(days=7 * week)])
                 for week in range(1, 5) if date - pd.Timedelta(days=7 * week) in history.index]
        if not prior:
            prior = history[history.index.dayofweek == date.dayofweek].tail(8).tolist()
        result.append(float(np.mean(prior)) if prior else float(history.mean()))
    return np.asarray(result)


def features(date, past):
    # Training receives only rows strictly preceding date. At inference, past
    # is extended with predictions, never with future actual observations.
    return [np.sin(2 * np.pi * date.dayofweek / 7), np.cos(2 * np.pi * date.dayofweek / 7),
            *[past[-lag] for lag in (1, 7, 14, 28)],
            float(np.mean(past[-7:])), float(np.mean(past[-28:]))]


def poisson_forecast(history, dates):
    values = history.to_numpy(dtype=float).tolist()
    x = [features(history.index[i], values[:i]) for i in range(28, len(values))]
    y = values[28:]
    if not y or not np.any(y):
        return np.zeros(len(dates))
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=100, max_leaf_nodes=7, min_samples_leaf=15,
        l2_regularization=1.0, learning_rate=0.05, early_stopping=False, random_state=42,
    )
    model.fit(x, y)
    result = []
    for date in dates:
        value = max(0.0, float(model.predict([features(date, values)])[0]))
        result.append(value)
        values.append(value)
    return np.asarray(result)


def prophet_forecast(history, dates, yearly):
    model = Prophet(yearly_seasonality=yearly, weekly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.1,
                    seasonality_mode="additive", interval_width=0.8,
                    uncertainty_samples=0)
    model.fit(pd.DataFrame({"ds": history.index, "y": history.values}), seed=42)
    return np.maximum(0, model.predict(pd.DataFrame({"ds": dates})).yhat.to_numpy())


def evaluate(data, windows):
    records, pooled = [], {}
    for start, end in windows:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start != start.normalize() or end != end.normalize() or end < start:
            raise ValueError("Require an ascending daily window")
        train = data[data.demand_date < start]
        test = data[data.demand_date.between(start, end)]
        if len(train) < 120 or len(test) != (end - start).days + 1:
            raise ValueError("Require >=120 training days and complete test coverage")
        dates = pd.DatetimeIndex(test.demand_date)
        for target in ("bookings", "guests"):
            history = train.set_index("demand_date")[target]
            predictions = {
                "weekday_baseline": weekday_forecast(history, dates),
                "prophet_current": prophet_forecast(history, dates, True),
                "prophet_weekly": prophet_forecast(history, dates, False),
                "poisson_boosting": poisson_forecast(history, dates),
                "zero_reference": np.zeros(len(test)),
            }
            for method, prediction in predictions.items():
                actual = test[target].to_numpy(dtype=float)
                for output, values in (("raw", prediction), ("rounded", np.rint(prediction))):
                    records.append({"start": str(start.date()), "end": str(end.date()),
                                    "train_days": len(train), "target": target, "method": method,
                                    "output": output, **metrics(actual, values)})
                    pooled.setdefault((target, method, output), []).append((actual, values))
    summary = []
    for (target, method, output), parts in pooled.items():
        summary.append({"target": target, "method": method, "output": output,
                        **metrics(np.concatenate([a for a, _ in parts]),
                                  np.concatenate([p for _, p in parts]))})
    return {"protocol": "fixed_origin; expanding history; no holdout actuals; development evaluation",
            "limitations": ["Rounded results exclude dashboard cross-target guest floors and adjustments",
                            "Overlapping test windows, if supplied, repeat observations in pooled metrics",
                            "No serving artifacts are selected or saved"],
            "windows": records, "pooled": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--window", action="append", required=True, help="YYYY-MM-DD:YYYY-MM-DD")
    args = parser.parse_args()
    data = load_history(args.input)
    windows = []
    for value in args.window:
        parts = value.split(":")
        if len(parts) != 2:
            parser.error("Each window requires start:end")
        windows.append(parts)
    report = evaluate(data, windows)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    report["dataset"] = {"days": len(data), "bookings": int(data.bookings.sum()),
                         "guests": int(data.guests.sum()),
                         "start": str(data.demand_date.min().date()), "end": str(data.demand_date.max().date())}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
