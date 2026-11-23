"""Isolated, standard-library-only baseline experiment. No fitting or database access."""
import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import re

METHODS = {
    "current_4week": ((4, 3, 2, 1), (0.25, 0.25, 0.25, 0.25)),
    "short_2week": ((2, 1), (0.5, 0.5)),
    "weighted_4week": ((4, 3, 2, 1), (0.1, 0.2, 0.3, 0.4)),
}


def read_export(path):
    text = path.read_text(encoding="utf-8-sig")
    rows = re.findall(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", text, re.M)
    data = {}
    for ds, bookings, guests in rows:
        day = date.fromisoformat(ds)
        if day in data:
            raise ValueError("Duplicate day in input")
        b, g = int(bookings), int(guests)
        if g < b:
            raise ValueError("Guests below booking counts")
        data[day] = (b, g)
    expected_dates = {date(2025, 12, 1) + timedelta(days=i) for i in range(356)}
    if set(data) != expected_dates or sum(v[0] for v in data.values()) != 519 or sum(v[1] for v in data.values()) != 6669:
        raise ValueError("Export does not match verified 356-day snapshot")
    return data


def forecast(data, cutoff, target, method):
    history = {day: values[target] for day, values in data.items() if day < cutoff}
    lags, weights = METHODS[method]
    predicted = []
    for offset in range(7):
        forecast_day = cutoff + timedelta(days=offset)
        # No test actuals, recursive forecasts, imputation or filtering of zeros.
        values = [history[forecast_day - timedelta(weeks=lag)] for lag in lags]
        predicted.append(sum(value * weight for value, weight in zip(values, weights)))
    return predicted


def scores(actual, predicted):
    errors = [p - a for a, p in zip(actual, predicted)]
    absolute = sum(abs(e) for e in errors)
    total = sum(actual)
    return {"actual": total, "predicted": sum(predicted),
            "absolute_error": absolute, "net_error": sum(errors),
            "mae": absolute / len(errors),
            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "wape": 100 * absolute / total if total else None,
            "over": sum(e > 1e-9 for e in errors),
            "under": sum(e < -1e-9 for e in errors),
            "equal": sum(abs(e) <= 1e-9 for e in errors)}


def evaluate(data):
    cutoffs = [date(2026, 4, 6) + timedelta(weeks=i) for i in range(32)]
    weekly, summary = [], []
    daily = {}
    for target, target_name in enumerate(("bookings", "guests")):
        for method in METHODS:
            parts = []
            for cutoff in cutoffs:
                actual = [data[cutoff + timedelta(days=i)][target] for i in range(7)]
                prediction = forecast(data, cutoff, target, method)
                parts.append((actual, prediction))
                weekly.append({"target": target_name, "method": method,
                               "start": cutoff.isoformat(), "end": (cutoff + timedelta(days=6)).isoformat(),
                               "actual": sum(actual), "predicted": sum(prediction),
                               "error": sum(prediction) - sum(actual),
                               "absolute_error": abs(sum(prediction) - sum(actual))})
            daily[target_name, method] = parts
            for period, subset in (("overall_32weeks", parts), ("latest_8weeks", parts[-8:])):
                summary.append({"target": target_name, "method": method, "period": period,
                    "weekly": scores([sum(a) for a, p in subset], [sum(p) for a, p in subset]),
                    "daily": scores([v for a, p in subset for v in a], [v for a, p in subset for v in p])})
    comparisons = []
    for target in ("bookings", "guests"):
        baseline = [r for r in weekly if r["target"] == target and r["method"] == "current_4week"]
        for method in ("short_2week", "weighted_4week"):
            alternative = [r for r in weekly if r["target"] == target and r["method"] == method]
            for period, start in (("overall_32weeks", 0), ("latest_8weeks", 24)):
                differences = [a["absolute_error"] - b["absolute_error"]
                               for a, b in zip(alternative[start:], baseline[start:])]
                comparisons.append({"target": target, "method": method, "period": period,
                    "better_weeks": sum(v < -1e-9 for v in differences),
                    "worse_weeks": sum(v > 1e-9 for v in differences),
                    "tied_weeks": sum(abs(v) <= 1e-9 for v in differences),
                    "absolute_error_change": sum(differences)})
    expected = {"bookings": (31.29032258064516, 63.38709677419355, 54.03225806451613),
                "guests": (37.92449969678593, 87.07550030321407, 83.54545454545455)}
    for target, (weekly_wape, daily_wape, recent_wape) in expected.items():
        full = next(r for r in summary if r["target"] == target and r["method"] == "current_4week" and r["period"] == "overall_32weeks")
        recent = next(r for r in summary if r["target"] == target and r["method"] == "current_4week" and r["period"] == "latest_8weeks")
        for actual, reference in ((full["weekly"]["wape"], weekly_wape), (full["daily"]["wape"], daily_wape), (recent["weekly"]["wape"], recent_wape)):
            if not math.isclose(actual, reference, abs_tol=1e-9):
                raise ValueError("Existing baseline metrics did not reproduce")
    return {"summary": summary, "comparisons": comparisons, "weekly": weekly}


def serving_hashes(root):
    paths = list(root.glob("*.joblib")) + list(root.glob("*metadata.json")) + list(root.glob("*.py"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="New experiment output directory; never overwritten")
    args = parser.parse_args()
    experiment_root = Path(__file__).resolve().parent
    if args.output.resolve().parent != experiment_root:
        raise ValueError("Outputs must stay in this isolated experiment directory")
    if args.output.exists():
        raise ValueError("Output already exists")
    root = experiment_root.parent.parent
    before = serving_hashes(root)
    result = evaluate(read_export(args.input))
    after = serving_hashes(root)
    if before != after:
        raise ValueError("Serving files changed during experiment")
    result["protocol"] = {
        "cutoffs": "32 complete Monday-Sunday weeks, 2026-04-06 through 2026-11-15",
        "latest_8weeks": "2026-09-21 through 2026-11-15",
        "methods": METHODS, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "snapshot": {"days": 356, "bookings": 519, "guests": 6669},
        "output_scale": "Raw unrounded expected counts; no dashboard guest floors",
        "limitations": "Retrospective development experiment; not untouched prospective validation",
        "serving_hashes_unchanged": True,
        "serving_hashes": before,
    }
    args.output.mkdir()
    with (args.output / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with (args.output / "weekly_errors.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["weekly"][0]))
        writer.writeheader()
        writer.writerows(result["weekly"])
    print(json.dumps({key: value for key, value in result.items() if key not in ("weekly", "protocol")}, indent=2))
    print("Serving files unchanged; output:", args.output)


if __name__ == "__main__":
    main()
