"""Frozen small adaptive-baseline experiment; no model fitting or live access."""
import argparse
from datetime import date, timedelta
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location("fixed_baselines", HERE.parent / "offline_baseline_20261123" / "experiment.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

METHODS = ("current_4week", "short_2week", "weighted_4week", "adaptive", "robust_guests", "trend_adjusted")
PLAN = {
    "development": "24 weeks: 2026-04-06 through 2026-09-20",
    "holdout": "8 weeks: 2026-09-21 through 2026-11-15",
    "holdout_status": "Selection-excluded in this run; previously inspected, NOT untouched independent validation",
    "adaptive": "Choose two-week only when previous four completed weekly origins have >=10% lower absolute weekly error than four-week; otherwise four-week. Per target, all inner outcomes before outer cutoff.",
    "robust_guests": "Four-week baseline bookings; median of four matching-weekday guest counts, including zeros; no source rows deleted",
    "trend_adjusted": "Four-week forecast scaled by 1+0.5*(clip(mean last14 / mean preceding14,0.5,1.5)-1); factor range0.75..1.25. Both means zero =>1; older mean zero and recent positive =>1.25",
    "operational": "Python nonnegative round; positive guests imply >=1 booking; guest floor uses pre-cutoff median nonzero-day guests/bookings rounded to2 decimals; guests>=bookings",
    "selection": "Development operational WAPE, weekly RMSE then fixed method order for ties; one choice per target from complete same-method pairs. Mixed pair is fully re-scored; never selected from holdout.",
    "gates": "Each target: >=5% relative weekly WAPE and MAE gains; weekly RMSE and daily WAPE <=105% control; absolute weekly bias <=control absolute bias +10% control weekly MAE; low-demand MAE <=110% control. Zero-week error must not worsen where observed.",
    "low_demand": {"bookings": 5, "guests": 35},
    "output": "Isolated JSON only; no changes to serving files or data",
}


def predict(data, cutoff, target, method):
    history = {d: v for d, v in data.items() if d < cutoff}
    if method in base.METHODS:
        return base.forecast(history, cutoff, target, method), method
    if method == "adaptive":
        errors = {name: 0.0 for name in ("current_4week", "short_2week")}
        for lag in range(1, 5):
            origin = cutoff - timedelta(weeks=lag)
            actual = sum(history[origin + timedelta(days=i)][target] for i in range(7))
            for name in errors:
                expected = sum(base.forecast(history, origin, target, name))
                errors[name] += abs(expected - actual)
        chosen = "short_2week" if errors["short_2week"] < errors["current_4week"] * 0.9 else "current_4week"
        return base.forecast(history, cutoff, target, chosen), chosen
    current = base.forecast(history, cutoff, target, "current_4week")
    if method == "robust_guests":
        if target == 0:
            return current, "current_4week"
        return [median(history[cutoff + timedelta(days=i, weeks=-lag)][target]
                       for lag in range(1, 5)) for i in range(7)], "median_4week"
    if method == "trend_adjusted":
        recent = sum(history[cutoff-timedelta(days=i)][target] for i in range(1, 15)) / 14
        older = sum(history[cutoff-timedelta(days=i)][target] for i in range(15, 29)) / 14
        ratio = min(1.5, max(0.5, recent / older)) if older else (1.5 if recent else 1.0)
        factor = 1 + 0.5 * (ratio - 1)
        return [p * factor for p in current], factor
    raise ValueError(method)


def party_size(data, cutoff):
    values = [g / b for d, (b, g) in data.items() if d < cutoff and b > 0]
    return round(max(1.0, median(values)), 2) if values else 1.0


def operational(bookings, guests, party):
    result = [[], []]
    clamp = lambda value: int(max(0, round(float(value))))
    for b, g in zip(bookings, guests):
        b, g = clamp(b), clamp(g)
        if g > 0 and b < 1:
            b = 1
        g = max(g, clamp(b * party), b)
        result[0].append(b)
        result[1].append(g)
    return result


def summarize(parts, target):
    actual = [sum(a) for a, _ in parts]
    predicted = [sum(p) for _, p in parts]
    low = [(a, p) for a, p in parts if sum(a) <= PLAN["low_demand"][target]]
    zero = [(a, p) for a, p in parts if sum(a) == 0]
    return {
        "weeks": len(parts), "weekly": base.scores(actual, predicted),
        "daily": base.scores([v for a, _ in parts for v in a], [v for _, p in parts for v in p]),
        "low_demand_weeks": len(low),
        "low_demand": base.scores([sum(a) for a,p in low], [sum(p) for a,p in low]) if low else None,
        "zero_demand_weeks": len(zero),
        "zero_week_false_positive_total": sum(sum(p) for _,p in zero),
    }


def collect(data, cutoffs, pair):
    parts = {"raw": [[], []], "operational": [[], []]}
    details = []
    for cutoff in cutoffs:
        actual = [[data[cutoff+timedelta(days=i)][target] for i in range(7)] for target in (0, 1)]
        predicted = [predict(data, cutoff, target, pair[target]) for target in (0, 1)]
        raw = [p[0] for p in predicted]
        party = party_size(data, cutoff)
        op = operational(*raw, party)
        for output, values in (("raw", raw), ("operational", op)):
            for target in (0, 1):
                parts[output][target].append((actual[target], values[target]))
        details.append({"cutoff": cutoff.isoformat(), "actual": actual, "raw": raw,
                        "operational": op, "party_size": party,
                        "decisions": [p[1] for p in predicted]})
    summary = {output: {name: summarize(parts[output][i], name)
                        for i, name in enumerate(("bookings", "guests"))} for output in parts}
    return summary, details


def passes(candidate, control):
    c, b = candidate["weekly"], control["weekly"]
    low = candidate["low_demand"]
    reference_low = control["low_demand"]
    checks = {
        "weekly_wape": c["wape"] is not None and b["wape"] is not None and c["wape"] <= b["wape"] * .95,
        "weekly_mae": c["mae"] <= b["mae"] * .95,
        "weekly_rmse": c["rmse"] <= b["rmse"] * 1.05,
        "daily_wape": candidate["daily"]["wape"] <= control["daily"]["wape"] * 1.05,
        "bias": abs(c["net_error"] / candidate["weeks"]) <= abs(b["net_error"] / control["weeks"]) + .1 * b["mae"],
        "low_demand": low is not None and reference_low is not None and low["mae"] <= reference_low["mae"] * 1.1,
        "zero_weeks": candidate["zero_week_false_positive_total"] <= control["zero_week_false_positive_total"] + 1e-9,
    }
    return {"checks": checks, "numerical_gates_pass": all(checks.values()),
            "independent_validation_pass": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve().parent != HERE or args.output.exists():
        raise ValueError("Use a new isolated output directory")
    args.output.mkdir()
    # Freeze the protocol on disk BEFORE reading/scoring this run's data.
    with (args.output/"protocol.json").open("x") as handle:
        json.dump(PLAN, handle, indent=2)
    before = base.serving_hashes(ROOT)
    data = base.read_export(args.input)
    cutoffs = [date(2026,4,6)+timedelta(weeks=i) for i in range(32)]
    dev, hold = {}, {}
    details = {}
    for method in METHODS:
        dev[method], _ = collect(data, cutoffs[:24], (method, method))
    winners = []
    for target in ("bookings", "guests"):
        winners.append(min(METHODS, key=lambda name: (
            dev[name]["operational"][target]["weekly"]["wape"],
            dev[name]["operational"][target]["weekly"]["rmse"], METHODS.index(name))))
    mixed = tuple(winners)
    dev["locked_mixed"], _ = collect(data, cutoffs[:24], mixed)
    with (args.output/"selection_lock.json").open("x") as handle:
        json.dump({"bookings": mixed[0], "guests": mixed[1], "basis": "development operational WAPE only"}, handle, indent=2)
    for name in (*METHODS, "locked_mixed"):
        pair = mixed if name == "locked_mixed" else (name, name)
        hold[name], details[name] = collect(data, cutoffs[24:], pair)
    gates = {name: {target: passes(hold[name]["operational"][target],hold["current_4week"]["operational"][target])
                     for target in ("bookings", "guests")} for name in hold}
    if before != base.serving_hashes(ROOT):
        raise ValueError("Serving files changed")
    result = {"protocol": PLAN, "locked_pair": mixed, "development": dev,
              "holdout": hold, "holdout_weekly_details": details, "gates": gates,
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "serving_files_unchanged": True}
    with (args.output/"results.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print('LOCKED_PAIR', mixed)
    for period, groups in (("development", dev), ("holdout", hold)):
        for name, outputs in groups.items():
            print(period, name, json.dumps({output: {target: {
                "weekly_wape": m["weekly"]["wape"], "daily_wape": m["daily"]["wape"],
                "mae": m["weekly"]["mae"], "rmse": m["weekly"]["rmse"],
                "bias": m["weekly"]["net_error"]/m["weeks"],
                "low_mae": m["low_demand"]["mae"] if m["low_demand"] else None,
                "low_weeks": m["low_demand_weeks"],
                "zero_false_positives": m["zero_week_false_positive_total"],
            } for target,m in targets.items()} for output,targets in outputs.items()}))
    print('GATES', json.dumps(gates))
    print('SERVING_FILES_UNCHANGED', True)


if __name__ == '__main__':
    main()
