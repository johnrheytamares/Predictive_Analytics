"""Error metrics for raw forecasts. WAPE is a weighted absolute percentage error, not accuracy."""
from __future__ import annotations

import math
from typing import Sequence


def scores(actual: Sequence[float], predicted: Sequence[float]) -> dict:
    actual = [float(v) for v in actual]
    predicted = [float(v) for v in predicted]
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty and aligned")
    errors = [p - a for a, p in zip(actual, predicted)]
    absolute = sum(abs(e) for e in errors)
    total = sum(actual)
    return {
        "n": len(actual),
        "actual_total": total,
        "predicted_total": sum(predicted),
        "absolute_error": absolute,
        "net_error": sum(errors),
        "mae": absolute / len(errors),
        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
        # Denominator = sum(|actual|). With nonnegative demand this equals sum(actual).
        # Undefined when the denominator is zero.
        "wape": (100.0 * absolute / total) if total > 0 else None,
        "bias": sum(errors) / len(errors),
    }


def summarize_weeks(week_parts: list[tuple[list[float], list[float]]]) -> dict:
    """week_parts: list of (actual_7, predicted_7)."""
    weekly_actual = [sum(a) for a, _ in week_parts]
    weekly_pred = [sum(p) for _, p in week_parts]
    daily_actual = [v for a, _ in week_parts for v in a]
    daily_pred = [v for _, p in week_parts for v in p]
    return {
        "weeks": len(week_parts),
        "weekly": scores(weekly_actual, weekly_pred),
        "daily": scores(daily_actual, daily_pred),
    }
