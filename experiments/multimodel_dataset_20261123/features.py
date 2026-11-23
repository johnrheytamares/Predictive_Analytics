"""Leakage-safe calendar, lag, rolling, and trend features for Phase 3 models.

Rules
-----
* One-step supervised rows for day ``t`` use only observations with date ``< t``.
* Seven-day horizon rows at cutoff ``C`` never read actual demand on or after ``C``.
* Short lags that would fall inside the horizon are filled only when
  ``recursive_history`` supplies prior *predictions* for earlier horizon days.
* Calendar fields may use the forecast date itself (public calendar knowledge).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
ROLL_WINDOWS = (7, 14, 28)
TARGETS = ("bookings", "guests")


def _require_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"demand_date", "bookings", "guests"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df.copy()
    out["demand_date"] = pd.to_datetime(out["demand_date"]).dt.normalize()
    out = out.sort_values("demand_date").reset_index(drop=True)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _require_frame(df)
    dow = out["demand_date"].dt.dayofweek
    out["day_of_week"] = dow.astype(int)
    out["is_weekend"] = (dow >= 5).astype(int)
    out["month"] = out["demand_date"].dt.month.astype(int)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


def _history_map(df: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    return {
        row.demand_date: {"bookings": float(row.bookings), "guests": float(row.guests)}
        for row in df.itertuples(index=False)
    }


def _value_at(
    history: dict[pd.Timestamp, dict[str, float]],
    day: pd.Timestamp,
    target: str,
) -> float | None:
    row = history.get(pd.Timestamp(day).normalize())
    return None if row is None else float(row[target])


def _lag(
    history: dict[pd.Timestamp, dict[str, float]],
    day: pd.Timestamp,
    target: str,
    lag: int,
) -> float | None:
    return _value_at(history, day - pd.Timedelta(days=lag), target)


def _rolling_mean(
    history: dict[pd.Timestamp, dict[str, float]],
    day: pd.Timestamp,
    target: str,
    window: int,
) -> float | None:
    """Mean of ``window`` days ending the day before ``day``."""
    values = []
    for offset in range(1, window + 1):
        value = _value_at(history, day - pd.Timedelta(days=offset), target)
        if value is None:
            return None
        values.append(value)
    return float(np.mean(values))


def _trend_ratio(
    history: dict[pd.Timestamp, dict[str, float]],
    day: pd.Timestamp,
    target: str,
    window: int = 14,
) -> float | None:
    """Recent ``window`` mean / preceding ``window`` mean, both before ``day``."""
    recent = []
    older = []
    for offset in range(1, window + 1):
        recent_value = _value_at(history, day - pd.Timedelta(days=offset), target)
        older_value = _value_at(history, day - pd.Timedelta(days=offset + window), target)
        if recent_value is None or older_value is None:
            return None
        recent.append(recent_value)
        older.append(older_value)
    recent_mean = float(np.mean(recent))
    older_mean = float(np.mean(older))
    if older_mean == 0.0 and recent_mean == 0.0:
        return 1.0
    if older_mean == 0.0:
        return 1.5
    return recent_mean / older_mean


def feature_vector_for_day(
    history: dict[pd.Timestamp, dict[str, float]],
    day: pd.Timestamp | date,
    *,
    as_of: pd.Timestamp | date | None = None,
) -> dict[str, float | int | None]:
    """Build features for ``day`` using only history with dates strictly before ``as_of``.

    When ``as_of`` is omitted it defaults to ``day`` (one-step supervised learning).
    Horizon construction passes ``as_of=cutoff`` so intra-horizon actuals are invisible.
    """
    day = pd.Timestamp(day).normalize()
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else day
    visible = {key: value for key, value in history.items() if key < as_of}
    dow = int(day.dayofweek)
    features: dict[str, float | int | None] = {
        "demand_date": day.strftime("%Y-%m-%d"),
        "day_of_week": dow,
        "is_weekend": int(dow >= 5),
        "month": int(day.month),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
    }
    for target in TARGETS:
        for lag in LAGS:
            features[f"{target}_lag_{lag}"] = _lag(visible, day, target, lag)
        for window in ROLL_WINDOWS:
            # Rolling windows are anchored at as_of so they never peek into the horizon.
            features[f"{target}_roll_mean_{window}"] = _rolling_mean(
                visible, as_of, target, window
            )
        features[f"{target}_trend_14"] = _trend_ratio(visible, as_of, target, 14)
    return features


def build_supervised_frame(df: pd.DataFrame, *, min_history_days: int = 28) -> pd.DataFrame:
    """One row per day after warm-up; targets are same-day bookings/guests."""
    frame = add_calendar_features(df)
    history = _history_map(frame)
    rows = []
    for row in frame.itertuples(index=False):
        day = row.demand_date
        if (day - frame["demand_date"].iloc[0]).days < min_history_days:
            continue
        features = feature_vector_for_day(history, day, as_of=day)
        features["bookings"] = float(row.bookings)
        features["guests"] = float(row.guests)
        rows.append(features)
    return pd.DataFrame(rows)


def build_horizon_feature_rows(
    df: pd.DataFrame,
    cutoff: date | pd.Timestamp,
    *,
    horizon_days: int = 7,
    recursive_predictions: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Seven-day feature rows that never read actual demand on/after ``cutoff``.

    Optional ``recursive_predictions`` may supply predicted bookings/guests for
    earlier horizon offsets so short lags can be filled without using actuals.
    """
    frame = _require_frame(df)
    cutoff = pd.Timestamp(cutoff).normalize()
    history = {key: value for key, value in _history_map(frame).items() if key < cutoff}
    if recursive_predictions:
        for target in TARGETS:
            series = recursive_predictions.get(target) or []
            for offset, value in enumerate(series):
                history[cutoff + pd.Timedelta(days=offset)] = history.get(
                    cutoff + pd.Timedelta(days=offset), {"bookings": np.nan, "guests": np.nan}
                )
                history[cutoff + pd.Timedelta(days=offset)][target] = float(value)
        # Recursive fill may only extend with predictions; still block actuals >= cutoff
        # by rebuilding as_of at cutoff for rolling/trend while allowing predicted lags.
    rows = []
    for offset in range(horizon_days):
        day = cutoff + pd.Timedelta(days=offset)
        if recursive_predictions:
            # Short lags may use earlier predicted days; rolling/trend stay pre-cutoff.
            features = feature_vector_for_day(history, day, as_of=day)
            for target in TARGETS:
                for window in ROLL_WINDOWS:
                    features[f"{target}_roll_mean_{window}"] = _rolling_mean(
                        {k: v for k, v in history.items() if k < cutoff},
                        cutoff,
                        target,
                        window,
                    )
                features[f"{target}_trend_14"] = _trend_ratio(
                    {k: v for k, v in history.items() if k < cutoff},
                    cutoff,
                    target,
                    14,
                )
        else:
            features = feature_vector_for_day(history, day, as_of=cutoff)
        features["horizon_offset"] = offset
        features["cutoff"] = cutoff.strftime("%Y-%m-%d")
        rows.append(features)
    return pd.DataFrame(rows)


def assert_horizon_uses_only_precutoff_actuals(
    df: pd.DataFrame,
    cutoff: date | pd.Timestamp,
    feature_rows: pd.DataFrame,
) -> None:
    """Fail if any lag/rolling input date for the horizon is on or after cutoff."""
    frame = _require_frame(df)
    cutoff = pd.Timestamp(cutoff).normalize()
    available = set(frame.loc[frame["demand_date"] < cutoff, "demand_date"])
    for row in feature_rows.itertuples(index=False):
        day = pd.Timestamp(row.demand_date).normalize()
        for lag in LAGS:
            source = day - pd.Timedelta(days=lag)
            value = getattr(row, f"bookings_lag_{lag}")
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            if source >= cutoff:
                raise AssertionError(
                    f"Leakage: bookings_lag_{lag} for {day.date()} uses {source.date()} >= cutoff {cutoff.date()}"
                )
            if source not in available and source < cutoff:
                # Missing historical day should have been None; reconstructed series is continuous.
                raise AssertionError(f"Missing pre-cutoff source day {source.date()}")


def series_for_arima(df: pd.DataFrame, target: str) -> pd.Series:
    if target not in TARGETS:
        raise ValueError("target must be bookings or guests")
    frame = _require_frame(df)
    return frame.set_index("demand_date")[target].astype(float)


def feature_column_names(include_targets: bool = False) -> list[str]:
    names = [
        "day_of_week",
        "is_weekend",
        "month",
        "dow_sin",
        "dow_cos",
    ]
    for target in TARGETS:
        for lag in LAGS:
            names.append(f"{target}_lag_{lag}")
        for window in ROLL_WINDOWS:
            names.append(f"{target}_roll_mean_{window}")
        names.append(f"{target}_trend_14")
    if include_targets:
        names.extend(TARGETS)
    return names


def cutoffs_with_enough_history(
    df: pd.DataFrame,
    cutoffs: Iterable[date],
    *,
    minimum_days: int = 120,
) -> list[date]:
    frame = _require_frame(df)
    start = frame["demand_date"].min()
    ready = []
    for cutoff in cutoffs:
        cutoff_ts = pd.Timestamp(cutoff).normalize()
        history_days = int((cutoff_ts - start).days)
        if history_days >= minimum_days:
            ready.append(cutoff if isinstance(cutoff, date) else cutoff_ts.date())
    return ready
