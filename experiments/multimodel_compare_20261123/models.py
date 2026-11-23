"""Raw forecasting models for the isolated multimodel comparison."""
from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

PHASE2 = Path(__file__).resolve().parents[1] / "multimodel_dataset_20261123"
sys.path.insert(0, str(PHASE2))
from features import TARGETS, build_supervised_frame, feature_column_names, feature_vector_for_day  # noqa: E402

PREDICTORS = feature_column_names(include_targets=False)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
RF_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": 1,
}
ARIMA_ORDERS = ((1, 0, 0), (1, 0, 1), (1, 1, 1), (2, 1, 1))


def statsmodels_available() -> bool:
    return importlib.util.find_spec("statsmodels") is not None


def _history_maps(frame: pd.DataFrame, cutoff: pd.Timestamp) -> dict[pd.Timestamp, dict[str, float]]:
    history = {}
    for row in frame.itertuples(index=False):
        day = pd.Timestamp(row.demand_date).normalize()
        if day < cutoff:
            history[day] = {"bookings": float(row.bookings), "guests": float(row.guests)}
    return history


def weekday_baseline_forecast(frame: pd.DataFrame, cutoff: date, target: str) -> list[float]:
    """Same-weekday average over the prior 4 weeks, with serving fallbacks."""
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    series = (
        frame.loc[frame["demand_date"] < cutoff_ts]
        .set_index("demand_date")[target]
        .astype(float)
        .sort_index()
    )
    if series.empty:
        raise ValueError("No history before cutoff")
    mean = float(series.mean())
    predicted = []
    for offset in range(7):
        day = cutoff_ts + pd.Timedelta(days=offset)
        prior = [
            float(series.loc[day - pd.Timedelta(days=7 * week)])
            for week in range(1, 5)
            if (day - pd.Timedelta(days=7 * week)) in series.index
        ]
        if not prior:
            weekday_values = series.loc[series.index.dayofweek == day.dayofweek].tail(8)
            prior = weekday_values.astype(float).tolist()
        predicted.append(float(np.mean(prior)) if prior else mean)
    return predicted


def _matrix(rows: pd.DataFrame) -> np.ndarray:
    values = rows[PREDICTORS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training features contain missing or non-finite values")
    return values


def _fit_ridge(train: pd.DataFrame, target: str, alpha: float) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=42)),
        ]
    )
    model.fit(_matrix(train), train[target].to_numpy(dtype=float))
    return model


def _fit_rf(train: pd.DataFrame, target: str, **overrides) -> RandomForestRegressor:
    params = {**RF_CONFIG, **overrides}
    model = RandomForestRegressor(**params)
    model.fit(_matrix(train), train[target].to_numpy(dtype=float))
    return model


def _feature_row(history: dict, day: pd.Timestamp) -> dict:
    features = feature_vector_for_day(history, day, as_of=day)
    missing = [name for name in PREDICTORS if features.get(name) is None]
    if missing:
        raise ValueError(f"Incomplete features for {day.date()}: {missing}")
    return features


def _recursive_pair_forecast(
    frame: pd.DataFrame,
    cutoff: date,
    fit_bookings,
    fit_guests,
) -> dict[str, list[float]]:
    """Forecast both targets day-by-day; only predictions extend the history."""
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    history = _history_maps(frame, cutoff_ts)
    out = {"bookings": [], "guests": []}
    for offset in range(7):
        day = cutoff_ts + pd.Timedelta(days=offset)
        features = _feature_row(history, day)
        x = np.asarray([[features[name] for name in PREDICTORS]], dtype=float)
        bookings = float(fit_bookings.predict(x)[0])
        guests = float(fit_guests.predict(x)[0])
        # Raw forecasts may be negative; keep them raw for Phase 3 scoring.
        out["bookings"].append(bookings)
        out["guests"].append(guests)
        history[day] = {"bookings": bookings, "guests": guests}
    return out


def ridge_forecast(
    frame: pd.DataFrame,
    cutoff: date,
    *,
    alpha_bookings: float,
    alpha_guests: float,
) -> dict[str, list[float]]:
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    train = build_supervised_frame(frame[frame["demand_date"] < cutoff_ts], min_history_days=28)
    if len(train) < 60:
        raise ValueError("Insufficient supervised training rows")
    bookings_model = _fit_ridge(train, "bookings", alpha_bookings)
    guests_model = _fit_ridge(train, "guests", alpha_guests)
    return _recursive_pair_forecast(frame, cutoff, bookings_model, guests_model)


def random_forest_forecast(
    frame: pd.DataFrame,
    cutoff: date,
    **rf_overrides,
) -> dict[str, list[float]]:
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    train = build_supervised_frame(frame[frame["demand_date"] < cutoff_ts], min_history_days=28)
    if len(train) < 60:
        raise ValueError("Insufficient supervised training rows")
    bookings_model = _fit_rf(train, "bookings", **rf_overrides)
    guests_model = _fit_rf(train, "guests", **rf_overrides)
    return _recursive_pair_forecast(frame, cutoff, bookings_model, guests_model)


def arima_forecast(frame: pd.DataFrame, cutoff: date, target: str, order: tuple[int, int, int]) -> list[float]:
    if not statsmodels_available():
        raise RuntimeError(
            "statsmodels is not installed. Approve `pip install statsmodels` "
            "before running ARIMA evaluations."
        )
    from statsmodels.tsa.arima.model import ARIMA

    cutoff_ts = pd.Timestamp(cutoff).normalize()
    series = (
        frame.loc[frame["demand_date"] < cutoff_ts]
        .set_index("demand_date")[target]
        .astype(float)
        .asfreq("D")
    )
    if series.isna().any():
        raise ValueError("ARIMA series has gaps")
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=7)
    return [float(v) for v in forecast.to_numpy()]


def baseline_pair(frame: pd.DataFrame, cutoff: date) -> dict[str, list[float]]:
    return {
        "bookings": weekday_baseline_forecast(frame, cutoff, "bookings"),
        "guests": weekday_baseline_forecast(frame, cutoff, "guests"),
    }
