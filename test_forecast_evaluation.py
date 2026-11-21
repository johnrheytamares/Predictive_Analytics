import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluate_forecasts as evaluation
import train_real_model as training
import predict_real as inference


class ForecastEvaluationTests(unittest.TestCase):
    def test_all_zero_holdout_has_undefined_wape(self):
        scores = evaluation.metrics([0, 0], [1, 2])
        self.assertIsNone(scores["wape"])
        self.assertEqual(scores["mae"], 1.5)
        self.assertIsNone(training.compute_metrics(np.zeros(2), np.ones(2))["wape"])

    def test_input_rejects_missing_dates_instead_of_synthesizing_zero_days(self):
        pasted = "| 2026-01-01 | 1 | 2 |\n| 2026-01-03 | 1 | 2 |"
        with patch.object(evaluation.Path, "read_text", return_value=pasted):
            with self.assertRaisesRegex(ValueError, "continuous"):
                evaluation.load_history("unused.txt")

    def test_training_baseline_never_receives_holdout_actuals(self):
        train = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=40), "y": 1.0})
        test = pd.DataFrame({"ds": pd.date_range("2026-02-10", periods=28), "y": 100.0})
        class FakeProphet:
            def fit(self, frame):
                return self
            def predict(self, frame):
                return pd.DataFrame({"yhat": np.ones(len(frame)),
                                     "yhat_lower": np.zeros(len(frame)),
                                     "yhat_upper": np.ones(len(frame)) * 2})
        with patch.object(training, "build_prophet", return_value=FakeProphet()):
            _, baseline, _, _ = training.evaluate_prophet(train, test)
        self.assertEqual(baseline["mae"], 99.0)

    def test_recursive_poisson_features_use_predictions_not_future_actuals(self):
        history = pd.Series(np.ones(120), index=pd.date_range("2026-01-01", periods=120))
        seen = []
        class FakeBoosting:
            def __init__(self, **kwargs):
                pass
            def fit(self, x, y):
                return self
            def predict(self, x):
                seen.append(x[0])
                return np.array([2.0])
        with patch.object(evaluation, "HistGradientBoostingRegressor", FakeBoosting):
            evaluation.poisson_forecast(history, pd.date_range("2026-05-01", periods=2))
        self.assertEqual(seen[0][2], 1.0)
        self.assertEqual(seen[1][2], 2.0)

    def test_selected_baseline_metrics_do_not_report_prophet_coverage(self):
        metadata = {"mae": 99, "targets": {"bookings": {
            "selected_model": "weekday_baseline",
            "metrics": {"mae": 99, "prediction_interval_coverage": 95},
            "baseline_metrics": {"mae": 1, "rmse": 2, "wape": 50, "bias": 0},
        }}}
        rows = [{"prediction": 1, "lower": 0, "upper": 2}]
        with patch.object(inference, "get_model_bundles", return_value=({}, {})), \
             patch.object(inference, "predict_series", return_value=rows):
            result = inference.generate_forecast(pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-01"), metadata)
        self.assertEqual(result["model"]["metrics"]["mae"], 1)
        self.assertIsNone(result["model"]["metrics"]["prediction_interval_coverage"])

    def test_zero_holdout_selection_uses_mae(self):
        data = pd.DataFrame({"demand_date": pd.date_range("2026-01-01", periods=150), "bookings": 0})
        with patch.object(training, "evaluate_prophet", return_value=(
            {"wape": 0, "mae": 2}, {"wape": 0, "mae": 1}, 0, None
        )), patch.object(training, "train_full_model", return_value=object()):
            _, result = training.train_target(data, "bookings", 28)
        self.assertEqual(result["selected_model"], "weekday_baseline")
        self.assertEqual(result["selection_metric"], "mae")


if __name__ == "__main__":
    unittest.main()
