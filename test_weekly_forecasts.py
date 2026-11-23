import unittest
import numpy as np
import pandas as pd
from evaluate_weekly_forecasts import summarize, weekly_windows


class WeeklyBacktestTests(unittest.TestCase):
    def test_weekly_cancellation_is_not_reported_as_daily_accuracy(self):
        result = summarize([(np.array([1., 1.]), np.array([0., 2.]))])
        self.assertEqual(result["weekly_totals"]["wape"], 0)
        self.assertEqual(result["daily"]["wape"], 100)

    def test_zero_weeks_retain_false_positive_errors(self):
        result = summarize([(np.zeros(7), np.ones(7)), (np.ones(7), np.ones(7))])
        self.assertEqual(result["zero_demand_weeks"], 1)
        self.assertEqual(result["weekly_totals"]["wape"], 100)
        self.assertIsNone(summarize([(np.zeros(7), np.ones(7))])["weekly_totals"]["wape"])

    def test_windows_are_complete_disjoint_and_after_training_minimum(self):
        data = pd.DataFrame({"demand_date": pd.date_range("2025-12-01", "2026-11-21")})
        windows = list(weekly_windows(data))
        for i, (start, end) in enumerate(windows):
            self.assertEqual(start.dayofweek, 0)
            self.assertEqual((end-start).days, 6)
            self.assertGreaterEqual((start-data.demand_date.min()).days, 120)
            self.assertLessEqual(end, data.demand_date.max())
            if i:
                self.assertEqual((start-windows[i-1][1]).days, 1)


if __name__ == "__main__":
    unittest.main()
