import unittest
from datetime import date, timedelta
from experiment import forecast, scores


class BaselineExperimentTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = date(2026, 4, 6)
        self.data = {}
        for lag, value in ((4, 1), (3, 2), (2, 3), (1, 4)):
            for i in range(7):
                self.data[self.cutoff + timedelta(days=i, weeks=-lag)] = (value, value)

    def test_predefined_weight_direction_and_two_week_lookback(self):
        self.assertEqual(forecast(self.data, self.cutoff, 0, "current_4week"), [2.5] * 7)
        self.assertEqual(forecast(self.data, self.cutoff, 0, "short_2week"), [3.5] * 7)
        self.assertEqual(forecast(self.data, self.cutoff, 0, "weighted_4week"), [3.0] * 7)

    def test_future_actuals_cannot_change_forecasts(self):
        for method in ("current_4week", "short_2week", "weighted_4week"):
            before = forecast(self.data, self.cutoff, 0, method)
            changed = dict(self.data)
            for i in range(14):
                changed[self.cutoff + timedelta(days=i)] = (99999, 99999)
            self.assertEqual(before, forecast(changed, self.cutoff, 0, method))

    def test_zero_history_is_not_filtered(self):
        data = {key: (0, 0) for key in self.data}
        for method in ("current_4week", "short_2week", "weighted_4week"):
            self.assertEqual(forecast(data, self.cutoff, 1, method), [0] * 7)

    def test_zero_demand_errors_are_retained_and_wape_undefined_if_all_zero(self):
        self.assertIsNone(scores([0], [2])["wape"])
        self.assertEqual(scores([0, 2], [2, 2])["wape"], 100)


if __name__ == "__main__":
    unittest.main()
